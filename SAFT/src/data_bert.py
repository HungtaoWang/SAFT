import os
import sys
import math
import pickle
import random
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from torch.utils.data.dataset import Dataset, TensorDataset

import networkx as nx
import scipy
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import diags, eye, csr_matrix


logger = logging.getLogger(__name__)

def generate_approximate_dataset(filtered_indices, edges):
    result = []
    for i in range(len(edges)):
        new_row = [edges[i]]
        for idx in filtered_indices[i]:
            new_row.append(edges[idx])
        result.append(new_row)
    return result

def extract_indices_and_values(sparse_matrix):
    col_indices = sparse_matrix.indices
    values = sparse_matrix.data
    
    result_list = [(col_indices[i], values[i]) for i in range(len(values))]
    return result_list

def find_neighbors(edges, vectors, b, metric_func):
    u_edges, v_edges = {}, {}
    for idx, (u, v) in enumerate(edges):
        u_edges.setdefault(u, []).append(idx)
        v_edges.setdefault(v, []).append(idx)

    def sample_neighbors(neighbors, idx, current_vector):
        if len(neighbors) >= b:
            neighbor_vectors = np.array([vectors[i] for i in neighbors])
            best_indices = np.argsort(metric_func(neighbor_vectors, current_vector))[:b]
            return [neighbors[i] for i in best_indices]
        return neighbors + [idx] * (b - len(neighbors))

    sampled_neighbors_u = [
        sample_neighbors([i for i in u_edges[u] if i != idx], idx, vectors[idx])
        for idx, (u, _) in enumerate(edges)]
    sampled_neighbors_v = [
        sample_neighbors([i for i in v_edges[v] if i != idx], idx, vectors[idx])
        for idx, (_, v) in enumerate(edges)]

    return sampled_neighbors_u, sampled_neighbors_v

def dot_product_metric(vectors, current_vector):
    return np.dot(vectors, current_vector)

def l2_norm_metric(vectors, current_vector):
    return np.linalg.norm(vectors, axis=1)

def find_min_dot_product_neighbors(edges, x_vectors, b):
    return find_neighbors(edges, x_vectors, b, dot_product_metric)

def find_max_l2_norm_neighbors(edges, y_vectors, b):
    return find_neighbors(edges, y_vectors, b, l2_norm_metric)

def combined_neighbors(edges, x_vectors, y_vectors, b):
    sampled_neighbors_u_1, sampled_neighbors_v_1 = find_min_dot_product_neighbors(edges, x_vectors, b)
    sampled_neighbors_u_2, sampled_neighbors_v_2 = find_max_l2_norm_neighbors(edges, y_vectors, b)
    
    indices_d = [u1 + v1 for u1, v1 in zip(sampled_neighbors_u_1, sampled_neighbors_v_1)]
    indices_c = [u2 + v2 for u2, v2 in zip(sampled_neighbors_u_2, sampled_neighbors_v_2)]
    
    return indices_d, indices_c

def propagation_operators(args, edgelist):
    user_id_map = {user_id: idx for idx, user_id in enumerate({edge[0] for edge in edgelist})}
    item_id_map = {item_id: idx for idx, item_id in enumerate({edge[1] for edge in edgelist})}
    num_users, num_items = len(user_id_map), len(item_id_map)
    user_data = item_data = [1] * len(edgelist)
    D1_data, D2_data = np.ones(num_users), np.ones(num_items)
    user_id_row, user_id_col, item_id_row, item_id_col = [], [], [], []

    for idx, (user_id, item_id) in enumerate(edgelist):
        user_idx, item_idx = user_id_map[user_id], item_id_map[item_id]
        user_id_row.append(idx); user_id_col.append(user_idx)
        item_id_row.append(idx); item_id_col.append(item_idx)
        D1_data[user_idx] += 1; D2_data[item_idx] += 1

    E1 = csr_matrix((user_data, (user_id_row, user_id_col)), shape=(len(edgelist), num_users))
    E2 = csr_matrix((item_data, (item_id_row, item_id_col)), shape=(len(edgelist), num_items))
    inverse_sqr_matrix_1 = diags(np.reciprocal(np.sqrt(D1_data)))
    inverse_sqr_matrix_2 = diags(np.reciprocal(np.sqrt(D2_data)))

    user_propagation, item_propagation = E1.dot(inverse_sqr_matrix_1), E2.dot(inverse_sqr_matrix_2)
    if args.large:
        return extract_indices_and_values(user_propagation), extract_indices_and_values(item_propagation)
    return user_propagation, item_propagation

def centrality_encoding(args, edges):
    n, m = args.user_num + args.item_num, len(edges)
    data, row_ind, col_ind = [], [], []

    for x, (i, j) in enumerate(edges):
        row_ind.extend([i, j]); col_ind.extend([x, x])
        data.extend([1, -1])

    B = csr_matrix((data, (row_ind, col_ind)), shape=(n, m))
    U, Sigma, VT = randomized_svd(B, n_components=args.pe_dim + 1, flip_sign=True, random_state=0)

    return VT[1:, :].T

def distance_encoding(args, graph, es):

    E = nx.incidence_matrix(graph, edgelist=es, oriented=False) 

    degree_vector_graph = np.array([d for n, d in graph.degree()])

    inverse_sqrt_degree_matrix = diags(np.reciprocal(np.sqrt(degree_vector_graph).astype(float)))

    target_matrix = inverse_sqrt_degree_matrix.dot(E)

    U, Sigma, VT = randomized_svd(target_matrix, n_components=args.pe_dim + 1, flip_sign=True, random_state=0)

    U = U[:, 1:]
    Sigma = Sigma[1:]
    V = VT[1:, :].T

    Sigma_squared = Sigma ** 2
    result_diag = 1 - Sigma_squared / 2
    result_diag_inv = 1 / result_diag
    denominator_diag = np.sqrt(result_diag_inv)
    denominator = diags(denominator_diag)

    distance_embedding = V @ denominator

    return distance_embedding

def load_dataset_bert(args, tokenizer, evaluate=False, test=False):
    assert args.data_mode in ['bert']
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    def cached_file(mode):
        return os.path.join(args.data_path, f'cached-gpu{args.gpu_number}-_{args.data_mode}_{mode}_'
                            f'{list(filter(None, args.model_name_or_path.split("/"))).pop()}_{args.max_length}')

    cached_files = {"train": cached_file('train'), "val": cached_file('val'), "test": cached_file('test')}
    state = 'test' if test else 'val' if evaluate else 'train'

    if os.path.exists(cached_files["train"]):
        if args.local_rank in [-1, 0]:
            logger.info(f"Loading features from cached file {cached_files['train']}")
        features_train, features_val, features_test = (
            pickle.load(open(cached_files["train"], 'rb')),
            pickle.load(open(cached_files["val"], 'rb')),
            pickle.load(open(cached_files["test"], 'rb'))
        )
    else:
        if args.local_rank in [-1, 0]:
            logger.info("Creating features from dataset file at %s", args.data_path)
        features_train, features_val, features_test = read_process_data_bert(args, tokenizer, args.max_length, state)
        args.train_number = torch.LongTensor(features_train[0][0]).shape[0]
        args.val_number = torch.LongTensor(features_val[0][0]).shape[0]
        args.test_number = torch.LongTensor(features_test[0][0]).shape[0]
        for mode in ["train", "val", "test"]:
            pickle.dump(locals()[f"features_{mode}"], open(cached_files[mode], 'wb'))

    if args.local_rank == 0:
        torch.distributed.barrier()

    def convert_to_tensor(features):
        token_query_edges, attention_query_edges = torch.LongTensor(features[0][0]), torch.LongTensor(features[0][1])
        query_node, key_node = torch.LongTensor(features[1]), torch.LongTensor(features[2]) + args.user_num
        edge_labels_id = torch.LongTensor(features[3])
        edge_labels = torch.zeros(query_node.shape[0], args.class_num).scatter_(1, edge_labels_id.unsqueeze(-1), 1)
        return (token_query_edges, attention_query_edges, query_node, key_node, edge_labels,
                torch.Tensor(features[4]), torch.Tensor(features[5]), torch.Tensor(features[6]),
                torch.Tensor(features[7]))

    dataset_train = TensorDataset(*convert_to_tensor(features_train))
    dataset_val = TensorDataset(*convert_to_tensor(features_val))
    dataset_test = TensorDataset(*convert_to_tensor(features_test))

    return dataset_test if state == 'test' else (dataset_train, dataset_val, dataset_test)

def count_lines_in_file(file_path):
    with open(file_path) as f:
        return sum(1 for _ in f)

def read_file_lines(*file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path) as f:
            data.extend(f.readlines())
    return data

def process_line(line, tokenizer, max_length):
    a = line.strip().split('\\$\\$')
    if len(a) != 4:
        raise ValueError(f"Expected 4 fields, but got {len(a)}: {a}")
    text, query_n, key_n, label = a
    encoded_text = tokenizer.batch_encode_plus([text], max_length=max_length, padding='max_length', truncation=True)
    return encoded_text, int(query_n), int(key_n), int(label)

def prepare_graph_and_edge_ids(query_node_idx, key_node_idx, user_num):
    key_node_idx_temp = [key_node + user_num for key_node in key_node_idx]
    edge_ids_temp, edge_ids = list(zip(query_node_idx, key_node_idx_temp)), list(zip(query_node_idx, key_node_idx))
    return edge_ids_temp, edge_ids

def split_dataset(data, train_count, val_count, test_count):
    return data[:train_count], data[train_count:train_count + val_count], data[train_count + val_count:]

def read_process_data_bert(args, tokenizer, max_length, state):
    token_edges, attention_edges, edge_labels = [], [], []
    query_node_idx, key_node_idx = [], []
    train_count = count_lines_in_file(os.path.join(args.data_path, 'train.tsv'))
    val_count = count_lines_in_file(os.path.join(args.data_path, 'val.tsv'))
    test_count = count_lines_in_file(os.path.join(args.data_path, 'test.tsv'))

    data = read_file_lines(
        os.path.join(args.data_path, 'train.tsv'),
        os.path.join(args.data_path, 'val.tsv'),
        os.path.join(args.data_path, 'test.tsv'))

    for line in tqdm(data):
        encoded_text, query_n, key_n, label = process_line(line, tokenizer, max_length)
        token_edges.append(encoded_text['input_ids'])
        attention_edges.append(encoded_text['attention_mask'])
        edge_labels.append(label)
        query_node_idx.append(query_n)
        key_node_idx.append(key_n)

    edge_ids_temp, edge_ids = prepare_graph_and_edge_ids(
        query_node_idx, key_node_idx, args.user_num)
    G = nx.MultiGraph()
    G.add_edges_from(edge_ids_temp)

    distance_embedding = distance_encoding(args, G, edge_ids_temp)
    centrality_embedding = centrality_encoding(args, edge_ids_temp)

    if args.approximate:
        b = args.affiliated_edges
        indices_d, indices_c = combined_neighbors(
            edge_ids_temp, distance_embedding, centrality_embedding, b)
        filtered_indices = indices_d if args.sample_mode == "distance" else indices_c

        token_edges = generate_approximate_dataset(filtered_indices, token_edges)
        attention_edges = generate_approximate_dataset(filtered_indices, attention_edges)
        user_operator, item_operator = propagation_operators(args, edge_ids)
        user_operator = generate_approximate_dataset(filtered_indices, user_operator)
        item_operator = generate_approximate_dataset(filtered_indices, item_operator)
        query_node_idx = generate_approximate_dataset(filtered_indices, query_node_idx)
        key_node_idx = generate_approximate_dataset(filtered_indices, key_node_idx)
        distance_embedding = generate_approximate_dataset(filtered_indices, distance_embedding)
        centrality_embedding = generate_approximate_dataset(filtered_indices, centrality_embedding)
    else:
        user_operator, item_operator = propagation_operators(args, edge_ids)
        user_operator = user_operator.toarray()
        item_operator = item_operator.toarray()

    token_edges_train, token_edges_val, token_edges_test = split_dataset(
        token_edges, train_count, val_count, test_count)
    attention_edges_train, attention_edges_val, attention_edges_test = split_dataset(
        attention_edges, train_count, val_count, test_count)
    edge_labels_train, edge_labels_val, edge_labels_test = split_dataset(
        edge_labels, train_count, val_count, test_count)
    query_node_idx_train, query_node_idx_val, query_node_idx_test = split_dataset(
        query_node_idx, train_count, val_count, test_count)
    key_node_idx_train, key_node_idx_val, key_node_idx_test = split_dataset(
        key_node_idx, train_count, val_count, test_count)
    user_operator_train, user_operator_val, user_operator_test = split_dataset(
        user_operator, train_count, val_count, test_count)
    item_operator_train, item_operator_val, item_operator_test = split_dataset(
        item_operator, train_count, val_count, test_count)
    distance_embedding_train, distance_embedding_val, distance_embedding_test = split_dataset(
        distance_embedding, train_count, val_count, test_count)
    centrality_embedding_train, centrality_embedding_val, centrality_embedding_test = split_dataset(
        centrality_embedding, train_count, val_count, test_count)

    return (
        (token_edges_train, attention_edges_train), query_node_idx_train,
        key_node_idx_train, edge_labels_train, user_operator_train, 
        item_operator_train, distance_embedding_train, centrality_embedding_train), (
        (token_edges_val, attention_edges_val), query_node_idx_val,
        key_node_idx_val, edge_labels_val, user_operator_val, 
        item_operator_val, distance_embedding_val, centrality_embedding_val), (
        (token_edges_test, attention_edges_test), query_node_idx_test,
        key_node_idx_test, edge_labels_test, user_operator_test, 
        item_operator_test, distance_embedding_test, centrality_embedding_test)
