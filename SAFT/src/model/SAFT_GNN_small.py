import os
import math
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertLayer, BertEmbeddings, BertPreTrainedModel
from src.utils import roc_auc_score, mrr_score, ndcg_score
import scipy.sparse as sp


class SAFTEncoder(nn.Module):
    def __init__(self, config):
        super(SAFTEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.heter_embed_size = config.heter_embed_size
        self.dropout = config.dropout
        self.prop_layers = config.prop_layers
        self.tlambda = config.tlambda
        self.delta = config.delta
        self.num_hidden_layers = config.num_hidden_layers
        self.aggregation = nn.ModuleList([nn.Linear(3 * self.hidden_size, self.hidden_size) for _ in range(config.num_hidden_layers)])
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout2 = nn.Dropout(config.dropout)

    def sparse_prop(self, x):
        indices = []
        weights = []
        for i, row in enumerate(x):
            col_idx1, val1 = row.tolist()
            indices.append(int(col_idx1))
            weights.append(val1)
        k = len(indices)
        result = torch.zeros((k, k), dtype=torch.float32)
        for i in range(k):
            for j in range(k):
                if indices[i] == indices[j]:
                    result[i, j] = weights[i] * weights[j]
                else:
                    result[i, j] = 0.0
        return result

    def propagation(self, token_encodings, propogation_matrix):
        encoder_outputs = token_encodings
        for j in range(self.prop_layers):
            temp = torch.mm(propogation_matrix, token_encodings)
            temp = temp + self.delta * encoder_outputs
            token_encodings = temp
        return token_encodings

    def forward(self, hidden_states, attention_mask, query_embedding, key_embedding, user_operator, item_operator, distance_embedding, centrality_embedding):
        all_hidden_states = ()
        all_attentions = ()
        all_nodes_num, seq_length, emb_dim = hidden_states.shape
        user_operator = torch.mm(user_operator, user_operator.t())
        item_operator = torch.mm(item_operator, item_operator.t())
        sub_next_q = torch.zeros(all_nodes_num, emb_dim, dtype=torch.float32).to(user_operator.device)
        sub_next_k = torch.zeros(all_nodes_num, emb_dim, dtype=torch.float32).to(user_operator.device)

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if i > 0:
                hidden_states[:, 0] = query_embedding
                hidden_states[:, 1] = key_embedding
                hidden_states[:, 2] = distance_embedding
                hidden_states[:, 3] = centrality_embedding
            else:
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:, :, :, :4] = -10000.0
            sub_tensor_q = query_embedding
            sub_tensor_q = self.tlambda * sub_tensor_q + sub_next_q
            temp_p1 = self.propagation(sub_tensor_q, user_operator)
            temp_p1 = self.dense1(temp_p1)
            temp_p1 = self.dropout1(temp_p1)
            temp_p1 = self.LayerNorm1(temp_p1)
            sub_tensor_k = key_embedding
            sub_tensor_k = self.tlambda * sub_tensor_k + sub_next_k
            temp_p2 = self.propagation(sub_tensor_k, item_operator)
            temp_p2 = self.dense2(temp_p2)
            temp_p2 = self.dropout2(temp_p2)
            temp_p2 = self.LayerNorm2(temp_p2)

            if i > 0:
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask)
            hidden_states = layer_outputs[0]
            sub_next_tensor = hidden_states[:, 4, :].clone()
            sub_next_q = hidden_states[:, 0, :].clone()
            sub_next_k = hidden_states[:, 1, :].clone()
            sub_next_tensor = F.normalize(sub_next_tensor, p=2, dim=1)
            p = torch.cat((sub_next_tensor, temp_p1, temp_p2), dim=1)
            temp_outputs = self.aggregation[i](p)
            temp_outputs = F.relu(temp_outputs, inplace=True)
            temp_outputs = F.normalize(temp_outputs, p=2, dim=1)
            hidden_states[:, 4, :] = temp_outputs
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        out1 = all_hidden_states[self.num_hidden_layers - 1][:, 4:, :].mean(dim=1, keepdim=True).squeeze(1)
        out2 = all_hidden_states[self.num_hidden_layers][:, 4:, :].mean(dim=1, keepdim=True).squeeze(1)
        encoder_outputs = torch.mean(torch.stack([out1, out2], dim=0), dim=0, keepdim=True).squeeze(0)
        return encoder_outputs


class SAFT(BertPreTrainedModel):
    def __init__(self, config):
        super(SAFT, self).__init__(config=config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = SAFTEncoder(config=config)
        self.rd = config.rd
        self.heter_embed_size = config.heter_embed_size
        self.pe_dim = config.pe_dim
        self.dense_distance_embedding = nn.Linear(config.pe_dim, config.hidden_size)
        self.dense_centrality = nn.Linear(config.pe_dim, config.hidden_size)

    def init_node_embed(self, pretrain_embed, pretrain_mode, pretrain_dir, node_num, heter_embed_size):
        self.node_num = node_num
        self.heter_embed_size = heter_embed_size
        if not pretrain_embed:
            self.node_embedding = nn.Parameter(torch.FloatTensor(self.node_num, self.heter_embed_size))
            torch.manual_seed(self.rd)
            nn.init.xavier_normal_(self.node_embedding)
            self.node_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
        else:
            if pretrain_mode == 'BERTMF':
                checkpoint = pickle.load(open(os.path.join(pretrain_dir, f'BERTMF_{heter_embed_size}.pt'), 'rb'))
                self.node_embedding = nn.Parameter(checkpoint['author_embeddings'])
                self.node_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
                with torch.no_grad():
                    self.node_to_text_transform.weight.copy_(checkpoint['linear.weight'])
                    self.node_to_text_transform.bias.copy_(checkpoint['linear.bias'])
            elif pretrain_mode == 'MF':
                checkpoint = torch.load(os.path.join(pretrain_dir, f'MF_{heter_embed_size}.pt'), map_location='cpu')
                self.node_embedding = nn.Parameter(checkpoint['node_embedding'])
                self.node_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            elif pretrain_mode == 'MYEM':
                checkpoint = torch.load(os.path.join(pretrain_dir, f'MYEM{heter_embed_size}.pt'), map_location='cpu')
                self.node_embedding = nn.Parameter(checkpoint['node_embedding'])
                self.node_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            else:
                raise ValueError('Wrong pretrain mode!')

    def forward(self, input_ids, attention_mask, query_node_idx, key_node_idx, user_operator, item_operator, distance_embedding, centrality_embedding):
        all_nodes_num, seq_length = input_ids.shape
        embedding_output = self.embeddings(input_ids=input_ids)
        query_node_embed = self.node_to_text_transform(self.node_embedding[query_node_idx])
        key_node_embed = self.node_to_text_transform(self.node_embedding[key_node_idx])
        distance_embedding = self.dense_distance_embedding(distance_embedding)
        centrality_embedding = self.dense_centrality(centrality_embedding)
        sentence_embed = embedding_output.mean(dim=1, keepdim=True)
        station_mask = torch.ones((all_nodes_num, 5), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        station_placeholder = torch.zeros(all_nodes_num, 4, embedding_output.size(-1)).type(embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, sentence_embed, embedding_output], dim=1)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            query_embedding=query_node_embed,
            key_embedding=key_node_embed,
            user_operator=user_operator,
            item_operator=item_operator,
            distance_embedding=distance_embedding,
            centrality_embedding=centrality_embedding)
        return encoder_outputs


class SAFTClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = SAFT(config)
        self.hidden_size = config.hidden_size
        self.heter_embed_size = config.heter_embed_size
        self.init_weights()
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.class_num), nn.Softmax(dim=1))
        self.loss_func = nn.BCELoss()

    def init_node_embed(self, pretrain_embed, pretrain_mode, pretrain_dir):
        self.bert.init_node_embed(pretrain_embed, pretrain_mode, pretrain_dir, self.node_num, self.heter_embed_size)

    def infer(self, token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx, user_operator, item_operator, distance_embedding, centrality_embedding):
        edge_embeddings = self.bert(token_edges_batch.squeeze(1), attention_edges_batch.squeeze(1), query_node_idx, key_node_idx, user_operator, item_operator, distance_embedding, centrality_embedding)
        return edge_embeddings

    def test(self, token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx, edge_labels, user_operator, item_operator, distance_embedding, centrality_embedding, **kwargs):
        edge_embeddings = self.infer(token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx, user_operator, item_operator, distance_embedding, centrality_embedding)
        scores = self.classifier(edge_embeddings)
        label_id = torch.argmax(edge_labels, 1)
        return scores, label_id

    def forward(self, token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx, edge_labels, user_operator, item_operator, distance_embedding, centrality_embedding, **kwargs):
        edge_embeddings = self.infer(token_edges_batch, attention_edges_batch, query_node_idx, key_node_idx, user_operator, item_operator, distance_embedding, centrality_embedding)
        logit = self.classifier(edge_embeddings)
        loss = self.loss_func(logit, edge_labels)
        return loss
