import shutil
import glob
import logging
import os
import pickle
import random
from time import time
from collections import defaultdict
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from src.data_bert import load_dataset_bert
from transformers import BertConfig, BertTokenizerFast, AdamW
from transformers import AutoTokenizer


def cleanup():
    dist.destroy_process_group()


def delete_cached_files(args):
    cached_files_pattern = os.path.join(args.data_path, f'cached-gpu{args.gpu_number}-*')
    cached_files = glob.glob(cached_files_pattern)
    for cached_file in cached_files:
        os.remove(cached_file)
        print(f"Deleted: {cached_file}")


def load_bert(args):
    config = BertConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    config.heter_embed_size = args.heter_embed_size
    config.node_num = args.user_num + args.item_num
    config.class_num = args.class_num
    config.rd = args.rd
    config.dropout = args.dropout
    config.prop_layers = args.prop_layers
    config.tlambda = args.tlambda
    config.delta = args.delta
    config.pe_dim = args.pe_dim

    args.hidden_size = config.hidden_size
    args.node_num = args.user_num + args.item_num

    model_import_paths = {
        'SAFT_GNN_large': 'src.model.SAFT_GNN_large.SAFTClassification',
        'SAFT_GAU_large': 'src.model.SAFT_GAU_large.SAFTClassification',
        'SAFT_GNN_small': 'src.model.SAFT_GNN_small.SAFTClassification',
        'SAFT_GAU_small': 'src.model.SAFT_GAU_small.SAFTClassification'
    }

    if args.model_type in model_import_paths:
        module_path, class_name = model_import_paths[args.model_type].rsplit('.', 1)
        SAFTClassification = getattr(__import__(module_path, fromlist=[class_name]), class_name)
        model = SAFTClassification.from_pretrained(args.model_name_or_path, config=config) if args.pretrain_LM else SAFTClassification(config)
        model.node_num, model.edge_type, model.heter_embed_size = args.user_num + args.item_num, args.class_num, args.heter_embed_size
        model.init_node_embed(args.pretrain_embed, args.pretrain_mode, args.pretrain_dir)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model


def train(args):
    ckpt_dir = os.path.join(args.data_path, 'ckpt-gpu{}'.format(args.gpu_number))
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logging.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.data_mode in ['bert']:
        args.user_num, args.item_num, args.class_num = [pickle.load(open(os.path.join(args.data_path, 'node_num.pkl'), 'rb'))[key] for key in ('user_num', 'item_num', 'class_num')]
        train_set, val_set, test_set = load_dataset_bert(args, tokenizer, evaluate=False, test=False)
    else:
        raise ValueError('Data Mode is Incorrect here!')

    torch.manual_seed(args.rd)
    train_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
    val_sampler = SequentialSampler(val_set) if args.local_rank == -1 else DistributedSampler(val_set)
    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)

    if args.large:
        args.train_batch_size = args.train_batch_size
        args.val_batch_size = args.val_batch_size
        args.test_batch_size = args.test_batch_size
    else:
        args.train_batch_size = args.train_number
        args.val_batch_size = args.val_number
        args.test_batch_size = args.test_number

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)

    print(f'[Process:{args.local_rank}] Dataset Loading Over!')

    model = load_bert(args)
    if args.local_rank in [-1, 0]:
        logging.info('loading model: {}'.format(args.model_type))
    model.to(args.device)

    if args.load:
        model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.local_rank != -1:
        ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    loss = 0.0
    global_step = 0
    best_acc, best_count = 0.0, 0

    for ep in range(args.epochs):
        start_time = time()
        ddp_model.train()
        train_loader_iterator = tqdm(train_loader, desc=f"Epoch:{ep}|Iteration", disable=args.local_rank not in [-1,0])
        for step, batch in enumerate(train_loader_iterator):
            if args.enable_gpu:
                batch = [b.cuda() for b in batch]
            batch_loss = ddp_model(*batch)
            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            global_step += 1

            if args.local_rank in [-1, 0]:
                if global_step % args.log_steps == 0:
                    logging.info('cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(time() - start_time, global_step, optimizer.param_groups[0]['lr'], loss / args.log_steps))
                    loss = 0.0
                if args.local_rank == 0:
                    torch.distributed.barrier()
            else:
                torch.distributed.barrier()

        if args.local_rank in [-1, 0]:
            ckpt_path = os.path.join(args.data_path, 'ckpt-gpu{}'.format(args.gpu_number), '{}-{}-{}-{}-epoch-{}.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size, ep + 1))
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

            logging.info("Start validation for epoch-{}".format(ep + 1))
            acc = validate(args, model, val_loader)

            logging.info("validation time:{}".format(time() - start_time))
            if acc > best_acc:
                ckpt_path = os.path.join(args.data_path, 'ckpt-gpu{}'.format(args.gpu_number), '{}-{}-{}-{}-best.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size))
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")
                best_acc = acc
                best_count = 0
            else:
                best_count += 1
                if best_count >= args.early_stop:
                    start_time = time()
                    ckpt_path = os.path.join(args.data_path, 'ckpt-gpu{}'.format(args.gpu_number), '{}-{}-{}-{}-best.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size))
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    logging.info("Start testing for best")
                    acc = validate(args, model, test_loader)
                    logging.info("test time:{}".format(time() - start_time))
                    return acc
            if args.local_rank == 0:
                torch.distributed.barrier()
        else:
            torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        start_time = time()
        ckpt_path = os.path.join(args.data_path, 'ckpt-gpu{}'.format(args.gpu_number), '{}-{}-{}-{}-best.pt'.format(args.model_type, args.pretrain_LM, args.lr, args.heter_embed_size))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logging.info('load ckpt:{}'.format(ckpt_path))
        acc = validate(args, model, test_loader)
        logging.info("test time:{}".format(time() - start_time))
        if args.local_rank == 0:
            torch.distributed.barrier()
        return acc
    else:
        torch.distributed.barrier()
    if args.local_rank != -1:
        cleanup()


@torch.no_grad()
def validate(args, model, dataloader):
    model.eval()

    metrics_total = defaultdict(float)
    for step, batch in enumerate(tqdm(dataloader)):
        if args.enable_gpu:
            batch = [b.cuda() for b in batch]

        score, label = model.test(*batch)
        pred = torch.argmax(score, 1)

        if step == 0:
            preds = np.copy(pred.cpu())
            labels = np.copy(label.cpu())
            scores = np.copy(score.cpu())
        else:
            preds = np.concatenate((preds, pred.cpu()), 0)
            labels = np.concatenate((labels, label.cpu()), 0)
            scores = np.concatenate((scores, score.cpu()), 0)

    metrics_total['F1_macro'] = f1_score(labels, preds, average='macro')
    metrics_total['F1_micro'] = f1_score(labels, preds, average='micro')

    if args.tmetric == 'macro':
        metrics_total['main'] = metrics_total['F1_macro']
    elif args.tmetric == 'micro':
        metrics_total['main'] = metrics_total['F1_micro']

    logging.info("{}:{}".format('main', metrics_total['main']))
    logging.info("{}:{}".format('F1_macro', metrics_total['F1_macro']))
    logging.info("{}:{}".format('F1_micro', metrics_total['F1_micro']))

    return metrics_total['main']


def test(args):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if args.data_mode in ['bert']:
        args.user_num, args.item_num, args.class_num = [pickle.load(open(os.path.join(args.data_path, 'node_num.pkl'), 'rb'))[key] for key in ('user_num', 'item_num', 'class_num')]
        test_set = load_dataset_bert(args, tokenizer, evaluate=True, test=True)
    else:
        raise ValueError('Data Mode is Incorrect here!')

    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print('Dataset Loading Over!')

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    start_time = time()
    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    validate(args, model, test_loader)
    logging.info("test time:{}".format(time() - start_time))


def run(args):
    args.gpu_number = os.getenv('CUDA_VISIBLE_DEVICES')

    if args.large:
        delete_cached_files(args)
        if args.mode == "train":
            train(args)
        elif args.mode == "test":
            test(args)
    elif not args.large:
        delete_cached_files(args)
        if args.mode == "train":
            train(args)
        elif args.mode == "test":
            test(args)
