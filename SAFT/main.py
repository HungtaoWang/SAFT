import os
import logging
import argparse
from pathlib import Path
from src.run import train, test, run
from src.utils import setuplogging, str2bool, set_seed

parser = argparse.ArgumentParser(description='Experiments for SAFT: Structure-aware Transformers for Textual Interaction Classification.')
parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
parser.add_argument("--data_path", type=str, default="Apps/")
parser.add_argument("--data_mode", default="bert", type=str, choices=['bert'])
parser.add_argument("--pretrain_embed", type=str2bool, default=False)
parser.add_argument("--pretrain_dir", default="movie/pretrain", type=str, choices=['movie/pretrain'])
parser.add_argument("--pretrain_mode", default="MF", type=str, choices=['MF','BERTMF'])
parser.add_argument("--model_type", default="SAFT_GNN_small", type=str, choices=['SAFT_GNN_small', 'SAFT_GAU_small', 'SAFT_GNN_large', 'SAFT_GAU_large'])
parser.add_argument("--pretrain_LM", type=str2bool, default=True)
parser.add_argument("--heter_embed_size", type=int, default=64)
parser.add_argument("--max_length", type=int, default=64)
parser.add_argument("--train_batch_size", type=int, default=25)
parser.add_argument("--val_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stop", type=int, default=3)
parser.add_argument("--log_steps", type=int, default=100)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--rd", type=int, default=42)
parser.add_argument("--load", type=str2bool, default=False)
parser.add_argument("--max_grad_norm", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--adam_epsilon", type=float, default=1e-8)
parser.add_argument("--enable_gpu", type=str2bool, default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--prop_layers', type=int, default=3, help='Prop_layers for GNN')
parser.add_argument('--pe_dim', type=int, default=64, help='pe_dim')
parser.add_argument('--large', type=str2bool, default=False, help='large or small for dataset and model.')
parser.add_argument('--tlambda', type=float, default=0.4, help='tlambda')
parser.add_argument('--delta', type=float, default=0.4, help='delta')
parser.add_argument("--tmetric", default="macro", type=str, choices=['macro', 'micro', 'ap', 'auc'])
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--train_number', type=int, default=0)
parser.add_argument('--val_number', type=int, default=0)
parser.add_argument('--test_number', type=int, default=0)
parser.add_argument('--approximate', type=str2bool, default=False)
parser.add_argument('--affiliated_edges', type=int, default=1)
parser.add_argument("--sample_mode", default="distance", type=str, choices=['distance', 'centrality'])
parser.add_argument("--model_name_or_path", default="prajjwal1/bert-tiny", type=str, help="Path to pre-trained model or shortcut name.")
parser.add_argument("--load_ckpt_name", type=str, help="choose which ckpt to load and test")
parser.add_argument("--fp16", type=str2bool, default=True)

args = parser.parse_args()

assert args.data_mode == 'bert'

if args.local_rank in [-1, 0]:
    logging.info(args)
    print(args)

if __name__ == "__main__":
    set_seed(args.random_seed)
    setuplogging()

    if args.local_rank in [-1, 0]:
        print(os.getcwd())

    if args.mode == 'train':
        if args.local_rank in [-1, 0]:
            print('-----------train------------')
        run(args)

    if args.mode == 'test':
        print('-------------test--------------')
        assert args.local_rank == -1
        test(args)
