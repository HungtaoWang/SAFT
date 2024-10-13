CUDA_VISIBLE_DEVICES=0 python SAFT/main.py \
    --large True \
    --data_path "edge_data/poetry" \
    --model_type "SAFT_GNN_large" \
    --epochs 100 \
    --early_stop 3 \
    --lr 1e-5 \
    --weight_decay 1e-3 \
    --adam_epsilon 1e-8 \
    --rd 107 \
    --prop_layers 4 \
    --tlambda 1.5 \
    --delta 1.5 \
    --pe_dim 64 \
    --train_batch_size 25\
    --approximate True\
    --model_name_or_path "bert-base-uncased"
