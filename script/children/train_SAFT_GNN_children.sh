CUDA_VISIBLE_DEVICES=0 python SAFT/main.py \
    --large False \
    --data_path "edge_data/children" \
    --model_type "SAFT_GNN_small" \
    --epochs 300 \
    --early_stop 30 \
    --lr 1e-3 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --rd 0 \
    --prop_layers 3 \
    --tlambda 1 \
    --delta 2 \
    --pe_dim 256 \
    --model_name_or_path "prajjwal1/bert-tiny"
