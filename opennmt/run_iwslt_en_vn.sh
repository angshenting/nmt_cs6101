#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2 onmt-main --model_type LuongAttention \
          --config config/iwslt_en_vn_data_luong.yml --auto_config \
          train --with_eval --num_gpus 3
CUDA_VISIBLE_DEVICES=0,1,2 onmt-main --model ./models/bahdanau_attention.py \
          --config config/iwslt_en_vn_data_bahdanau.yml --auto_config \
          train --with_eval --num_gpus 3
CUDA_VISIBLE_DEVICES=0,1,2 onmt-main --model_type Transformer \
          --config config/iwslt_en_vn_data_transformer.yml --auto_config \
          train --with_eval --num_gpus 3
CUDA_VISIBLE_DEVICES=0,1,2 onmt-main --model_type TransformerBig \
          --config config/iwslt_en_vn_data_transformerbig.yml --auto_config \
          train --with_eval --num_gpus 3
