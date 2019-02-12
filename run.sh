#!/usr/bin/env bash

~/anaconda3/envs/pysl/bin/python -u main.py --model_name vanilla_rbf --dataset cifar-100 \
--D_out 100 --mode cross_val --epochs 30 --center_num 10 --load_from best