#!/bin/bash

python3 train.py --epoch 50 --iters 1000 --batch_size 64 --mean 5 --std 1
python3 train.py --epoch 50 --iters 1000 --batch_size 64 --mean 10 --std 1 
python3 train.py --epoch 50 --iters 1000 --batch_size 64 --mean 25 --std 1
python3 train.py --epoch 50 --iters 1000 --batch_size 64 --mean 100 --std 1

