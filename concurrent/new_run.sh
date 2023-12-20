#!/bin/sh

python3 main.py --algo="FedAvg" --K=100 --C=0.5 --E=1 --B=8 --T=15 --lr=0.01 --alpha=0.1 --gpu="gpu" --model="nn" --name="exp_sec2" > ../logs/exp_sec2.txt