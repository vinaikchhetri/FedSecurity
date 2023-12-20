#!/bin/sh

# #iid and mnist

# #nn

# #B=inf
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_inf_0" > ../logs/exp_nn_inf_0.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_inf_0.1" > ../logs/exp_nn_inf_0.1.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_inf_0.2" > ../logs/exp_nn_inf_0.2.txt

# #B=10
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0" > ../logs/exp_nn_10_0.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0.1" > ../logs/exp_nn_10_0.1.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0.2" > ../logs/exp_nn_10_0.2.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_0.5" > ../logs/exp_nn_10_0.5.txt
# python main.py --algo="FedAvg" --K=100 --C=1 --E=1 --B=10 --T=400 --lr=0.01 --gpu="gpu" --model="nn" --name="exp_nn_10_1" > ../logs/exp_nn_10_1.txt

#cnn

#B=inf
#python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0" > ../logs/exp_cnn_inf_0.txt
#python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0.1" > ../logs/exp_cnn_inf_0.1.txt
#python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0.2" > ../logs/exp_cnn_inf_0.2.txt
#python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_0.5" > ../logs/exp_cnn_inf_0.5.txt
#python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=8 --T=400 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_inf_1" > ../logs/exp_cnn_inf_1.txt

#B=10
#python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0" > ../logs/exp_cnn_10_0.txt
#python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0.1" > ../logs/exp_cnn_10_0.1.txt
#python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0.2" > ../logs/exp_cnn_10_0.2.txt
#python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_0.5" > ../logs/exp_cnn_10_0.5.txt
#python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=10 --T=100 --lr=0.01 --gpu="gpu" --model="cnn" --name="exp_cnn_10_1" > ../logs/exp_cnn_10_1.txt

# non-iid and mnist

# nn

# B=inf
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=8 --T=5000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_inf_0_f" > ../logs/exp_nn_inf_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=8 --T=2000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_inf_0.1_f" > ../logs/exp_nn_inf_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=8 --T=2000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_inf_0.2_f" > ../logs/exp_nn_inf_0.2_f.txt

# #B=10
# python main.py --algo="FedAvg" --K=100 --C=0 --E=1 --B=10 --T=4000 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0_f" > ../logs/exp_nn_10_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=1 --B=10 --T=800 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0.1_f" > ../logs/exp_nn_10_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=1 --B=10 --T=800 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0.2_f" > ../logs/exp_nn_10_0.2_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=1 --B=10 --T=600 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_0.5_f" > ../logs/exp_nn_10_0.5_f.txt
# python main.py --algo="FedAvg" --K=100 --C=1 --E=1 --B=10 --T=600 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="exp_nn_10_1_f" > ../logs/exp_nn_10_1_f.txt

# #cnn

# #B=inf
# python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0_f" > ../logs/exp_cnn_inf_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0.1_f" > ../logs/exp_cnn_inf_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0.2_f" > ../logs/exp_cnn_inf_0.2_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=8 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_inf_0.5_f" > ../logs/exp_cnn_inf_0.5_f.txt

# #B=10
# python main.py --algo="FedAvg" --K=100 --C=0 --E=5 --B=10 --T=1500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0_f" > ../logs/exp_cnn_10_0_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.1 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0.1_f" > ../logs/exp_cnn_10_0.1_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.2 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0.2_f" > ../logs/exp_cnn_10_0.2_f.txt
# python main.py --algo="FedAvg" --K=100 --C=0.5 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_0.5_f" > ../logs/exp_cnn_10_0.5_f.txt
# python main.py --algo="FedAvg" --K=100 --C=1 --E=5 --B=10 --T=500 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="exp_cnn_10_1_f" > ../logs/exp_cnn_10_1_f.txt


# Figure 2
#iid cnn
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="cnn" --name="fig2_cnn_B${b}_E${e}" > ../new_logs/mnist/"fig2_cnn_B${b}_E${e}"
          
#     done     
# done 

# ## Pathological Non-IID split cnn
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="cnn" --iid="false" --name="niid_fig2_cnn_B${b}_E${e}" > ../new_logs/mnist/"niid_fig2_cnn_B${b}_E${e}"
          
#     done     
# done 


# # Figure 7
# #iid 2nn
# for b in 8 50 10
# do
#     for e in 1 10 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=10 --lr=0.01 --gpu="gpu" --model="nn" --name="fig2_nn_B${b}_E${e}" > ../new_logs/mnist/"fig2_nn_B${b}_E${e}"
          
#     done     
# done 

# ## Pathological Non-IID split 2nn
# for b in 8 50 10
# do
#     for e in 1 10 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=10 --lr=0.01 --gpu="gpu" --model="nn" --iid="false" --name="niid_fig2_nn_B${b}_E${e}" > ../new_logs/mnist/"niid_fig2_nn_B${b}_E${e}"
          
#     done     
# done 




# #iid resnet
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=2000 --lr=0.01 --gpu="gpu" --model="resnet" --name="resnet_B${b}_E${e}" > ../new_logs/cifar-100/"resnet_B${b}_E${e}"
          
#     done     
# done 

# ## Pathological Non-IID split resnet
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --K=100 --C=0.1 --E=$e --B=$b --T=2000 --lr=0.01 --gpu="gpu" --model="resnet" --iid="false" --name="niid_resnet_B${b}_E${e}" > ../new_logs/cifar-100/"niid_resnet_B${b}_E${e}"
          
#     done     
# done 


#-----
#iid resnet
# for b in 8 50 10
# do
#     for e in 1 5 20 
#     do
#         python3 main.py \
#             --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="resnet" --name="resnet_B${b}_E${e}" > ../new_logs/cifar-100/"resnet_B${b}_E${e}"
          
#     done     
# done 

## Pathological Non-IID split resnet
for b in 8 50 10
do
    for e in 1 5 20   
    do
        python3 main.py \
            --algo="FedAvg" --dataset 'cifar-100' --K=100 --C=0.1 --E=$e --B=$b --T=1000 --lr=0.01 --gpu="gpu" --model="resnet" --iid="false" --name="niid_resnet_B${b}_E${e}" > ../new_logs/cifar-100/"niid_resnet_B${b}_E${e}"
          
    done     
done 


