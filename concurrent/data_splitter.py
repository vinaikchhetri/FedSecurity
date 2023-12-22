import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import utils

import core

def splitter(args):
    clients = []
    if args.algo == "FedAvg":
        if args.dataset == "mnist":
            dataset_name = 'mnist'
            trainset = torchvision.datasets.MNIST(root='../data/'+dataset_name, train=True, download=True, transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor()
                        ]))
            testset = torchvision.datasets.MNIST(root='../data'+dataset_name, train=False, download=True, transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                    ]))

            if args.iid == "true":
                #construct an iid mnist dataset.
                #distribute data among clients
                client_data_dict = {}
                all_indices = np.arange(0,len(trainset))
                available_indices = np.arange(len(trainset))
                for client_idx in range(args.K):
                    selected_indices = np.random.choice(available_indices, 600, replace=False)
                    client_data_dict[client_idx] = selected_indices
                    available_indices = np.setdiff1d(available_indices, selected_indices)

                # Construct dataset here for posioned samples for each client and send them to api
                # - Randomly sample data for each client and concatenate them into an array.
                # - Then ship them.
                alpha = args.alpha
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
                train_batch  = next(iter(train_loader))
                x_list = torch.zeros_like(train_batch[0][0:1])
                y_list = torch.zeros_like(train_batch[1][0:1])
                boundaries = []
                count = 0
                for client_idx in range(args.K):
                    chosen_indices = client_data_dict[client_idx]
                    sampling_amount = int(alpha*np.shape(chosen_indices)[0])
                    #boundaries of poison
                    count+=sampling_amount
                    boundaries.append(count)
                    sampled_indices = np.random.choice(chosen_indices, sampling_amount, replace=False)
                    #[(x1,y1),(x2,y2),...,(xn,yn)]
                    x = train_batch[0][sampled_indices]
                    y = train_batch[1][sampled_indices]
                    x_list = torch.cat([x_list,x])
                    y_list = torch.cat([y_list,y])
                x_list = x_list[1:]
                y_list = y_list[1:]
                #train_poison = utils.CustomDataset(x_list, y_list)
                
                if args.target == "all":
                    if args.pattern == "pixel":
                        pattern = torch.zeros((1, 28, 28), dtype=torch.float32)
                        pattern[0, -2, -2] = 1.0
                        weight = torch.zeros((1, 28, 28), dtype=torch.float32)
                        weight[0, -2, -2] = 1.0
                        res = weight * pattern
                        weight = 1.0 - weight
                    else:
                        pattern = torch.zeros((28, 28), dtype=torch.float32)
                        pattern[-3:, -3:] = 1.0
                        weight = torch.zeros((28, 28), dtype=torch.float32)
                        weight[-3:, -3:] = 1.0
                        res = weight * pattern
                        weight = 1.0 - weight
                        if pattern.dim() == 2:
                            pattern = pattern.unsqueeze(0)
                        if weight.dim() == 2:
                            weight = weight.unsqueeze(0)

                    res = res.repeat(len(x_list),1,1,1)
                    weight = weight.repeat(len(x_list),1,1,1)
                    trx_list = weight * x_list + res
                    try_list = torch.remainder(y_list+1, 10)
                    torch.save(trx_list,'x2.pt')
                    torch.save(try_list,'y2.pt')

                    #test
                
                    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
                    test_batch  = next(iter(test_loader))

                    chosen_indices = np.arange(len(test_batch[0]))
                    sampling_amount = int(alpha*np.shape(chosen_indices)[0])
                    sampled_indices = np.random.choice(chosen_indices, sampling_amount, replace=False)
                    
                    
                    if args.pattern == "pixel":
                        pattern = torch.zeros((1, 28, 28), dtype=torch.float32)
                        pattern[0, -2, -2] = 1.0
                        weight = torch.zeros((1, 28, 28), dtype=torch.float32)
                        weight[0, -2, -2] = 1.0
                        res = weight * pattern
                        weight = 1.0 - weight
                    else:
                        pattern = torch.zeros((28, 28), dtype=torch.float32)
                        pattern[-3:, -3:] = 1.0
                        weight = torch.zeros((28, 28), dtype=torch.float32)
                        weight[-3:, -3:] = 1.0
                        res = weight * pattern
                        weight = 1.0 - weight
                        if pattern.dim() == 2:
                            pattern = pattern.unsqueeze(0)
                        if weight.dim() == 2:
                            weight = weight.unsqueeze(0)

                    res = res.repeat(len(test_batch[0][sampled_indices.astype(int)]),1,1,1)
                    weight = weight.repeat(len(test_batch[0][sampled_indices.astype(int)]),1,1,1)
                    tstx_list = weight * test_batch[0][sampled_indices.astype(int)] + res
                    tsty_list = torch.remainder(test_batch[1][sampled_indices.astype(int)]+1, 10)
                    torch.save(tstx_list,'tstx2.pt')
                    torch.save(tsty_list,'tsty2.pt')

                    # tst_X = torch.cat([tstx_list,test_batch[0]])
                    # tst_Y = torch.cat([tsty_list,test_batch[1]])
                    # print(tst_X.shape)
                    # print(tst_Y.shape)
                    # test_poison = utils.CustomDataset(tst_X, tst_Y)
                    test_poison = utils.CustomDataset(tstx_list, tsty_list)
            
            else:
                #construct a non-iid mnist dataset.
                #distribute data among clients
                client_data_dict = {}
                labels = trainset.targets.numpy()
                sorted_indices = np.argsort(labels)

                all_indices = np.arange(0,200)
                available_indices = np.arange(0,200)
                for client_idx in range(args.K):
                    selected_indices = np.random.choice(available_indices, 2, replace=False)               
                    A = sorted_indices[selected_indices[0]*300:selected_indices[0]*300+300]
                    B = sorted_indices[selected_indices[1]*300:selected_indices[1]*300+300]
                    merged_shards = np.concatenate((np.expand_dims(A, 0), np.expand_dims(B,0)), axis=1)
                    client_data_dict[client_idx] = merged_shards[0]
                    available_indices = np.setdiff1d(available_indices, selected_indices)

        if args.dataset == "cifar-100":
            dataset_name = 'cifar-100'
            train_data = torchvision.datasets.CIFAR100('./', train=True, download=True)
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = torchvision.datasets.CIFAR100(root='../data/'+dataset_name,
                                            train=True,
                                            download=True,
                                            transform=transform)

            testset = torchvision.datasets.CIFAR100(root='../data/'+dataset_name,
                                            train=False,
                                            download=True,
                                            transform=transform)
            if args.iid == "true":
                #construct an iid mnist dataset.
                #distribute data among clients
                client_data_dict = {}
                all_indices = np.arange(0,len(trainset))
                available_indices = np.arange(len(trainset))
                for client_idx in range(args.K):
                    selected_indices = np.random.choice(available_indices, 500, replace=False)
                    client_data_dict[client_idx] = selected_indices
                    available_indices = np.setdiff1d(available_indices, selected_indices)

                # Construct dataset here for posioned samples for each client and send them to api
                # - Randomly sample data for each client and concatenate them into an array.
                # - Then ship them.
                alpha = args.alpha
                train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
                train_batch  = next(iter(train_loader))
                x_list = torch.zeros_like(train_batch[0][0:1])
                y_list = torch.zeros_like(train_batch[1][0:1])
                for client_idx in range(args.K):
                    chosen_indices = client_data_dict[client_idx]
                    sampling_amount = int(alpha*np.shape(chosen_indices)[0])
                    sampled_indices = np.random.choice(chosen_indices, sampling_amount, replace=False)
                    #[(x1,y1),(x2,y2),...,(xn,yn)]
                    x = train_batch[0][sampled_indices]
                    y = train_batch[1][sampled_indices]
                    x_list = torch.cat([x_list,x])
                    y_list = torch.cat([y_list,y])
                x_list = x_list[1:]
                y_list = y_list[1:]
                train_poison = utils.CustomDataset(x_list, y_list)
                print("here")
                

                # pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
                # pattern[0, -2, -2] = 255
                # weight = torch.zeros((1, 28, 28), dtype=torch.float32)
                # weight[0, -2, -2] = 1.0
                # res = weight * pattern
                # weight = 1.0 - weight
                # (weight * img + res).type(torch.uint8)



            
            else:
                #construct a non-iid mnist dataset.
                #distribute data among clients
                client_data_dict = {}
            
                labels = np.asarray(trainset.targets)
                sorted_indices = np.argsort(labels)

                all_indices = np.arange(0,1000)
                available_indices = np.arange(0,1000)
            
                for client_idx in range(args.K):
                    merged_shards = np.array([[]])
                    selected_indices = np.random.choice(available_indices, 10, replace=False)
                    for index in range(10):
                        temp = sorted_indices[selected_indices[index]*50:selected_indices[index]*50+50]               
                        merged_shards = np.concatenate((merged_shards, np.expand_dims(temp,0)), axis=1)
                    client_data_dict[client_idx] = merged_shards[0].astype(int)
                    available_indices = np.setdiff1d(available_indices, selected_indices)
        
    
        # return trainset,testset,client_data_dict
        return train_batch, testset, test_poison, client_data_dict, trx_list, try_list, boundaries

            


