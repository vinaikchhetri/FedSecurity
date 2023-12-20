import torchvision
import torchvision.transforms as transforms
import numpy as np

def splitter(args):
    clients = []
    if args.algo == "FedAvg":
        if args.dataset == "mnist":
            dataset_name = 'mnist'
            trainset = torchvision.datasets.MNIST(root='../data/'+dataset_name, train=True, download=True, transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                        ]))
            testset = torchvision.datasets.MNIST(root='../data'+dataset_name, train=False, download=True, transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
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
                # Randomly sample data for each client and concatenate them into an array.
                # Then ship them.
            
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
        
    
        return trainset,testset,client_data_dict

            


