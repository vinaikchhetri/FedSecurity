import argparse

def arg_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--algo", type=str, help="Run FedAvg")

    parser.add_argument("--K", type=int, help="total no. of clients.")
    parser.add_argument("--C", type=float, help="C-fraction of clients.")
    parser.add_argument("--E", type=int, help="local epochs.")
    parser.add_argument("--B", type=int, help="batch size.")
    parser.add_argument("--T", type=int, help="total no. of rounds.")
    parser.add_argument("--lr", type=float, help="learning rate.")

    parser.add_argument("--dataset", type=str, default = "mnist", help="dataset choice.")
    parser.add_argument("--iid", type=str, default = "true", help="data distribution.")
    parser.add_argument("--model", type=str, default = "nn", help="model choice.")
    parser.add_argument("--gpu", type=str, default = "cpu", help="gpu or cpu.")
    parser.add_argument("--name", type=str,  help="save stats list as ...")
    args = parser.parse_args()
    
    return args