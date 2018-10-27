import torch
import argparse
import numpy as np
from utils import str2bool
from solver import Solver

def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    net = Solver(args)
    if args.mode == 'train':
        net.train()
    elif args.mode == 'test':
        net.test()
    # TODO:elif generate -- adv
    else: return

    print('[*] finished!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RBF trainer')
    parser.add_argument('--model_name', type=str, default='vanilla_rbf')
    parser.add_argument('--dataset', type=str, default='catvdog')
    parser.add_argument('--D_out', type=int, default=10)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_ckpt', type=bool, default=True, help='load from checkpoint') # TODO: change to an address
    parser.add_argument('--epoch', type=int, default=200, help='epoch size')


    parser.add_argument('--center_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

    parser.add_argument('--print_iter', type=int, default=200, help='')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--shuffle', type=bool, default=True)


    args = parser.parse_args()
    main(args)



