import argparse
import time
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--cuda_idx', type=int, default=0)

    # pretrain and train
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--train_num_neg', type=int, default=4)
    parser.add_argument('--start_epoch_idx', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--opt_factor', type=float, default=1)
    parser.add_argument('--opt_warmup', type=int, default=4000)
    parser.add_argument('--print_every', type=int, default=8,
                        help='Iteration interval of printing loss.')
    parser.add_argument('--save_every', type=int, default=4,
                        help='Iteration interval of saving model.')
    parser.add_argument('--evaluate_every', type=int, default=4,
                        help='Epoch interval of evaluation.')

    # validation and test
    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--test_num_neg', type=int, default=99)
    parser.add_argument('--test_neg', type=bool, default=True)
    parser.add_argument('--k_list', type=list, default=[5, 10])

    # model
    parser.add_argument('--corr_factor', type=float, default=0.1)
    parser.add_argument('--num_head', type=int, default=4,
                        choices=[1, 2, 4, 8])
    parser.add_argument('--enc_num_layer', type=int, default=2,
                        choices=[1, 2, 3])
    parser.add_argument('--sub_seq_num', type=int, default=2,
                        choices=[1, 2, 3, 4])
    parser.add_argument('--emb_size', type=int, default=32,
                        choices=[16, 32, 48, 64, 80])
    parser.add_argument('--hid_size', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--trained_model_path', type=str, default='')

    # data
    parser.add_argument('--data_name', type=str, default='JDsearch', choices=['JDsearch', 'Amazon_Clothing', 'Amazon_Electronics'])
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--padding_value', type=int, default=0)
    parser.add_argument('--query_max_len', type=int, default=50)

    args = parser.parse_args()

    args.data_root = os.path.join(args.data_root, args.data_name)
    data_meta_path = os.path.join(args.data_root, 'meta.csv')
    user_num, product_num, term_num = pd.read_csv(data_meta_path, sep='\t').values.squeeze()
    args.user_vocab = user_num + 1
    args.product_vocab = product_num + 1
    args.term_vocab = term_num + 1
    args.bos_id = args.term_vocab + 1 # Begin-of-Sentence
    args.eos_id = args.term_vocab # End-of-Sentence

    return args


if __name__ == "__main__":
    # for test only
    pass