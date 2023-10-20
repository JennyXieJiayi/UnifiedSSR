from utils.parser import parse_args
from utils.logger import create_log_id, logging_config
from utils.utils import load_model
from metrics import evaluate_product
from data import UnifiedDataset
from batch import collate_test
from model import get_model
import numpy as np
import pandas as pd
import os, logging, random
import torch
from torch.utils.data import DataLoader

def predict_product(args):
	# log
	log_name = f'log_test_{args.tasks[0]}'
	log_save_id = create_log_id(args.save_dir, name=log_name)
	logging_config(folder=args.save_dir, name='{}_{:d}'.format(log_name, log_save_id), no_console=False)
	logging.info(args)

	# GPU / CPU
	args.use_cuda = args.use_cuda & torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args.cuda_idx) if args.use_cuda else "cpu")

	# load data
	data = UnifiedDataset(args.phase, args.tasks, args.data_root, logging)

	data_loader = DataLoader(data,
	                         shuffle=False,
	                         batch_size=args.test_batch_size,
	                         collate_fn=lambda x: collate_test(x, args))

	# load model
	model = get_model(args)
	model = load_model(model, args.trained_model_path).to(device)

	# evaluate
	hits, ndcgs = evaluate_product(model, data_loader, len(data), args, device)
	for k_idx, topk in enumerate(args.k_list):
		logging.info(
			'Evaluation (K={}): HR {:.4f} NDCG {:.4f}'.format(topk, hits[k_idx], ndcgs[k_idx]))

	# initialize metrics
	result_save_file = os.path.join(args.save_dir, 'test_results.csv')
	init_metrics = pd.DataFrame(['HR@{}'.format(k) for k in args.k_list] +
	                            ['NDCG@{}'.format(k) for k in args.k_list]).transpose()
	init_metrics.to_csv(result_save_file, mode='a', header=False, sep='\t', index=False)
	metrics = pd.DataFrame(hits.tolist() + ndcgs.tolist()).transpose()
	metrics.to_csv(result_save_file, mode='a', header=False, sep='\t', index=False)
	return hits, ndcgs


if __name__ == "__main__":
	args = parse_args()

	# Seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Evaluation
	args.phase = 'finetune'
	args.tasks = ['recommendation']  # 'recommendation', 'search'

	pretrain_dir = 'models/Amazon_Clothing/pretrain_recommendation_search/finetune_recommendation'
	# pretrain_dir = 'models/Amazon_Clothing/pretrain_recommendation_search/finetune_search'
	args.trained_model_path = os.path.join(pretrain_dir, f'model.pth')
	args.save_dir = pretrain_dir
	predict_product(args)
