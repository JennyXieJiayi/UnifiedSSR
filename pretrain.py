from utils.parser import parse_args
from utils.logger import create_log_id, logging_config
from utils.optimizer import NoamOpt
from utils.utils import save_model, load_model
from data import UnifiedDataset
from batch import BatchSampler, collate_pretrain
from model import get_model
import os, time, logging, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_epoch(args, model, data_loader, optimizer, epoch_idx, batch_num, device):
	time1 = time.time()
	model.train()
	total_out_loss = 0.
	total_corr_loss = 0.
	total_batch_loss = 0.
	for idx, (cur_task, batch_data) in enumerate(data_loader):
		batch_idx = idx + 1
		time2 = time.time()
		for key in batch_data.keys():
			if isinstance(batch_data[key], list):
				batch_data[key] = [[i.to(device) for i in data] for data in batch_data[key]]
			else:
				batch_data[key] = batch_data[key].to(device)
		out = model.forward(cur_task, batch_data)

		if cur_task == 'recommendation':
			if args.corr_factor > 0:
				sub_seq_wins = model.get_sub_seq_wins(out)
				out, _, corr_loss = model.intra_corr_loss(out, sub_seq_wins, batch_data['pids_mask'])
			else:
				corr_loss = torch.tensor(0.)
			out_loss = model.loss(out.view(-1, out.size(-1)),
			                      batch_data['pids_mask'].view(-1),
			                      batch_data['pids_tgt'].view(-1),
			                      batch_data['pids_neg'].view(-1, args.train_num_neg))
		else:  # cur_task == 'search':
			p_out, q_out = out
			p_out = p_out[:, 1:, :]
			q_out = q_out[:, 1:, :]
			mask = batch_data['pids_mask'][:, :, 1:]

			if args.corr_factor > 0:
				p_sub_seq_wins = model.get_sub_seq_wins(p_out)
				q_sub_seq_wins = model.get_sub_seq_wins(q_out)
				p_out, q_out, corr_loss = model.inter_corr_loss(p_out, p_sub_seq_wins, q_out, q_sub_seq_wins, mask)
			else:
				corr_loss = torch.tensor(0.)
			out_loss = model.loss(p_out.reshape(-1, p_out.size(-1)),
			                      q_out.reshape(-1, q_out.size(-1)),
			                      mask.reshape(-1),
			                      batch_data['pids_tgt'].view(-1),
			                      batch_data['pids_neg'].view(-1, args.train_num_neg))

		batch_loss = out_loss + args.corr_factor * corr_loss
		batch_loss.backward()
		cur_lr = optimizer.step()
		optimizer.optimizer.zero_grad()
		total_out_loss += out_loss.item()
		total_corr_loss += corr_loss.item()
		total_batch_loss += batch_loss.item()
		if (batch_idx % args.print_every) == 0:
			logging.info(
				'Training: Epoch {:04d} Iter {:04d} / {:04d} | Current Task {} | Time {:.1f}s | L_Rate {:.5f}'.format(
					epoch_idx, batch_idx, batch_num, cur_task, time.time() - time2, cur_lr))
			logging.info(
				'Training: Iter Loss {:.4f} | Out Loss {:.4f} | Corr Loss {:.4f}'.format(batch_loss.item(),
				                                                                         out_loss.item(),
				                                                                         corr_loss.item()))
			logging.info(
				'Training: Iter Mean Loss {:.4f} | Out Mean Loss {:.4f} | Corr Mean Loss {:.4f}'.format(
					total_batch_loss / batch_idx, total_out_loss / batch_idx, total_corr_loss / batch_idx))

	logging.info(
		'Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s'.format(epoch_idx, batch_num,
		                                                                       time.time() - time1))
	logging.info(
		'Training: Iter Mean Loss {:.4f} | Out Mean Loss {:.4f} | Corr Mean Loss {:.4f}'.format(
			total_batch_loss / batch_num, total_out_loss / batch_num, total_corr_loss / batch_num))

	# save model
	if (epoch_idx % args.save_every) == 0:
		save_model(model, args.save_dir, epoch_idx)


def pretrain(args):
	# log
	log_name = 'log_pretrain'
	log_save_id = create_log_id(args.save_dir, name=log_name)
	logging_config(folder=args.save_dir, name='{}_{:d}'.format(log_name, log_save_id), no_console=False)
	logging.info(args)

	# GPU / CPU
	args.use_cuda = args.use_cuda & torch.cuda.is_available()
	device = torch.device("cuda:{}".format(args.cuda_idx) if args.use_cuda else "cpu")

	# load data
	data = UnifiedDataset(args.phase, args.tasks, args.data_root, logging)

	batch_sampler = BatchSampler(data, args.train_batch_size)
	data_loader = DataLoader(data,
	                         batch_sampler=batch_sampler,
	                         collate_fn=lambda x: collate_pretrain(x, args))
	batch_num = len(data_loader)

	# construct model
	model = get_model(args)
	model.to(device)
	logging.info(model)

	if os.path.isfile(args.trained_model_path):
		logging.info("Loading pre-trained model: {}".format(args.trained_model_path))
		model = load_model(model, args.trained_model_path)
	else:
		logging.info('Parameters initializing ...')
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_((p))
		model.seq_partition.reset_offset()

	# define optimizer
	optimizer = NoamOpt(args.emb_size, args.opt_factor, args.opt_warmup,
	                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
	logging.info(optimizer)

	start_epoch_idx = args.start_epoch_idx or 1
	for epoch_idx in range(start_epoch_idx, args.num_epoch + start_epoch_idx):
		# train and save model
		run_epoch(args, model, data_loader, optimizer, epoch_idx, batch_num, device)


if __name__ == '__main__':
	args = parse_args()

	# Seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Pretrain
	args.phase = 'pretrain'
	args.tasks = ['recommendation', 'search']
	args.save_dir = f'models/{args.data_name}/{"_".join([args.phase] + args.tasks)}/{time.strftime("%Y%m%d_%H%M%S")}/'
	pretrain(args)