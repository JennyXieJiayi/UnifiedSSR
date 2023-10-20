import os
import logging
import joblib
from torch.utils.data import Dataset


class UnifiedDataset(Dataset):
	def __init__(self, phase, tasks, data_root='', logging=False):
		assert os.path.exists(data_root)
		assert phase in ['pretrain', 'finetune'], 'Phase must be pretrain or finetune.'
		assert set(tasks).issubset({'recommendation', 'search'}), 'Task must be specified as in recommendation and search.'
		self.data_root = data_root
		self.phase = phase
		self.tasks = tasks
		self.tasks_num_sample = []
		self.user_seq = []

		for task in tasks:
			data = joblib.load(os.path.join(data_root, f'{phase}_{task}.pkl'))
			self.tasks_num_sample.append(len(data))
			self.user_seq.extend(data)

		if logging:
			self.print_info()

	def __len__(self):
		return len(self.user_seq)

	def __getitem__(self, index):
		return self.user_seq[index]

	def print_info(self):
		logging.info(f'current data path: {self.data_root}')
		logging.info(f'current phase: {self.phase}')
		logging.info(f'current task: {self.tasks}')
		for num_sample, task in zip(self.tasks_num_sample, self.tasks):
			logging.info(f"the number of samples for task {task}: {num_sample}")


if __name__ == '__main__':
	# for test only
	pass

