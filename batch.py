import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence


def subsequent_mask(size):
    """
    The shape of the attention map (size, size) = (max_len, max_len)
    Return the lower triangular mask
    """
    return torch.tril(torch.ones((1, size, size)), diagonal=0).bool()


def neg_sampling(batch_data, num_neg, vocab):
    batch_neg = []
    for data in batch_data:
        if isinstance(data, list):
            batch_neg.append(neg_sampling(data, num_neg, vocab))
        else:
            sampled_negs = []
            for _ in range(num_neg):
                neg = np.random.randint(1, vocab)
                while neg == data or neg in sampled_negs:
                    neg = np.random.randint(1, vocab)
                sampled_negs.append(neg)
            batch_neg.append(sampled_negs)
    return batch_neg


def collate_pretrain(batch_data, args):
    num_neg = args.train_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()
    if batch_data[0]['flag'] == 'recommendation':
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_out = [data['pid_list'][1:] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True, padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True,
                                    padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        return 'recommendation', {'uid': uids,
                           'pids_in': pids_in,
                           'pids_mask': pids_mask,
                           'pids_tgt': pids_out,
                           'pids_neg': pids_out_neg}

    else: # 'search'
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pids_out = [data['pid_list'][1:] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True, padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True, padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)

        qrys_in = [data['qry_list'] for data in batch_data]  # [BS, Seq Len, Qry Len]
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in_mask = torch.stack([torch.cat([torch.ones(1)] * len(qrys) + [torch.zeros(1)] * (qrys_in_seq_max_len - len(qrys))) for qrys in qrys_in])
        qrys_in_mask = (qrys_in_mask != padding_value).unsqueeze(-2)
        # qrys_in_mask [BS, 1, Seq Max Len]
        qrys_in = [[torch.tensor(qry).long() for qry in qrys] + [torch.zeros(1).long()] * (qrys_in_seq_max_len - len(qrys)) for qrys in qrys_in]  # padding
        # qrys_in [BS, Seq Max Len, Qry Len]

        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pids_tgt': pids_out,
                          'pids_neg': pids_out_neg,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask}


def collate_train(batch_data, args):
    num_neg = args.train_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()
    if batch_data[0]['flag'] == 'recommendation':
        pids_in = [torch.tensor(data['pid_list'][:-3]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_out = [data['pid_list'][1:-2] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True, padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True, padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        return 'recommendation', {'uid': uids,
                           'pids_in': pids_in,
                           'pids_mask': pids_mask,
                           'pids_tgt': pids_out,
                           'pids_neg': pids_out_neg}

    else: # 'search'
        pids_in = [torch.tensor(data['pid_list'][:-3]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pids_out = [data['pid_list'][1:-2] for data in batch_data]
        pids_out_neg = neg_sampling(pids_out, num_neg, p_vocab)
        pids_out = pad_sequence([torch.tensor(pid_out).long() for pid_out in pids_out], batch_first=True,
                                padding_value=padding_value)
        pids_out_neg = pad_sequence([torch.tensor(pid_neg).long() for pid_neg in pids_out_neg], batch_first=True,
                                    padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)

        qrys_in = [data['qry_list'][:-2] for data in batch_data]
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in_mask = torch.stack([torch.cat([torch.ones(1)] * len(qrys) + [torch.zeros(1)] * (qrys_in_seq_max_len - len(qrys))) for qrys in qrys_in])
        qrys_in_mask = (qrys_in_mask != padding_value).unsqueeze(-2)
        qrys_in = [[torch.tensor(qry).long() for qry in qrys] + [torch.zeros(1).long()] * (qrys_in_seq_max_len - len(qrys)) for qrys in qrys_in]

        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pids_tgt': pids_out,
                          'pids_neg': pids_out_neg,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask}


def collate_val(batch_data, args):
    num_neg = args.test_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()
    if batch_data[0]['flag'] == 'recommendation':
        pids_in = [torch.tensor(data['pid_list'][:-2]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 3 for data in batch_data])

        pid_tgt = [data['pid_list'][-2] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)

        return 'recommendation', {'uid': uids,
                           'pids_in': pids_in,
                           'pids_mask': pids_mask,
                           'pid_pred': pid_pred,
                           'pid_last_idx': pid_last_idx}

    else: # 'search'
        pids_in = [torch.tensor(data['pid_list'][:-2]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pid_tgt = [data['pid_list'][-2] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 3 for data in batch_data])

        qrys_in = [data['qry_list'][:-1] for data in batch_data]
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in_mask = torch.stack([torch.cat([torch.ones(1)] * len(qrys) + [torch.zeros(1)] * (qrys_in_seq_max_len - len(qrys))) for qrys in qrys_in])
        qrys_in_mask = (qrys_in_mask != padding_value).unsqueeze(-2)
        qrys_in = [[torch.tensor(qry).long() for qry in qrys] + [torch.zeros(1).long()] * (qrys_in_seq_max_len - len(qrys)) for qrys in qrys_in]

        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pid_pred': pid_pred,
                          'pid_last_idx': pid_last_idx,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask}


def collate_test(batch_data, args):
    num_neg = args.test_num_neg
    p_vocab = args.product_vocab
    padding_value = args.padding_value
    uids = [int(data['uid']) for data in batch_data]
    uids = torch.tensor(uids).long()
    if batch_data[0]['flag'] == 'recommendation':
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 2 for data in batch_data])

        pid_tgt = [data['pid_list'][-1] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)

        return 'recommendation', {'uid': uids,
                                  'pids_in': pids_in,
                                  'pids_mask': pids_mask,
                                  'pid_pred': pid_pred,
                                  'pid_last_idx': pid_last_idx}

    else:  # 'search'
        pids_in = [torch.tensor(data['pid_list'][:-1]).long() for data in batch_data]
        pids_in = pad_sequence(pids_in, batch_first=True, padding_value=padding_value)
        pids_in = F.pad(pids_in, (1, 0), value=padding_value)
        pid_tgt = [data['pid_list'][-1] for data in batch_data]
        pid_tgt_neg = neg_sampling(pid_tgt, num_neg, p_vocab)
        pid_tgt = torch.tensor(pid_tgt).long()
        pid_tgt_neg = torch.tensor(pid_tgt_neg).long()
        pid_pred = torch.cat((pid_tgt.unsqueeze(-1), pid_tgt_neg), -1)
        pids_mask = (pids_in != padding_value).unsqueeze(-2)
        pid_last_idx = torch.tensor([data['length'] - 2 for data in batch_data])

        qrys_in = [data['qry_list'] for data in batch_data]
        qrys_in_seq_max_len = max(len(qrys) for qrys in qrys_in)
        qrys_in_mask = torch.stack(
            [torch.cat([torch.ones(1)] * len(qrys) + [torch.zeros(1)] * (qrys_in_seq_max_len - len(qrys))) for qrys in
             qrys_in])
        qrys_in_mask = (qrys_in_mask != padding_value).unsqueeze(-2)
        qrys_in = [
            [torch.tensor(qry).long() for qry in qrys] + [torch.zeros(1).long()] * (qrys_in_seq_max_len - len(qrys)) for
            qrys in qrys_in]

        return 'search', {'uid': uids,
                          'pids_in': pids_in,
                          'pids_mask': pids_mask,
                          'pid_pred': pid_pred,
                          'pid_last_idx': pid_last_idx,
                          'qrys_in': qrys_in,
                          'qrys_in_mask': qrys_in_mask}


class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(BatchSampler, self).__init__(data_source)
        self.recommendation_data_len = []
        self.search_data_len = []
        self.batch_size = batch_size
        self.pool_size = batch_size * 100
        for idx, data in enumerate(data_source):
            if data['flag'] == 'recommendation':
                self.recommendation_data_len.append((idx, data['length']))
            else: # data['flag'] == 'search'
                self.search_data_len.append((idx, data['length']))

    def __iter__(self):
        batch_idxes_list = []
        if self.recommendation_data_len:
            recommendation_ori_idx, recommendation_len = zip(*self.recommendation_data_len)
            recommendation_idx = zip(recommendation_len, np.random.permutation(len(self.recommendation_data_len)), recommendation_ori_idx)
            recommendation_idx = sorted(recommendation_idx, key=lambda e: (e[1] // self.pool_size, e[0]), reverse=True)
            for i in range(0, len(recommendation_idx), self.batch_size):
                batch_idxes_list.append([recommendation_idx_[2] for recommendation_idx_ in recommendation_idx[i:i+self.batch_size]])
        if self.search_data_len:
            search_ori_idx, search_len = zip(*self.search_data_len)
            search_idx = zip(search_len, np.random.permutation(len(self.search_data_len)), search_ori_idx)
            search_idx = sorted(search_idx, key=lambda e: (e[1] // self.pool_size, e[0]), reverse=True)
            for i in range(0, len(search_idx), self.batch_size):
                batch_idxes_list.append([search_idx_[2] for search_idx_ in search_idx[i:i+self.batch_size]])
        random.shuffle(batch_idxes_list)
        for batch_idxes in batch_idxes_list:
            yield batch_idxes

    def __len__(self):
        recommendation_batches = (len(self.recommendation_data_len) + self.batch_size - 1) // self.batch_size
        search_batches = (len(self.search_data_len) + self.batch_size - 1) // self.batch_size
        return recommendation_batches + search_batches


if __name__ == "__main__":
    # for test only
    pass