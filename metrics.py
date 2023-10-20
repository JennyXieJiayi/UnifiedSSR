import torch
import numpy as np


def evaluate_product(trained_model, data_loader, total_sample_num, args, device='cuda'):
    trained_model.eval()
    k_list = args.k_list or [5, 10]
    hit_at_k = np.array([0] * len(k_list))
    ndcg_at_k = np.array([0] * len(k_list))

    with torch.no_grad():
        for idx, (cur_task, batch_data) in enumerate(data_loader):
            for key in batch_data.keys():
                if isinstance(batch_data[key], list):
                    batch_data[key] = [[i.to(device) for i in data] for data in batch_data[key]]
                else:
                    batch_data[key] = batch_data[key].to(device)

            out = trained_model.forward(cur_task, batch_data)

            if cur_task == 'recommendation':
                if args.corr_factor > 0:
                    sub_seq_wins = trained_model.get_sub_seq_wins(out)
                    out, _, _ = trained_model.intra_corr_loss(out, sub_seq_wins, batch_data['pids_mask'])

                if args.test_neg: # rank candidates
                    out = trained_model.next_product_predict(out, batch_data['pid_last_idx'], batch_data['pid_pred'])
                    tgt = torch.zeros((out.shape[0], 1)).long()
                else: # rank all products
                    out = trained_model.next_product_predict(out, batch_data['pid_last_idx'])
                    tgt = batch_data['pid_pred'][:,0].unsqueeze(-1).to('cpu')
            elif cur_task == 'search':
                p_out, q_out = out
                p_out = p_out[:, 1:, :]
                q_out = q_out[:, 1:, :]
                if args.corr_factor > 0:
                    mask = batch_data['pids_mask'][:, :, 1:]
                    p_sub_seq_wins = trained_model.get_sub_seq_wins(p_out)
                    q_sub_seq_wins = trained_model.get_sub_seq_wins(q_out)
                    p_out, q_out, _ = trained_model.inter_corr_loss(p_out, p_sub_seq_wins, q_out, q_sub_seq_wins, mask)

                if args.test_neg: # rank candidates
                    out = trained_model.next_product_search(p_out, q_out, batch_data['pid_last_idx'], batch_data['pid_pred'])
                    tgt = torch.zeros((out.shape[0], 1)).long()
                else: # rank all products
                    out = trained_model.next_product_search(p_out, q_out, batch_data['pid_last_idx'])
                    tgt = batch_data['pid_pred'][:,0].unsqueeze(-1).to('cpu')

            _, out_rank = torch.sort(out, descending=True)
            if out_rank.device.type == 'cuda':
                out_rank = out_rank.cpu()

            for idx_k, k in enumerate(k_list):
                hit_at_k[idx_k] += hit_at_k_per_batch(out_rank, tgt, k)
                ndcg_at_k[idx_k] += ndcg_at_k_per_batch(out_rank, tgt, k)

    hit_at_k = hit_at_k / total_sample_num
    ndcg_at_k = ndcg_at_k / total_sample_num
    return hit_at_k, ndcg_at_k


def hit_at_k_per_batch(pred, tgt, k):
    hits_num = 0
    for i in range(len(tgt)):
        tgt_set = set(tgt[i].numpy())
        pred_set = set(pred[i][:k].numpy())
        hits_num += len(tgt_set & pred_set)
    return hits_num


def recall_at_k_per_batch(pred, tgt, k):
    sum_recall = 0.
    num_sample = 0
    for i in range(len(tgt)):
        tgt_set = set(tgt[i].numpy())
        pred_set = set(pred[i][:k].numpy())
        if len(tgt_set) != 0:
            sum_recall += len(tgt_set & pred_set) / float(len(tgt_set))
            num_sample += 1
    return num_sample, sum_recall


def ndcg_at_k_per_batch(pred, tgt, k):
    ndcg_score = 0.
    for i in range(len(tgt)):
        sample_pred = pred[i, :k].numpy()
        sample_tgt = tgt[i].numpy()
        ndcg_score += ndcg_at_k_per_sample(sample_pred, sample_tgt)
    return ndcg_score


def ndcg_at_k_per_sample(pred, tgt, method=1):
    r = np.zeros_like(pred, dtype=np.float32)
    ideal_r = np.zeros_like(pred, dtype=np.float32)
    for i, v in enumerate(pred):
        if v in tgt and v not in pred[:i]:
            r[i] = 1.
    ideal_r[:len(tgt)] = 1.

    idcg = dcg_at_k_per_sample(ideal_r, method)
    if not idcg:
        return 0.
    return dcg_at_k_per_sample(r, method) / idcg


def dcg_at_k_per_sample(r, method=1):
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1: # 01相关性，仅相关时(r=1)有相加
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


if __name__ == "__main__":
    # for test only
    pass