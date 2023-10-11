'''
@author: Jiayi Xie (xjyxie@whu.edu.cn)
Pytorch Implementation of UnifiedSSR model in:
UnifiedSSR: A Unified Framework of Sequential Search and Recommendation
'''
import os
from collections import OrderedDict
import torch


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        model_state_file = os.path.join(model_dir, 'model_best.pth'.format(current_epoch))
    else:
        model_state_file = os.path.join(model_dir, 'model_{}.pth'.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'], False)
        # model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]
            # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def degrade_saved_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    save_path = os.path.join(os.path.dirname(model_path), 'degrade_version')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(checkpoint, os.path.join(save_path, os.path.basename(model_path)), _use_new_zipfile_serialization=False)


def early_stopping(cur_scores, best_scores, stopping_count, patient=100, logging=None):
    update_flag = False
    for cur_score, best_score in zip(cur_scores, best_scores):
        if cur_score > best_score:
            update_flag = True

    if update_flag == True:
        stopping_count = 0
        best_scores = cur_scores
    else: stopping_count += 1

    if stopping_count >= patient:
        if logging:
            logging.info("Early stopping is trigger at step: {}".format(stopping_count))
        should_stop = True
    else:
        if logging:
            logging.info("Current stopping count: {}".format(stopping_count))
        should_stop = False
    return best_scores, stopping_count, should_stop
