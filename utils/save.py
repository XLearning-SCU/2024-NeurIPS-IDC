import os
import torch


def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def save_hard_samples(args, hard_index, hard_label, name):
    path = os.path.join(args.model_path, "hard_samples_{}.tar".format(name))
    hard_sample = {'hard_index': hard_index, 'hard_label': hard_label}
    torch.save(hard_sample, path)


def save_easy_hard(args, hard_index, easy_index, name):
    path = os.path.join(args.model_path, "easy_hard_{}.tar".format(name))
    samples = {'hard_index': hard_index, 'easy_index': easy_index}
    torch.save(samples, path)


def save_pl_samples(args, pos_index, pos_label, pos_candidate, neg_index, neg_label, selected_index, selected_label, name):
    path = os.path.join(args.model_path, "pl_samples_{}.tar".format(name))
    pl_sample = {'pos_index': pos_index, 'pos_label': pos_label, 'pos_candidate': pos_candidate, 'neg_index': neg_index, 'neg_label': neg_label, 'selected_index': selected_index, 'selected_label': selected_label}
    torch.save(pl_sample, path)
