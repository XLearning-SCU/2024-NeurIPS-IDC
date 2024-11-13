import datetime
import os
import argparse
import random
import torch
import numpy as np
from utils import yaml_config_hook
from models import network
from evaluation import evaluation
from torch.utils import data
from torch.utils.data import Subset
from utils.data import build_dataset


def get_confidence_max(loader, model, return_all=False):
    print("Computing confidence")
    model.eval()
    index_vector = []
    confidence_vector = []
    label_vector = []
    for step, (x, y, origin_index) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            con = model.forward_cluster_confidence(x, soft=True)
        index_vector.extend(origin_index.cpu().detach().numpy())
        confidence_vector.extend(con.cpu().detach().numpy())
        label_vector.extend(y.numpy())
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]")
    index_vector = torch.tensor(np.array(index_vector))
    confidence_vector = torch.tensor(np.array(confidence_vector))
    label_vector = torch.tensor(np.array(label_vector))
    temp = torch.max(confidence_vector, dim=1)
    confidence_max = temp[0]
    prediction_vector = temp[1]
    if return_all:
        return index_vector, prediction_vector, confidence_vector, label_vector
    return index_vector, prediction_vector, confidence_max, label_vector


# 大于threshold_in选入，小于threshold_out淘汰
def generate_easy_samples(c, pseudo_label_cur, threshold_out, threshold_in, class_num):
    gamma = 0.2

    batch_size = c.shape[0]
    pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
    tmp = torch.arange(0, batch_size).to(device)

    prediction = c.argmax(dim=1)
    confidence = c.max(dim=1).values
    unconfident_pred_index = confidence < threshold_out
    pseudo_per_class = np.ceil(batch_size / class_num * gamma).astype(int)

    for i in range(class_num):
        class_idx = prediction == i
        if class_idx.sum() == 0:
            continue
        confidence_class = confidence[class_idx]
        num = min(confidence_class.shape[0], pseudo_per_class)
        confident_idx = torch.argsort(-confidence_class)
        for j in range(num):
            idx = tmp[class_idx][confident_idx[j]]
            if confidence[idx] > threshold_in:
                pseudo_label_nxt[idx] = i
            else:
                break

    todo_index = pseudo_label_cur == -1
    pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
    pseudo_label_nxt = pseudo_label_cur
    pseudo_label_nxt[unconfident_pred_index] = -1
    return pseudo_label_nxt


def fine_tuning_together(epoch, pseudo_label, model, device, class_num):
    print(f"----------epoch {epoch}----------")
    loss_epoch = 0
    pos_epoch = 0
    neg_epoch = 0
    high_epoch = 0

    if epoch % 1 == 0:
        index_vector, prediction_vector, confidence_vector, label_vector = get_confidence_max(val_data_with_index,
                                                                                              model, return_all=True)
        nmi_all, ari_all, _, acc_all = evaluation.evaluate(label_vector.numpy(), prediction_vector.numpy())
        print("nmi_all: {:.2f}, acc_all: {:.2f}, ari_all: {:.2f}".format(nmi_all * 100, acc_all * 100, ari_all * 100))

    unlabeled_iter = iter(data_loader_with_index)
    labeled_iter = iter(selected_loader)
    for step in range(len(labeled_iter)):
        optimizer.zero_grad()
        try:
            (unlabeled_weak, unlabeled_strong, unlabeled_val), _, unlabeled_index = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(data_loader_with_index)
            (unlabeled_weak, unlabeled_strong, unlabeled_val), _, unlabeled_index = next(unlabeled_iter)

        try:
            (labeled_weak, labeled_strong, labeled_val), _, labeled_index = next(labeled_iter)
        except:
            labeled_iter = iter(selected_loader)
            (labeled_weak, labeled_strong, labeled_val), _, labeled_index = next(labeled_iter)

        unlabeled_strong, unlabeled_val = unlabeled_strong.to(device), unlabeled_val.to(device)
        labeled_strong = labeled_strong.to(device)

        model.eval()
        with torch.no_grad():
            # 动态更新高置信度样本
            high_val = model.forward_cluster_confidence(unlabeled_val, soft=True)
            pseudo_label_cur = generate_easy_samples(high_val, pseudo_label[unlabeled_index], threshold_in=0.99, threshold_out=0.99, class_num=class_num)
            pseudo_label[unlabeled_index] = pseudo_label_cur
            index_cur = pseudo_label_cur != -1

        model.train()

        high_strong = model.forward_cluster_confidence(unlabeled_strong[index_cur], soft=False)
        high_label = pseudo_label_cur[index_cur].to(device).to(torch.long)
        idx, counts = torch.unique(high_label, return_counts=True)
        freq = high_label.shape[0] / counts.float()
        weight = torch.ones(class_num).to(device)
        weight[idx] = freq
        high_loss = torch.nn.functional.cross_entropy(high_strong, high_label, weight=weight)
        high_epoch += high_loss.item()

        # 将batch的索引(0-数据集大小-1)转换为现有的索引(0-labeled_num-1)
        selected_index_current = torch.nonzero(torch.eq(selected_index.unsqueeze(0), labeled_index.unsqueeze(1)))[:, 1]
        selected_label_current = selected_label[selected_index_current]
        selected_index_pos_current = torch.where(selected_label_current != -1)[0]
        selected_index_neg_current = torch.where(selected_label_current == -1)[0]

        pos_index_current = torch.nonzero(torch.eq(pos_index.unsqueeze(0), labeled_index.unsqueeze(1)))[:, 1]
        pos_label_current = pos_label[pos_index_current].to(device).to(torch.long)
        neg_index_current = torch.nonzero(torch.eq(neg_index.unsqueeze(0), labeled_index.unsqueeze(1)))[:, 1]
        neg_label_current = neg_label[neg_index_current]
        random_num = torch.randint(0, args.candidate_num, size=(neg_label_current.shape[0], 1))
        neg_label_current = torch.gather(neg_label_current, 1, random_num).view(-1)
        neg_label_current = neg_label_current.to(device).to(torch.long)

        pos_strong = model.forward_cluster_confidence(labeled_strong[selected_index_pos_current], soft=False)
        idx_pos, counts_pos = torch.unique(pos_label_current, return_counts=True)
        freq_pos = pos_label_current.shape[0] / counts_pos.float()
        weight_pos = torch.ones(class_num).to(device)
        weight_pos[idx_pos] = freq_pos
        pos_loss = torch.nn.functional.cross_entropy(pos_strong, pos_label_current, weight=weight_pos)
        pos_epoch += pos_loss.item()
        # 正类用y*log(p)求和，负类用y*log(1-p)求和
        if len(selected_index_neg_current) != 0:
            neg_strong = model.forward_cluster_confidence(labeled_strong[selected_index_neg_current], soft=True)
            neg_strong = torch.log(torch.clamp_min((1 - neg_strong), 1e-5))
            neg_loss = torch.nn.functional.nll_loss(neg_strong, neg_label_current)
            neg_epoch += neg_loss.item()
        else:
            neg_loss = 0

        loss = high_loss + pos_loss + neg_loss

        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    high_confidence_index = pseudo_label != -1
    high_confidence_index = high_confidence_index.cpu()
    high_confidence_label = pseudo_label[high_confidence_index].cpu()

    idx_first, counts_first = torch.unique(high_confidence_label, return_counts=True)
    increase_first = torch.zeros(class_num)
    increase_first[idx_first] = counts_first.cpu().float()
    print(f"high_confidence: {increase_first}")
    print(f"high_confidence_num: {high_confidence_label.shape[0]}")

    print(f"epoch: {epoch}, loss: {loss_epoch}, high_loss: {high_epoch}, pos_loss: {pos_epoch}, neg_loss: {neg_epoch}")

    if (epoch+1) % 100 == 0:
        index_vector, prediction_vector, confidence_vector, label_vector = get_confidence_max(val_data_with_index,
                                                                                              model, return_all=True)
        nmi_all, ari_all, _, acc_all = evaluation.evaluate(label_vector.numpy(), prediction_vector.numpy())
        print("nmi_all: {:.2f}, acc_all: {:.2f}, ari_all: {:.2f}".format(nmi_all * 100, acc_all * 100, ari_all * 100))


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    print("开始时间：", start_time)

    parser = argparse.ArgumentParser()
    config = yaml_config_hook.load_yaml_config("./configs/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    # 这两句顺序不能反
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(args.dataset)

    ft_data, class_num = build_dataset(type="train", args=args)
    val_data, _ = build_dataset(type="test", args=args)

    batch_size = min(100, args.budget)

    data_loader_with_index = torch.utils.data.DataLoader(
        ft_data,
        batch_size=5*batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )

    val_data_with_index = torch.utils.data.DataLoader(
        val_data,
        batch_size=250,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers
    )

    print(f"candidate_num: {args.candidate_num}")
    save_path = str(args.candidate_num) + "-" + str(args.budget) + "-h1g1"
    pl_sample_path = os.path.join(args.model_path, "pl_samples_{}.tar".format(save_path))
    print(pl_sample_path)

    pl_sample = torch.load(pl_sample_path)
    pos_index = pl_sample['pos_index']
    pos_label = pl_sample['pos_label']
    neg_index = pl_sample['neg_index']
    neg_label = pl_sample['neg_label']
    selected_index = pl_sample['selected_index']
    selected_label = pl_sample['selected_label']
    print("positive samples num: ", len(pos_index))
    print("negative samples num: ", len(neg_index))
    print("selected samples num: ", len(selected_index))

    selected_dataset = Subset(ft_data, selected_index)
    selected_loader = torch.utils.data.DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )

    res = network.get_resnet(args)
    model = network.Network(res, args.feature_dim, class_num)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(1200))
    model_state_dict = torch.load(model_fp, map_location=device.type)['net']
    model_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(model_state_dict)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    epochs = 100
    print(f"epochs: {epochs}")
    pseudo_label = -torch.ones(len(ft_data), dtype=torch.long).to(device)
    for i in range(epochs):
        fine_tuning_together(i, pseudo_label, model, device, class_num)
        if (i + 1) % 100 == 0:
            path = os.path.join(args.model_path, "5-500-h1g1_{}.tar".format(i + 1))
            print(path)

            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, path)
    end_time = datetime.datetime.now()
    print("结束时间：", end_time)
