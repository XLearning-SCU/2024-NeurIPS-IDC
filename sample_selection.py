import os
import argparse
import random
import torch
import numpy as np
from utils.save import save_pl_samples
from evaluation.evaluation import get_y_preds
from utils import yaml_config_hook
from models import network
from evaluation import evaluation
from torch.utils import data
from utils.data import build_dataset


def inference(loader, model, device):
    model.eval()
    origin_vector = []
    embedding_vector = []
    embedding_512_vector = []
    prediction_vector = []
    confidence_vector = []
    labels_vector = []
    for step, (x, y, index) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            embedding = model.forward_embedding(x)
            embedding_512 = model.forward_embedding_512(x)
            c = model.forward_cluster(x)
            c2 = model.forward_cluster_confidence(x)
        origin = x.detach()
        embedding = embedding.detach()
        embedding_512 = embedding_512.detach()
        c = c.detach()
        c2 = c2.detach()
        origin_vector.extend(origin.cpu().detach().numpy())
        embedding_vector.extend(embedding.cpu().detach().numpy())
        embedding_512_vector.extend(embedding_512.cpu().detach().numpy())
        prediction_vector.extend(c.cpu().detach().numpy())
        confidence_vector.extend(c2.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    origin_vector = np.array(origin_vector)
    embedding_vector = np.array(embedding_vector)
    embedding_512_vector = np.array(embedding_512_vector)
    prediction_vector = np.array(prediction_vector)
    confidence_vector = np.array(confidence_vector)
    labels_vector = np.array(labels_vector)
    print("embeddings.shape {}".format(embedding_vector.shape))
    return origin_vector, embedding_vector, embedding_512_vector, prediction_vector, confidence_vector, labels_vector


def construct_matrix(loader, model, device):
    model.eval()
    with torch.no_grad():
        matrix = []
        for step, (inputs, _, _) in enumerate(loader):
            feature = model.forward_embedding_512(inputs.to(device))
            matrix.append(feature)
    return torch.cat(matrix, dim=0)


def kNN(x_train, x_test, K):
    assert len(x_train.shape) == 2
    assert len(x_test.shape) == 2
    N1, N2 = x_test.size(0), x_train.size(0)
    x_1 = torch.pow(x_test, 2).sum(1, keepdim=True).expand(N1, N2)
    x_2 = torch.pow(x_train, 2).sum(1, keepdim=True).expand(N2, N1).t()
    dist = x_1 + x_2
    dist.addmm_(x_test, x_train.t(), beta=1, alpha=-2)
    d_knn, ind_knn = torch.topk(dist, k=K, dim=1, largest=False, sorted=False)
    return ind_knn, d_knn


def partitioned_kNN(feats_list, K=20, partitions_size=130000):
    partitions = int(np.ceil(feats_list.shape[0] / partitions_size))
    print("Partitions:", partitions)

    # Assume the last partition has at least K elements
    ind_knns = torch.zeros(
        (feats_list.size(0), partitions * K), dtype=torch.long)
    d_knns = torch.zeros(
        (feats_list.size(0), partitions * K), dtype=torch.float)

    def get_sampled_data(ind):
        return feats_list[ind * partitions_size: (ind + 1) * partitions_size]

    for ind_i in range(partitions):  # ind_i: train dimension
        for ind_j in range(partitions):  # ind_j: test dimension
            print("Running with indices: {}, {}".format(ind_i, ind_j))
            x_train = get_sampled_data(ind_i).cuda()
            x_test = get_sampled_data(ind_j).cuda()

            ind_knn, d_knn = kNN(x_train, x_test, K=K)
            # ind_knn, d_knn: test dimension, K (indices: train dimension)
            ind_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size, ind_i * K: (ind_i + 1) * K] = \
                ind_i * partitions_size + ind_knn.cpu()
            d_knns[ind_j * partitions_size: (ind_j + 1) * partitions_size,
            ind_i * K: (ind_i + 1) * K] = d_knn.cpu()

            del ind_knn, d_knn, x_train, x_test

    d_sorted_inds = d_knns.argsort(dim=1)
    d_selected_inds = d_sorted_inds[:, :K]
    ind_knns_selected = torch.gather(
        ind_knns, dim=1, index=d_selected_inds)
    d_knns_selected = torch.gather(d_knns, dim=1, index=d_selected_inds)
    d_knns = d_knns_selected
    ind_knns = ind_knns_selected

    del ind_knns_selected, d_knns_selected

    return d_knns, ind_knns

def selection(matrix, hardness, budget, beta, gamma):
    selected_index = []
    remaining_index = torch.arange(matrix.shape[0])
    d_knns, ind_knns = partitioned_kNN(matrix)
    neighbors_dist = d_knns.mean(dim=1)
    representativeness = torch.log(1 / neighbors_dist)
    origin_score = representativeness + beta * hardness
    for i in range(budget):
        if i == 0:
            score = origin_score
        else:
            remaining_matrix = matrix[remaining_index]
            selected_matrix = matrix[selected_index]
            diversity = -torch.matmul(remaining_matrix, selected_matrix.t())
            diversity = torch.log(1+diversity)
            diversity = diversity.view(diversity.shape[0], -1)
            diversity = torch.min(diversity, dim=1).values
            score = origin_score[remaining_index] + gamma * diversity
        temp_index = torch.argmax(score)
        current_index = remaining_index[temp_index].item()
        print("---------第%d个：%d" % (i + 1, current_index))
        print("origin_score: %.2f, rep: %.2f, hard: %.2f" % (origin_score[current_index], representativeness[current_index], beta*hardness[current_index]))
        if i != 0:
            print("diversity: ", gamma * diversity[temp_index])
        selected_index.append(current_index)
        print("selected_index: ", selected_index)
        remaining_index = torch.cat((remaining_index[:temp_index], remaining_index[temp_index + 1:]))
        print("remaining_index.shape: ", remaining_index.shape)
    return selected_index


if __name__ == "__main__":
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

    data_loader_with_index = torch.utils.data.DataLoader(
        ft_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers
    )

    val_data_with_index = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers
    )

    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(1200))
    res = network.get_resnet(args)
    model = network.Network(res, args.feature_dim, class_num)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model_state_dict = torch.load(model_fp, map_location=device.type)['net']
    model_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(model_state_dict)

    print("### Creating features from model ###")

    # X1: pesudo label, X2: softmax probability
    origin, embedding, embedding_512, X1, X2, Y = inference(val_data_with_index, model, device)

    print("Y.shape: ", Y.shape)
    print(np.unique(Y))
    print("X1.shape: ", X1.shape)
    print(np.unique(X1))
    nmi, ari, f, acc = evaluation.evaluate(Y, X1)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))

    # cch计算聚类中心及标签
    label_adjusted = get_y_preds(X1, Y, len(set(Y)))
    label_adjusted = label_adjusted.astype(int)
    centers = np.zeros([class_num, 512])
    cluster_num = np.zeros(class_num)
    image_center = np.zeros((class_num, 224, 224, 3))

    for i in range(class_num):
        cluster_num[i] = np.sum(X1 == i)
        centers[i] = np.mean(embedding_512[X1 == i], axis=0)
    print(f"centers.shape: {centers.shape}")
    distances = np.zeros((Y.shape[0], class_num))

    # 余弦相似性衡量sample和类中心的接近程度, top1-top2越接近0越hard, hardness=1-(top1-top2)
    for i in range(class_num):
        distances[:, i] = np.dot(embedding_512, centers[i]) / np.linalg.norm(centers[i])
    print(f"distances.shape: {distances.shape}")
    cluster_center = np.argmax(distances, axis=0)
    print(f"cluster_center: {cluster_center}")

    for i in range(class_num):
        image_center[i] = np.transpose(origin[cluster_center[i]], (1, 2, 0))

    nearing_top1 = np.argsort(-distances, axis=1)[:, 0].reshape(-1, 1)
    nearing_top2 = np.argsort(-distances, axis=1)[:, :2]
    nearing_top = np.argsort(-distances, axis=1)

    hardness = torch.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        hardness[i] = distances[i, nearing_top2[i, -2]] - distances[i, nearing_top2[i, -1]]
    hardness = 1 - hardness
    hardness = torch.log(hardness)
    print(f'hardness.shape: {hardness.shape}')
    print(f"hardness: {hardness}")

    val_data_with_index = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers
    )

    matrix = construct_matrix(val_data_with_index, model, device)
    print(f"matrix.shape: {matrix.shape}")

    selection_result = selection(matrix.cpu(), hardness, budget=args.budget, beta=1., gamma=1.)

    pos_index = []
    pos_label = []
    pos_candidate = []
    neg_index = []
    neg_label = []
    selected_index = []
    selected_label = []
    pos_distribution_gt = np.zeros(class_num)
    pos_distribution_pre = np.zeros(class_num)
    neg_distribution_gt = np.zeros(class_num)
    neg_distribution_pre = np.zeros(class_num)

    # 直接根据匈牙利匹配对齐gt和聚类结果
    for i in selection_result:
        if label_adjusted[i] in nearing_top[i, :args.candidate_num]:
            pos_index.append(i)
            pos_label.append(label_adjusted[i])
            for j in range(args.candidate_num):
                if nearing_top[i, j] != label_adjusted[i]:
                    pos_candidate.append(nearing_top[i, j])
            pos_distribution_gt[label_adjusted[i]] += 1
            pos_distribution_pre[X1[i]] += 1
            selected_index.append(i)
            selected_label.append(label_adjusted[i])
        else:
            neg_index.append(i)
            for j in range(args.candidate_num):
                neg_label.append(nearing_top[i, j])
            neg_distribution_gt[label_adjusted[i]] += 1
            neg_distribution_pre[X1[i]] += 1
            selected_index.append(i)
            selected_label.append(-1)

    pos_index = torch.tensor(pos_index)
    pos_label = torch.tensor(pos_label)
    if args.candidate_num > 1:
        pos_candidate = torch.tensor(pos_candidate).view(-1, args.candidate_num-1)
    else:
        pos_candidate = None
    neg_index = torch.tensor(neg_index)
    neg_label = torch.tensor(neg_label).view(-1, args.candidate_num)
    selected_index = torch.tensor(selected_index)
    selected_label = torch.tensor(selected_label)

    print(f"candidate_num: {args.candidate_num}")
    print(f"pos_len: {len(pos_index)}; neg_len: {len(neg_index)}")
    print(f"pos_distribution_gt: {pos_distribution_gt}")
    print(f"pos_distribution_pre: {pos_distribution_pre}")
    print(f"neg_distribution_gt: {neg_distribution_gt}")
    print(f"neg_distribution_pre: {neg_distribution_pre}")
    print(f"distribution_gt: {pos_distribution_gt + neg_distribution_gt}")
    print(f"distribution_pre: {pos_distribution_pre + neg_distribution_pre}")

    save_path = str(args.candidate_num) + "-" + str(args.budget) + "-h1g1"
    print(save_path)
    save_pl_samples(args, pos_index, pos_label, pos_candidate, neg_index, neg_label, selected_index, selected_label, save_path)
















