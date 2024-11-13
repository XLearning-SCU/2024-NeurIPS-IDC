import os
import argparse
import torch
import torchvision
import numpy as np
from sklearn import metrics
from torch.utils.data import Subset
from models import network
from utils import yaml_config_hook, transform
from evaluation import evaluation
from torch.utils import data
import copy


def inference(loader, model, device):
    model.eval()
    origin_vector = []
    embedding_vector = []
    prediction_vector = []
    confidence_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            embedding = model.forward_embedding(x)
            c = model.forward_cluster(x)
            c2 = model.forward_cluster_confidence(x)
        origin = x.detach()
        embedding = embedding.detach()
        c = c.detach()
        c2 = c2.detach()
        origin_vector.extend(origin.cpu().detach().numpy())
        embedding_vector.extend(embedding.cpu().detach().numpy())
        prediction_vector.extend(c.cpu().detach().numpy())
        confidence_vector.extend(c2.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 1 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    origin_vector = np.array(origin_vector)
    embedding_vector = np.array(embedding_vector)
    prediction_vector = np.array(prediction_vector)
    confidence_vector = np.array(confidence_vector)
    labels_vector = np.array(labels_vector)
    print("embeddings.shape {}".format(embedding_vector.shape))
    return origin_vector, embedding_vector, prediction_vector, confidence_vector, labels_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook.load_yaml_config("./configs/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    # 这两句顺序不能反
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(args.dataset)

    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.build_transform(is_train=False, args=args)
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            download=True,
            transform=transform.build_transform(is_train=False, args=args)
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.build_transform(is_train=False, args=args)
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.build_transform(is_train=False, args=args)
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "STL-10":
        train_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="train",
            download=True,
            transform=transform.build_transform(is_train=False, args=args)
        )
        test_dataset = torchvision.datasets.STL10(
            root=args.dataset_dir,
            split="test",
            download=True,
            transform=transform.build_transform(is_train=False, args=args)
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.build_transform(is_train=False, args=args)
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs/train',
            transform=transform.build_transform(is_train=False, args=args)
        )
        class_num = 15
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    res = network.get_resnet(args)
    model = network.Network(res, args.feature_dim, class_num)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(1200))
    # model_fp = os.path.join(args.model_path, "5-500-h1g1_{}.tar".format(100))
    print(model_fp)

    pl_sample_path = os.path.join(args.model_path, "pl_samples_{}.tar".format('500-h1g1'))
    print(pl_sample_path)

    pl_sample = torch.load(pl_sample_path)
    selected_index = pl_sample['selected_index']
    selected_label = pl_sample['selected_label']
    print("selected samples num: ", len(selected_index))

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    model_state_dict = torch.load(model_fp, map_location=device.type)['net']
    model_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(model_state_dict)
    model.to(device)

    print("### Creating features from model ###")

    # X1: pesudo label, X2: softmax probability
    origin, embedding, X1, X2, Y = inference(val_loader, model, device)

    if args.dataset == "CIFAR-100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        Y_copy = copy.copy(Y)
        for i in range(20):
            for j in super_label[i]:
                Y[Y_copy == j] = i

    nmi, ari, f, acc, pred_adjusted = evaluation.evaluate(Y, X1, return_adjusted=True)
    # 直接给挑选样本正确的标签
    for i in selected_index:
        pred_adjusted[i] = Y[i]
    nmi_new = metrics.normalized_mutual_info_score(Y, pred_adjusted)
    ari_new = metrics.adjusted_rand_score(Y, pred_adjusted)
    acc_new = metrics.accuracy_score(pred_adjusted, Y)
    print('NMI = {:.4f} ARI = {:.4f} ACC = {:.4f}'.format(nmi, ari, acc))
    print('NMI_new = {:.4f} ARI_new = {:.4f} ACC_new = {:.4f}'.format(nmi_new, ari_new, acc_new))









