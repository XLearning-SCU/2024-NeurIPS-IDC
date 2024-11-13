import os
import torchvision
from typing import Any, Callable, Optional, Tuple, Dict, List
from PIL import Image
from torchvision.datasets.folder import default_loader, make_dataset, find_classes
from utils.transform import build_transform
from torch.utils import data
import numpy as np
import utils.image_folder


class STL10(torchvision.datasets.STL10):
    def __init__(
            self,
            root: str,
            split: str = "train",
            folds: Optional[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        super(STL10, self).__init__(root, split, folds, transform, target_transform, download)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.split == 'train':
            return img, target, index
        else:
            return img, target, index + 5000


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        super(CIFAR10, self).__init__(
            root, train, transform, target_transform, download
        )
        self.train = train

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target, index
        else:
            return img, target, index + 50000


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        super(CIFAR100, self).__init__(
            root, train, transform, target_transform, download
        )
        new_ = self.targets
        for idx, target in enumerate(self.targets):
            new_[idx] = _cifar100_to_cifar20(target)
        self.targets = new_


def _cifar100_to_cifar20(target):
    _dict = \
        {0: 4,
         1: 1,
         2: 14,
         3: 8,
         4: 0,
         5: 6,
         6: 7,
         7: 7,
         8: 18,
         9: 3,
         10: 3,
         11: 14,
         12: 9,
         13: 18,
         14: 7,
         15: 11,
         16: 3,
         17: 9,
         18: 7,
         19: 11,
         20: 6,
         21: 11,
         22: 5,
         23: 10,
         24: 7,
         25: 6,
         26: 13,
         27: 15,
         28: 3,
         29: 15,
         30: 0,
         31: 11,
         32: 1,
         33: 10,
         34: 12,
         35: 14,
         36: 16,
         37: 9,
         38: 11,
         39: 5,
         40: 5,
         41: 19,
         42: 8,
         43: 8,
         44: 15,
         45: 13,
         46: 14,
         47: 17,
         48: 18,
         49: 10,
         50: 16,
         51: 4,
         52: 17,
         53: 4,
         54: 2,
         55: 0,
         56: 17,
         57: 4,
         58: 18,
         59: 17,
         60: 10,
         61: 3,
         62: 2,
         63: 12,
         64: 12,
         65: 16,
         66: 12,
         67: 1,
         68: 9,
         69: 19,
         70: 2,
         71: 10,
         72: 0,
         73: 1,
         74: 16,
         75: 12,
         76: 9,
         77: 13,
         78: 15,
         79: 13,
         80: 16,
         81: 19,
         82: 2,
         83: 4,
         84: 6,
         85: 19,
         86: 5,
         87: 5,
         88: 8,
         89: 19,
         90: 18,
         91: 1,
         92: 2,
         93: 15,
         94: 6,
         95: 0,
         96: 17,
         97: 8,
         98: 14,
         99: 13}

    return _dict[target]


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class DatasetFolder(torchvision.datasets.VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


def build_dataset(type, args):
    is_train = type == "train"
    transform = build_transform(is_train, args)
    root = args.dataset_dir

    if args.dataset == "CIFAR-10":
        class_num = 10
        dataset = data.ConcatDataset(
            [
                CIFAR10(
                    root=root,
                    train=True,
                    download=False,
                    transform=transform,
                ),
                CIFAR10(
                    root=root,
                    train=False,
                    download=False,
                    transform=transform,
                ),
            ]
        )
    elif args.dataset == "CIFAR-100":
        class_num = 20
        dataset = data.ConcatDataset(
            [
                CIFAR100(
                    root=root,
                    train=True,
                    download=False,
                    transform=transform,
                ),
                CIFAR100(
                    root=root,
                    train=False,
                    download=False,
                    transform=transform,
                ),
            ]
        )
    elif args.dataset == "ImageNet-10":
        class_num = 10
        dataset = utils.image_folder.ImageFolder(root=os.path.join(root, "imagenet-10"), transform=transform)
    elif args.dataset == "ImageNet-dogs":
        class_num = 15
        dataset = utils.image_folder.ImageFolder(root=os.path.join(root, "imagenet-dogs/train"), transform=transform)
    elif args.dataset == "STL-10":
        class_num = 10
        dataset = data.ConcatDataset(
            [
                STL10(
                    root=root,
                    split="train",
                    download=False,
                    transform=transform,
                ),
                STL10(
                    root=root,
                    split="test",
                    download=False,
                    transform=transform,
                ),
            ]
        )
    else:
        raise NotImplementedError

    return dataset, class_num
