import os.path as osp
from torchvision import datasets
from diffusion.utils import DATASET_ROOT, get_classes_templates
from diffusion.dataset.objectnet import ObjectNetBase
from diffusion.dataset.imagenet import ImageNet as ImageNetBase
from typing import Any, Callable, cast, Optional, Tuple

import random
import numpy as np
import os
seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    class_to_idx = {str(i): i for i in range(10)}


def get_target_dataset(name: str, train=False, transform=None, target_transform=None):
    """Get the torchvision dataset that we want to use.
    If the dataset doesn't have a class_to_idx attribute, we add it.
    Also add a file-to-class map for evaluation
    """

    if name == "cifar10":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=transform,
                                   target_transform=target_transform, download=True)
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root=DATASET_ROOT, train=train, transform=transform,
                                    target_transform=target_transform, download=True)
    elif name == "stl10":
        dataset = datasets.STL10(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                 target_transform=target_transform, download=True)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name == "toy":
        class STL10_toy(datasets.STL10):
            base_folder = "stl10_binary"
            url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
            filename = "stl10_binary.tar.gz"
            tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
            class_names_file = "class_names.txt"
            folds_list_file = "fold_indices.txt"
            train_list = [
                ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
                ["train_y.bin", "5a34089d4802c674881badbb80307741"],
                ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
            ]

            test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
            splits = ("train", "train+unlabeled", "unlabeled", "test")
            
            def __init__(
                self, root: str, 
                split: str = "train", 
                folds: Optional[int] = None, 
                transform: Optional[Callable] = None, 
                target_transform: Optional[Callable] = None, 
                download: bool = False
                ) -> None:
                super().__init__(root, split, folds, transform, target_transform, download)
                        
            def __getitem__(self, index: int) -> Tuple[Any, Any]:
                return super().__getitem__(index * 16)
            
            def __len__(self) -> int:
                return int(self.data.shape[0] / 16)
            
        dataset = STL10_toy(root=DATASET_ROOT, split='train' if train else 'test', transform=transform,
                                target_transform=target_transform, download=True)
    elif name == "pets":
        dataset = datasets.OxfordIIITPet(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                         target_transform=target_transform, download=True)

        # lower case every key in the class_to_idx
        dataset.class_to_idx = {k.lower(): v for k, v in dataset.class_to_idx.items()}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._images, dataset._labels)}
    elif name == "flowers":
        dataset = datasets.Flowers102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                      target_transform=target_transform, download=True)
        classes = list(get_classes_templates('flowers')[0].keys())  # in correct order
        dataset.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}
    elif name == "aircraft":
        dataset = datasets.FGVCAircraft(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                        target_transform=target_transform, download=True)

        # replace backslash with underscore -> need to be dirs
        dataset.class_to_idx = {
            k.replace('/', '_'): v
            for k, v in dataset.class_to_idx.items()
        }

        dataset.file_to_class = {
            fn.split("/")[-1].split(".")[0]: lab
            for fn, lab in zip(dataset._image_files, dataset._labels)
        }
        # dataset.file_to_class = {
        #     fn.split("/")[-1].split(".")[0]: lab
        #     for fn, lab in zip(dataset._image_files, dataset._labels)
        # }

    elif name == "food":
        dataset = datasets.Food101(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                   target_transform=target_transform, download=True)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    elif name == "eurosat":
        if train:
            raise ValueError("EuroSAT does not have a train split.")
        dataset = datasets.EuroSAT(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
                                   download=True)
    elif name == 'imagenet':
        assert not train
        base = ImageNetBase(transform, location=DATASET_ROOT)
        dataset = datasets.ImageFolder(root=osp.join(DATASET_ROOT, 'imagenet/val'), transform=transform,
                                       target_transform=target_transform)
        dataset.class_to_idx = None  # {cls: i for i, cls in enumerate(base.classnames)}
        dataset.classes = base.classnames
        dataset.file_to_class = None
    elif name == 'objectnet':
        base = ObjectNetBase(transform, DATASET_ROOT)
        dataset = base.get_test_dataset()
        dataset.class_to_idx = dataset.label_map
        dataset.file_to_class = None  # todo
    elif name == "caltech101":
        if train:
            raise ValueError("Caltech101 does not have a train split.")
        dataset = datasets.Caltech101(root=DATASET_ROOT, target_type="category", transform=transform,
                                      target_transform=target_transform, download=True)

        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.categories)}
        dataset.file_to_class = {str(idx): dataset.y[idx] for idx in range(len(dataset))}
    elif name == "mnist":
        dataset = MNIST(root=DATASET_ROOT, train=train, transform=transform, target_transform=target_transform,
                        download=True)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    if name in {'mnist', 'cifar10', 'stl10', 'aircraft', 'cifar100'}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1]
            for idx in range(len(dataset))
        }
    elif name in {'toy'}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1] for idx in range(len(dataset))
        }
        dataset.class_to_idx = {
            cls: i for i, cls in enumerate(dataset.classes)
        }
        # print(dataset.file_to_class)
        # {0: 41, 1: 53, 2: 46, 3: 58, 4: 64, 5: 36, 6: 57, 7: 59, 8: 38, 9: 48}

    assert hasattr(dataset, "class_to_idx"), f"Dataset {name} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {name} does not have a file_to_class attribute."
    return dataset
