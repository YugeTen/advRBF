import torch
import os
import numpy as np

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing

import pickle
from utils import calculate_normalisation_params


class CATDOGDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        # list of data/catvdog/cat.0.jpg --> cat
        self.label_names = [os.path.split(filename)[-1].split('.')[0] for filename in self.filenames]

        # convert list of string --> list of numeric classes
        le = preprocessing.LabelEncoder()
        le.fit(list(set(self.label_names)))
        self.labels = list(le.transform(self.label_names))

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def get_loader(data_dir,
               batch_size,
               dataset="cifar-10",
               random_seed=1,
               shuffle=True,
               trial=1):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    if len(os.listdir(data_dir)) == 0 or 'cifar' in data_dir:
        normalise_vector = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    else:
        normalise_vector = calculate_normalisation_params(data_dir)
    normalize = transforms.Normalize(normalise_vector[0],
                                     normalise_vector[1])

    # define transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    if dataset == 'catvdog':
        train_dataset = CATDOGDataset(data_dir=os.path.join(data_dir, dataset),
                                               transform=train_transform)

        test_dataset = CATDOGDataset(data_dir=os.path.join(data_dir, dataset),
                                       transform=test_transform)
    elif dataset == 'cifar-10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                         download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=test_transform)
    elif dataset == 'cifar-100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                          download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                         download=True, transform=test_transform)
    else:
        raise ('unknown dataset')

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    # cross validation splits
    nfolds = 10
    n = dataset_size // nfolds
    start_idx = n * (trial - 1)
    end_idx = start_idx + n

    train_indices, test_indices = indices[:start_idx] + indices[end_idx:], \
                                  indices[start_idx:end_idx]

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=4,
    )

    data_loader = {}
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader

    print("Loaded {} examples, {} used for training and {} for testing.".format(dataset_size,
                                                                                len(train_indices),
                                                                                len(test_indices)))

    return data_loader


