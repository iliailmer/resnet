from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.util import img_as_float32
from skimage.transform import resize
from skimage import color
import torch
from torch.utils.data import Dataset as Dataset
from sklearn.utils import class_weight
import numpy as np
from additions import rescale
from torchvision.transforms import ToTensor


class Loader(Dataset):
    def __init__(self, image_path_dict, labels, image_name=None,
                 train=True, transform=None, color_space='rgb'):
        """
        Args:

        """
        data = list(image_path_dict.keys())  # image ids
        self.path = image_path_dict
        self.train = train
        # self.name = image_name
        self.transform = transform  # augmentation transforms
        self.train_names, self.test_names, \
        self.train_labels, self.test_labels = train_test_split(
            np.asarray(data),
            np.asarray(labels),
            test_size=0.2)
        self.color_transform_dict = {
            'rgb': color.rgb2rgbcie,
            'hed': color.rgb2hed,
            'hsv': color.rgb2hsv, None: None}

        if self.train:
            self.weights = class_weight.compute_class_weight(
                'balanced',
                np.unique(
                    self.train_labels),
                self.train_labels)
            if self.color_transform_dict[color_space] is not None:
                self.train_data = np.asarray([
                    rescale(
                        resize(
                            self.color_transform_dict[color_space](
                                rescale(
                                    io.imread(
                                        self.path[name])
                                        .astype('float32'))),
                            (64, 64), anti_aliasing=True,  # (150,150)
                            mode='reflect')) for name in self.train_names])
            else:
                self.train_data = np.asarray([
                    rescale(
                        resize(
                            rescale(
                                io.imread(self.path[name])
                                    .astype('float32')),
                            (64, 64), anti_aliasing=True,
                            mode='reflect')) for name in self.train_names])
            self.train_labels = torch.from_numpy(self.train_labels)

        else:
            self.weights = class_weight.compute_class_weight(
                'balanced',
                np.unique(
                    self.test_labels),
                self.test_labels)
            if self.color_transform_dict[color_space] is not None:
                self.test_data = np.asarray([
                    rescale(
                        resize(
                            self.color_transform_dict[color_space](
                                rescale(
                                    io.imread(
                                        self.path[name])
                                        .astype('float32'))),
                            (64, 64), anti_aliasing=True,
                            mode='reflect')) for name in self.test_names])
            else:
                self.test_data = np.asarray([
                    rescale(
                        resize(rescale(
                            io.imread(self.path[name])
                                .astype('float32')),
                            (64, 64), anti_aliasing=True,
                            mode='reflect')) for name in self.test_names])
            self.test_labels = torch.from_numpy(self.test_labels)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            if self.transform is not None:
                image = ToTensor()(self.transform(
                    **{'image': self.train_data[index]})['image'])
                label = self.train_labels[index]
            else:
                image, label = ToTensor()(
                    self.train_data[index]), self.train_labels[index]
        else:
            image, label = ToTensor()(
                self.test_data[index]), self.test_labels[index]
        return image, label


class LoaderSmall(Dataset):
    def __init__(self, image_path_dict, labels, image_name=None,
                 train=True, transform=None, color_space='rgb'):
        """
        Args:

        """
        data = list(image_path_dict.keys())  # image ids
        self.path = image_path_dict
        self.train = train
        # self.name = image_name
        self.transform = transform  # augmentation transforms
        self.train_names, self.test_names, \
        self.train_labels, self.test_labels = train_test_split(
            np.asarray(data),
            np.asarray(labels),
            test_size=0.15)
        self.color_transform_dict = {
            'rgb': color.rgb2rgbcie,
            'hed': color.rgb2hed,
            'hsv': color.rgb2hsv,
            'lab': color.rgb2lab,
            'lbp': self.rgb_lbp,
            None: None}

        if self.train:
            self.weights = class_weight.compute_class_weight(
                'balanced',
                np.unique(
                    self.train_labels),
                self.train_labels)
            if self.color_transform_dict[color_space] is not None:
                self.train_data = np.asarray([
                    self.color_transform_dict[color_space](
                        img_as_float32(io.imread(
                            self.path[name]))) for name in self.train_names])
            else:
                self.train_data = np.asarray([
                    img_as_float32(io.imread(
                        self.path[name])) for name in self.train_names])
            self.train_labels = torch.from_numpy(self.train_labels)
            # self.train_data = (self.train_data-self.train_data.mean())/self.train_data.std()

        else:
            self.weights = class_weight.compute_class_weight(
                'balanced',
                np.unique(
                    self.test_labels),
                self.test_labels)
            if self.color_transform_dict[color_space] is not None:
                self.test_data = np.asarray([
                    self.color_transform_dict[color_space](
                        img_as_float32(io.imread(
                            self.path[name]))) for name in self.test_names])
            else:
                self.test_data = np.asarray([
                    img_as_float32(io.imread(
                        self.path[name])) for name in self.test_names])
            self.test_labels = torch.from_numpy(self.test_labels)
            # self.test_data = (self.test_data - self.test_data.mean()) / self.test_data.std()

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def rgb_lbp(self, image, P=9, R=1):
        result = np.zeros_like(image)
        result[..., 0] = local_binary_pattern(image[..., 0], P=P, R=R, method='uniform')
        result[..., 1] = local_binary_pattern(image[..., 1], P=P, R=R, method='uniform')
        result[..., 2] = local_binary_pattern(image[..., 2], P=P, R=R, method='uniform')
        return rescale(result)

    def __getitem__(self, index):
        if self.train:
            if self.transform is not None:
                image = ToTensor()(self.transform(
                    **{'image': self.train_data[index].astype(np.float32)})['image'])
                label = self.train_labels[index]
            else:
                image, label = ToTensor()(
                    self.train_data[index].astype(np.float32)), self.train_labels[index]
        else:
            image, label = ToTensor()(
                self.test_data[index].astype(np.float32)), self.test_labels[index]
        return image, label
