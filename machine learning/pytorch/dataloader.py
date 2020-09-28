import os
import pandas as pd

from PIL.Image import open, new
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from .utils import m_rgb_to_hsv, rgb_to_lab


class ImageCSVLoader(Dataset):

    def __init__(self, transform, train_data_path, train_label_path, is_grey_scale):
        self.transform = transform
        self.is_grey_scale = is_grey_scale

        x_img_name = os.listdir(train_data_path)
        y_label = pd.read_csv(train_label_path, header=0)
        y_label = y_label['label']  # label column

        x_img_path = list()
        for item in x_img_name:
            x_img_path.append(train_data_path + '/' + item)

        self.len = len(x_img_name)
        self.x_img_path = x_img_path
        self.y_label = y_label

    def __getitem__(self, index):
        new_img = open(self.x_img_path[index])

        if not self.is_grey_scale:
            rgb_img = new("RGB", new_img.size)
            rgb_img.paste(new_img)

        out_img = self.transform(new_img)

        return out_img, self.y_label[index]     # data, target

    def __len__(self):
        return self.len


class ImageImageLoader(Dataset):

    def __init__(self, transform, train_x_path, train_y_path, is_grey_scale, input_size, hsv_mode):
        self.transform = transform
        self.is_grey_scale = is_grey_scale
        self.input_size = input_size
        self.hsv_mode = hsv_mode

        x_img_name = os.listdir(train_x_path)
        y_img_name = os.listdir(train_y_path)

        self.x_img_path = []
        self.y_img_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)

        img_paths = zip(x_img_name, y_img_name)
        for item in img_paths:
            self.x_img_path.append(train_x_path + os.sep + item[0])
            self.y_img_path.append(train_y_path + os.sep + item[1])

        self.len = len(x_img_name)

        del x_img_name
        del y_img_name

    def __getitem__(self, index):
        new_img_x = open(self.x_img_path[index])
        new_img_y = open(self.y_img_path[index])

        if not self.is_grey_scale:

            if self.hsv_mode:
                # RGB to HSV value
                new_img_x = m_rgb_to_hsv(new_img_x.resize((self.input_size, self.input_size)))
                new_img_y = m_rgb_to_hsv(new_img_y.resize((self.input_size, self.input_size)))

            else:
                new_img_x = new_img_x.resize((self.input_size, self.input_size))
                new_img_y = new_img_y.resize((self.input_size, self.input_size))

        out_img_x = self.transform(new_img_x)
        out_img_y = self.transform(new_img_y)

        # S,V channel only
        if self.hsv_mode and self.input_size == 2:
            out_img_x = out_img_x[1:3]
            out_img_y = out_img_y[1:3]

        return out_img_x, out_img_y    # data, target

    def __len__(self):
        return self.len


# main val data loader
class ValidationLoader:

    def __init__(self, dataset_path, label_path, input_size, is_grey_scale, hsv_mode, batch_size=64, num_workers=0, pin_memory=True):
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.validation_data_path = dataset_path
        self.validation_label_path = label_path
        self.is_grey_scale = is_grey_scale
        self.hsv_mode = hsv_mode

        # Data augmentation and normalization
        self.validation_trans = transforms.Compose([transforms.Resize(self.input_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                    ])

        self.ValidationDataLoader = DataLoader(ImageImageLoader(self.validation_trans,
                                                                self.validation_data_path,
                                                                self.validation_label_path,
                                                                self.is_grey_scale,
                                                                self.input_size,
                                                                self.hsv_mode),
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=pin_memory)

    def __len__(self):
        return self.ValidationDataLoader.__len__()


# main train data loader
class DTrainLoader:

    def __init__(self, dataset_path, label_path, input_size, is_grey_scale, hsv_mode, batch_size=64, num_workers=0, pin_memory=True):
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_data_path = dataset_path
        self.train_label_path = label_path
        self.is_grey_scale = is_grey_scale
        self.hsv_mode = hsv_mode

        # # Data augmentation and normalization
        self.train_trans = transforms.Compose([transforms.Resize((self.input_size, self.input_size)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomPerspective(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ])

        # use your own data loader
        self.TrainDataLoader = DataLoader(ImageImageLoader(self.train_trans,
                                                           self.train_data_path,
                                                           self.train_label_path,
                                                           self.is_grey_scale,
                                                           self.input_size,
                                                           self.hsv_mode),
                                          batch_size=self.batch_size,
                                          num_workers=self.num_workers,
                                          shuffle=True,
                                          pin_memory=self.pin_memory)

    def __len__(self):
        return self.TrainDataLoader.__len__()


# main train data loader
class GTrainLoader:

    def __init__(self, dataset_path, label_path, input_size, is_grey_scale, hsv_mode, batch_size=64, num_workers=0, pin_memory=True):
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_data_path = dataset_path
        self.train_label_path = label_path
        self.is_grey_scale = is_grey_scale
        self.hsv_mode = hsv_mode

        # # Data augmentation and normalization
        self.train_trans = transforms.Compose([transforms.Resize((self.input_size, self.input_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ])

        # use your own data loader
        self.TrainDataLoader = DataLoader(ImageImageLoader(self.train_trans,
                                                           self.train_data_path,
                                                           self.train_label_path,
                                                           self.is_grey_scale,
                                                           self.input_size,
                                                           self.hsv_mode),
                                          batch_size=self.batch_size,
                                          num_workers=self.num_workers,
                                          shuffle=True,
                                          pin_memory=self.pin_memory)

    def __len__(self):
        return self.TrainDataLoader.__len__()
