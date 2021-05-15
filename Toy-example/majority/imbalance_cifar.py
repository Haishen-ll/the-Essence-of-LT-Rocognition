import os
import random

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True, transform=None, target_transform=None, rand_transform = None,
                 download=False, img_num_list=None):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.rand_transform = rand_transform
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)

        self.gen_imbalanced_data(img_num_list)
        self.train_length = self.get_train_len()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        print('img_num_per_cls:',img_num_per_cls)
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)

        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets


    def get_train_len(self):
        
        length = 0
        for i in range(self.cls_num):
            length = length + self.num_per_cls_dict[i]
        return length



    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    # def get_val_data(self):
    #     val_img = []
    #
    #     if self.transform is not None:
    #         for img in self.val_set:
    #             img = Image.fromarray(img)
    #             img = self.transform(img)
    #             val_img.append(img.unsqueeze(0))
    #         val_img = torch.cat(val_img, dim=0)
    #         val_target = torch.tensor(self.val_targets, dtype=torch.int64)
    #         return val_img, val_target
    #
    #     return False

    def gen_val_data(self, num_per_cls=10):
        val_data = []
        val_target = []
        targets_np = np.array(self.targets, dtype=np.int64)

        for i in range(self.cls_num):
            idx = np.where(targets_np == i)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:num_per_cls]
            val_data.append(self.data[selec_idx, ...])
            val_target.extend([i, ] * num_per_cls)
        val_data = np.vstack(val_data)

        val_img = []
        if self.transform is not None:
            for img in val_data:
                img = Image.fromarray(img)
                img = self.transform(img)
                val_img.append(img.unsqueeze(0))
            val_img = torch.cat(val_img, dim=0)
            val_target = torch.tensor(val_target, dtype=torch.int64)
            return val_img, val_target

    def getshape(self):
        return self.data.shape






class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


# Dataset for ImageNet_LT and Places_LT
class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.txt = os.path.join(root, txt)
        with open(self.txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        if 'Image' in root:
            self.cls_num = 1000
        else:
            self.cls_num = 365
        self.num_per_cls_dict = self.get_num_per_cls(self.labels)

    def get_num_per_cls(self, labels):
        targets_np = np.array(labels, dtype=np.int64)
        classes = np.unique(targets_np)
        num_per_cls_dict = dict()
        for clx in classes:
            idx = np.where(targets_np == clx)[0]
            class_num = idx.shape[0]
            num_per_cls_dict[clx] = class_num
        return num_per_cls_dict

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
        
    def gen_val_data(self, num_per_cls=10):
        val_path = []
        val_labels = []
        targets_np = np.array(self.labels, dtype=np.int64)

        for i in range(self.cls_num):
            idx = np.where(targets_np == i)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:num_per_cls]
            for j in selec_idx:
                val_path.append(self.img_path[j])
            val_labels.extend([i, ] * num_per_cls)

        val_img = []
        if self.transform is not None:
            for path in val_path:
                with open(path, 'rb') as f:
                    sample = Image.open(f).convert('RGB')
                sample = self.transform(sample)
                val_img.append(sample.unsqueeze(0))
            val_img = torch.cat(val_img, dim=0)
            val_labels = torch.tensor(val_labels, dtype=torch.int64)
            return val_img, val_labels
    def getshape(self):
        return self.data.shape

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR10(root='/home/public_data/liulei/cifar10/', train=True,
                    download=True, transform=transform)
    print(trainset.getshape())
    # trainloader = iter(trainset)
    # data, label = next(trainloader)
    # import pdb; pdb.set_trace()