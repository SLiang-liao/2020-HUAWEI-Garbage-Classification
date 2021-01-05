'''
Author: SlytherinGe
LastEditTime: 2020-12-06 22:05:26
'''
# torch
import torch
import torch.utils.data as data
# utility
import numpy as np
import cv2
# system
import os.path as osp
import glob as glob
import random


class TrashDataset(data.Dataset):
    def __init__(self, data_root, label_files, num_classes, transform=None):

        self.img_and_label = []
        self.data_root = data_root
        self.transform = transform
        self.num_classes = num_classes
        print('reading annotations into memory...')
        for label_file in label_files:
            with open(label_file, 'r') as f:
                line = f.readline()
                line_split = line.split(',')
                if len(line_split) != 2:
                    print('%s contain error lable' % osp.basename(label_file))
                    continue
                img_name, label = line_split[0], int(line_split[1])
                self.img_and_label.append((img_name, label))
                f.close()
        print('done reading annotations! read {} annotations'.format(self.__len__()))

    def __len__(self):
        return (len(self.img_and_label))

    def __getitem__(self, index):
        img_path = osp.join(self.data_root, self.img_and_label[index][0])
        
        assert osp.exists(img_path), 'Image path does not exist: {}'.format(img_path)
        #print(img_path.replace('\\','/'))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #print(img)
        # to rgb
        img = img[:,:,(2,1,0)]

        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.tensor(np.zeros(self.num_classes), dtype=torch.long)
        label[self.img_and_label[index][1]] = 1

        return torch.from_numpy(img).permute(2,0,1), label

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))) 
        return fmt_str

def generate_train_and_val_dataset(data_root, num_classes, train_transform=None, val_transform=None, train_ratio=0.9, shuffle=True):

    label_files = glob.glob(osp.join(data_root, '*.txt'))
    if shuffle:
        random.shuffle(label_files)
    num_labels = len(label_files)
    num_train = int(train_ratio * num_labels)
    train_label_files = label_files[: num_train]
    val_label_files = label_files[num_train: ]

    train_dataset = TrashDataset(data_root, train_label_files, num_classes, train_transform)
    val_dataset = TrashDataset(data_root, val_label_files, num_classes, val_transform)
    
    return train_dataset, val_dataset

if __name__ == '__main__':
    train_dataset, val_dataset = generate_train_and_val_dataset(r"F:\比赛\华为学习赛\garbage_classify\train_data", 43)
    print(train_dataset)

