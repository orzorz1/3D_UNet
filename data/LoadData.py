import numpy
import numpy as np
from commons.plot import save_nii
import nibabel as nib
import torch
from torch.utils.data import Dataset
from commons.plot import *
import torch.nn as nn
import random
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def get_bounding_box(img):
    width, height, deep = img.shape
    box = [0, 0, 0, 0, 0, 0]  # [width_low, width_high, height_low, height_high, deep_low, deep_high]
    for i in range(width):
        img_x = img[i, :, :]
        a = numpy.ones(img_x.shape)
        if (img_x * a).sum() != 0:
            box[0] = i
            break
    for i in range(width):
        img_x = img[width-i-1, :, :]
        a = numpy.ones(img_x.shape)
        if (img_x * a).sum() != 0:
            box[1] = width-i-1
            break
    for i in range(height):
        img_y = img[:, i, :]
        a = numpy.ones(img_y.shape)
        if (img_y * a).sum() != 0:
            box[2] = i
            break
    for i in range(height):
        img_y = img[:, height-i-1, :]
        a = numpy.ones(img_y.shape)
        if (img_y * a).sum() != 0:
            box[3] = height-i-1
            break
    for i in range(deep):
        img_z = img[:, :, i]
        a = numpy.ones(img_z.shape)
        if (img_z * a).sum() != 0:
            box[4] = i
            break
    for i in range(deep):
        img_z = img[:, :, deep-i-1]
        a = numpy.ones(img_z.shape)
        if (img_z * a).sum() != 0:
            box[5] = deep-i-1
            break
    return box


def setHU(img_arr, min, max):
    img_arr = np.clip(img_arr, min, max)
    img_arr = img_arr.astype(np.float32)
    return img_arr


def set_label(x):
    if x != 1:
        x = 0
    return x

# (width, height, deep) -> (channel, deep, width, height)
def reshape(pic):
    pic = np.expand_dims(pic, axis=0)
    return pic


def read_dataset(path):
    img = nib.load(path)
    img_arr = np.array(img.dataobj)
    img_arr = setHU(img_arr, 0, 1200)
    return img_arr

def read_label(path):
    img = nib.load(path)
    img_arr = np.array(img.dataobj)
    # triangle_ufunc1 = np.frompyfunc(set_label, 1, 1)
    # out = triangle_ufunc1(img_arr)
    # out = out.astype(np.float)
    out = img_arr
    return out

# 从一个图像中随机取包含肿瘤的patch
def get_patchs_from_one_img(img_x, img_y, patch_size, number):
    # random.seed(123)
    patchs_x = []
    patchs_y = []
    box = get_bounding_box(img_y)
    # triangle_ufunc1 = np.frompyfunc(set_label, 1, 1)
    # out = triangle_ufunc1(img_y)
    # out = out.astype(np.float)
    # box = get_bounding_box(out)
    width, height, deep = img_y.shape
    verte_range = [max(box[0]-patch_size[0], 0), min(box[1], width-patch_size[0]),
                   max(box[2]-patch_size[1], 0), min(box[3], height-patch_size[1]),
                   max(box[4]-patch_size[2], 0), min(box[5], deep-patch_size[2])]
    for i in range(number):
        verte = [random.randint(verte_range[0],verte_range[1]), random.randint(verte_range[2],verte_range[3]), random.randint(verte_range[4],verte_range[5])]
        patch_x = []
        patch_x.append(np.expand_dims(img_x[verte[0]:verte[0]+patch_size[0],verte[1]:verte[1]+patch_size[1],verte[2]:verte[2]+patch_size[2]], axis=0))
        patch_x.append(np.array(verte))
        patchs_x.append(patch_x)
        patch_y = []
        patch_y.append(img_y[verte[0]:verte[0]+patch_size[0],verte[1]:verte[1]+patch_size[1],verte[2]:verte[2]+patch_size[2]].tolist())
        patchs_y.append(patch_y)
    patchs_x = np.array(patchs_x)
    # patchs_x[index][0]为图像，patchs_x[index][1]为patch顶点在原始图像中的位置
    patchs_y = np.array(patchs_y) # (channel, deep, width, height)
    return patchs_x, patchs_y

# 将从多张图片取的随机patch组合在一起
def get_patches(begin, end, patch_size, seed):
    random.seed(seed)
    patches_x = []
    patches_y = []
    l =  [x for x in range(begin, end+1)]
    random.shuffle(l)
    for i in l:
        path_x = "./dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz".format(index=i)
        x = read_dataset(path_x)
        path_y = "./dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=i)
        y = read_label(path_y)
        x1, y1 = get_patchs_from_one_img(x, y, patch_size, 10)
        for j in range(10):
            patches_x.append(x1[j])
            patches_y.append(y1[j])
        # x2 = np.array(mean_patch(reshape(x), patch_size, 1))
        # y2 = np.array(mean_patch(reshape(y), patch_size, 1))
        # permutation = np.random.permutation(x2.shape[0])
        # x2 = x2[permutation]
        # y2 = y2[permutation]
        # for j in range(10):
        #     x2[j][0] = x2[j][0].astype(np.float32)
        #     patches_x.append(x2[j])
        #     patches_y.append(y2[j][0])
    patches_x = np.array(patches_x)
    patches_y = np.array(patches_y)
    return patches_x, patches_y


class load_dataset_one(Dataset):
    def __init__(self, index, patch_size):
        path_x = "./dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz".format(index=index)
        x = read_dataset(path_x)
        path_y = "./dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
        y = read_label(path_y)
        patchs_x, patchs_y = get_patchs_from_one_img(x, y, patch_size, 30)
        print("数据加载完成，shape：", patchs_y.shape)
        imgs = []
        for i in range(patchs_x.shape[0]):
            imgs.append((patchs_x[i], patchs_y[i]))
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        position = img[1]
        img = np.array(img[0]).astype(float)
        target = np.array(label)
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long(), position

    def __len__(self):
        return len(self.imgs)

class load_dataset(Dataset):
    def __init__(self, begin, end, seed, patch_size):
        """

        :rtype: object
        """
        patchs_x, patchs_y = get_patches(begin, end, patch_size, seed)
        print("数据加载完成，shape：", patchs_y.shape)
        imgs = []
        for i in range(patchs_x.shape[0]):
            imgs.append((patchs_x[i], patchs_y[i]))
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        position = img[1]
        img = np.array(img[0]).astype(float)
        target = np.array(label)
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long(), position

    def __len__(self):
        return len(self.imgs)

# 均匀取patch
def mean_patch(img_arr, size, overlap_factor):
    patchs = []
    patch_size = size
    img_arr = torch.cat((torch.tensor(np.zeros((1, 512, 512, 8)).astype(float)), torch.tensor(img_arr.astype(float))),
                        dim=3).numpy()
    channel, width, height, deep = img_arr.shape
    for i in range(0, width-patch_size[0]+1, patch_size[0]//overlap_factor):
        for j in range(0, height-patch_size[1]+1, patch_size[1]//overlap_factor):
            for k in range(0, deep-patch_size[2]+1, patch_size[2]//overlap_factor):
                patch = []
                patch.append(img_arr[:,i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]])
                patch.append([i, j, k])
                patchs.append(patch) #patchs[index][0]为patch，patchs[index][1]为patch在原始图像中的位置
    return np.array(patchs)

class load_dataset_test(Dataset):
    def __init__(self, index, patch_size):
        path_x = "./dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz".format(index=index)
        x = read_dataset(path_x)
        x = reshape(x)
        path_y = "./dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
        y = read_label(path_y)
        y = reshape(y)
        patchs_x = mean_patch(x, patch_size, 4)
        patchs_y = mean_patch(y, patch_size, 4)
        print("数据加载完成")
        imgs = []
        for i in range(len(patchs_x)):
            imgs.append((patchs_x[i], patchs_y[i]))
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        position = img[1]
        img = np.array(img[0]).astype(float)
        target = np.array(label[0])
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long(), position

    def __len__(self):
        return len(self.imgs)


# path = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=1)
# print(get_bounding_box(img))

# path = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=1)
# img = read_label(path)
# img = reshape(img)
# img1 = patch(img,[128,128,32])