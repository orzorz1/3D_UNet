import numpy
import numpy as np
from commons.plot import save_nii
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random


def get_bounding_box(img):
    width, height, deep = img.shape
    box = [0, 0, 0, 0, 0, 0]  # [width_low, width_high, height_low, height_high, deep_low, deep_high]
    flag = 0
    for i in range(width):
        img_x = img[i, :, :]
        a = numpy.ones(img_x.shape)
        if flag == 0 and (img_x * a).sum() != 0:
            box[0] = i
            flag = 1
        if flag == 1 and (img_x * a).sum() == 0:
            box[1] = i
            flag = 0
    for i in range(height):
        img_y = img[:, i, :]
        a = numpy.ones(img_y.shape)
        if flag == 0 and (img_y * a).sum() != 0:
            box[2] = i
            flag = 1
        if flag == 1 and (img_y * a).sum() == 0:
            box[3] = i
            flag = 0
    for i in range(deep):
        img_z = img[:, :, i]
        a = numpy.ones(img_z.shape)
        if flag == 0 and (img_z * a).sum() != 0:
            box[4] = i
            flag = 1
        if flag == 1 and (img_z * a).sum() == 0:
            box[5] = i
            flag = 0
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

def get_patchs_from_one_img(img_x, img_y, size, number):
    # random.seed(123)
    patchs_x = []
    patchs_y = []
    box = get_bounding_box(img_y)
    width, height, deep = img_y.shape
    verte_range = [max(box[0]-size[0], 0), min(box[1], width-size[0]),
                   max(box[2]-size[1], 0), min(box[3], height-size[1]),
                   max(box[4]-size[2], 0), min(box[5], deep-size[2])]
    for i in range(number):
        verte = [random.randint(verte_range[0],verte_range[1]), random.randint(verte_range[2],verte_range[3]), random.randint(verte_range[4],verte_range[5])]
        patch_x = []
        patch_x.append(np.expand_dims(img_x[verte[0]:verte[0]+size[0],verte[1]:verte[1]+size[1],verte[2]:verte[2]+size[2]], axis=0))
        patch_x.append(np.array(verte))
        patchs_x.append(patch_x)
        patch_y = []
        patch_y.append(img_y[verte[0]:verte[0]+size[0],verte[1]:verte[1]+size[1],verte[2]:verte[2]+size[2]].tolist())
        patchs_y.append(patch_y)
    patchs_x = np.array(patchs_x)
    # patchs_x[index][0]为图像，patchs_x[index][1]为patch顶点在原始图像中的位置
    patchs_y = np.array(patchs_y) # (channel, deep, width, height)
    return patchs_x, patchs_y

def get_patches(begin, end, size, seed):
    random.seed(seed)
    patches_x = []
    patches_y = []
    l =  [x for x in range(begin, end+1)]
    random.shuffle(l)
    for i in l:
        path_x = "../dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz".format(index=i)
        x = read_dataset(path_x)
        path_y = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=i)
        y = read_label(path_y)
        x, y = get_patchs_from_one_img(x, y, size, 10)
        for j in range(10):
            patches_x.append(x[j])
            patches_y.append(y[j])

    patches_x = np.array(patches_x)
    patches_y = np.array(patches_y)
    return patches_x, patches_y

def patch(img_arr, size):
    patchs = []
    patch_size = size
    img_arr = torch.cat((torch.tensor(np.zeros((1, 512, 512, 8)).astype(float)), torch.tensor(img_arr.astype(float))),
                        dim=3).numpy()
    channel, width, height, deep = img_arr.shape
    for i in range(0, width-patch_size[0]+1, patch_size[0]//2):
        for j in range(0, height-patch_size[1]+1, patch_size[1]//2):
            for k in range(0, deep-patch_size[2]+1, patch_size[2]//2):
                patch = []
                patch.append(img_arr[:,i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]])
                patch.append([i, j, k])
                patchs.append(patch) #patchs[index][0]为patch，patchs[index][1]为patch在原始图像中的位置
    return patchs


class load_dataset_one(Dataset):
    def __init__(self, index):
        path_x = "../dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz".format(index=index)
        x = read_dataset(path_x)
        path_y = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
        y = read_label(path_y)
        patch_size = [64, 64, 32]
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
    def __init__(self,begin, end, seed):
        patch_size = [64, 64, 32]
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

class load_dataset_test(Dataset):
    def __init__(self, index):
        path_x = "../dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz".format(index=index)
        x = read_dataset(path_x)
        x = reshape(x)
        path_y = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
        y = read_label(path_y)
        y = reshape(y)
        patch_size = [64, 64, 32]
        patchs_x = patch(x, patch_size)
        patchs_y = patch(y, patch_size)
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
