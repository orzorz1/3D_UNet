import numpy
import numpy as np
from commons.plot import save_nii
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn as nn

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def concat_patches(pre):
    # 将patch依次拼接
    pre = []
    pre = np.array(pre)
    print(pre.shape)
    deep = []
    for i in range(64):  # （width/size[0]) * (height/size[1]）
        d = pre[4 * i]
        for j in range(1, 4):
            d = np.concatenate((d, pre[4 * i + j]), axis=2)
        deep.append(d.tolist())

    deep = np.array(deep)
    height = []
    for i in range(8):  # (height/size[1]）
        h = deep[8 * i]
        for j in range(1, 8):  # （height/size[1]）
            h = np.concatenate((h, deep[8 * i + j]), axis=1)
        height.append(h.tolist())

    height = np.array(height)
    predict = height[0]
    for i in range(1, 8):  # (width/size[0])
        predict = np.concatenate((predict, height[i]), axis=0)
    return predict


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
    triangle_ufunc1 = np.frompyfunc(set_label, 1, 1)
    out = triangle_ufunc1(img_arr)
    out = out.astype(np.float)
    return out

def get_patchs(img_x, img_y, size):
    patch_x = []
    patch_y = []
    box = get_bounding_box(img_y[0])
    width, height, deep = img_y.shape


def patch(img_arr, size):
    patchs = []
    # patch = []
    patch_size = size
    channel, width, height, deep = img_arr.shape
    img_arr = torch.cat((torch.tensor(np.zeros((1, 512, 512, 8)).astype(float)), torch.tensor(img_arr.astype(float))),
                        dim=3).numpy()
    for i in range(0, width, patch_size[0]):
        for j in range(0, height, patch_size[1]):
            for k in range(0, deep, patch_size[2]):
                # patch.append(img_arr[i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]])
                # patch.append([i,j,k])
                # patchs.append(patch) #patchs[index][0]为patch，patchs[index][1]为patch在原始图像中的位置
                patch = img_arr[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]].tolist()
                patchs.append(patch)
    patchs = np.array(patchs)
    return patchs #[index, channel, width, height, deep]


class load_dataset(Dataset):
    def __init__(self, index):
        path_x = "../dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz".format(index=index)
        x = read_dataset(path_x)
        path_y = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
        y = read_label(path_y)
        y = reshape(y)
        patch_size = [128, 128, 32]
        patch_x = patch(x, patch_size)
        patch_y = patch(y, patch_size)
        print("数据加载完成，shape：", patch_x.shape)
        imgs = []
        for i in range(patch_x.shape[0]):
            imgs.append((patch_x[i], patch_y[i]))
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = np.array(img).astype(float)
        target = np.array(label)
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long()

    def __len__(self):
        return len(self.imgs)


path = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=1)
img = read_label(path)
print(get_bounding_box(img))
