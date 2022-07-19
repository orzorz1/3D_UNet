import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import nibabel as nib
import torch
import surface_distance as surfdist


def set_label(x):
    if x != 1:
        x = 0
    else:
        x = 1
    return x

def read_nii(path):
    img = nib.load(path)
    img_arr = np.array(img.dataobj)
    triangle_ufunc1 = np.frompyfunc(set_label, 1, 1)
    out = triangle_ufunc1(img_arr)
    out = out.astype(np.float)
    # out = img_arr
    return out

def dice_score(pred, target):
    m1 = pred
    m2 = target
    intersection = (m1 * m2).sum()

    return (2.0 * intersection) / (m1.sum() + m2.sum())

for index in range(101,106):
    path_x = "../driver/pre-{index}.nii".format(index=index)
    x = read_nii(path_x)
    # x = torch.tensor(x)
    path_y = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
    y = read_nii(path_y)
    # y = torch.tensor(y)
    # print(dice_score(x, y))

    surface_distances = surfdist.compute_surface_distances(
        y, x, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    print(avg_surf_dist)