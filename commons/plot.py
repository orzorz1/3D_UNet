from matplotlib import pylab as plt
import nibabel as nib
import numpy as np

def print2D(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def printXandY(out, label):
    plt.subplot(1, 2, 1)
    plt.imshow(out, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='gray')
    plt.show()

def save_nii(img_arr, name, index=1):
    img = nib.load('../dataset/crossmoda2021_ldn_{index}_ceT1.nii.gz'.format(index=index))
    img_affine = img.affine
    new_img = nib.Nifti1Image(img_arr, img_affine)
    nib.save(new_img, "{name}.nii".format(name=name))
    # nib.Nifti1Image(img_arr, np.eye(4)).to_filename(f'{name}.nii.gz'.format(name=name))
    # img_arr要为int16的nparray

def draw(loss, loss2, name):
    x = [range(0, len(loss))]
    x = [[row[i] for row in x] for i in range(len(x[0]))]
    fig, ax = plt.subplots()
    ax.plot(x, loss, color="red", label="loss_train")
    ax.plot(x, loss2, color="blue", label="loss_val")
    ax.set_title(name)
    ax.legend()
    plt.show()

def draw1(loss,  name):
    x = [range(0, len(loss))]
    x = [[row[i] for row in x] for i in range(len(x[0]))]
    fig, ax = plt.subplots()
    ax.plot(x, loss, color="red", label="loss_train")
    ax.set_title(name)
    ax.legend()
    plt.show()