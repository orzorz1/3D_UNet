import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from data.LoadData import *
from torch.utils.data import DataLoader
import torch
from models.UNet_3D import UNet_3D
from modules.functions import dice_loss
import math
from commons.plot import save_nii
import torchsummary
from commons.log import make_print_to_file
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def predct():
    size = [64, 64, 32]
    batch_size = 1
    model = UNet_3D()
    try:
        model.load_state_dict(torch.load("UNet_3D-shuffle-27.pth", map_location='cpu'))
    except FileNotFoundError:
        print("模型不存在")
    else:
        print("加载模型成功")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for index in range(101,106):
        val_data = load_dataset_test(index)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
        model.eval()
        # path_x = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
        # x = read_dataset(path_x)
        x = np.zeros((512, 512, 128))
        predict = np.zeros_like(x)
        count = np.zeros_like(x)
        for batch, (batch_x, batch_y, position) in enumerate(val_loader):
            for i in range(len(position)):
                position[i] = position[i].cpu().numpy().tolist()[0]
            print(position)
            batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
            out = model(batch_x)
            # out = np.around(nn.Softmax(dim=1)(out).cpu().detach().numpy()[0])
            # out = out[1]
            # out = torch.max(nn.Softmax(dim=1)(out), 1)[1].cpu().detach().numpy()[0]
            out = out.cpu().detach().numpu()[0]
            predict[0:3, position[0]:position[0] + size[0], position[1]:position[1] + size[1], position[2]:position[2] + size[2]] += out
            count[0:3, position[0]:position[0] + size[0], position[1]:position[1] + size[1],position[2]:position[2] + size[2]] += np.ones_like(out)
            # out = nn.Sigmoid()(out)
            # if l != 0:
            #     # print("√", l, end="  ")
            #         save_nii(batch_x.cpu().numpy().astype(np.int16)[n][0],
            #                  '{i}-X-{l}'.format(i=i, l=loss))
            #         save_nii(batch_y.cpu().numpy().astype(np.int16)[n][0],
            #                  '{i}-Y-{l}'.format(i=i, l=loss))
            #         save_nii(out.cpu().detach().numpy().astype(np.int16)[n][0],
            #                  '{i}-Out-{l}'.format(i=i, l=loss))

            # pre.append(out.cpu().detach().numpy().astype(np.int64)[0][0].tolist())


        pre= predict / count

        pre = pre[:,:,8:129]
        print(pre.shape)
        pre = torch.tensor(pre)
        pre = torch.max(nn.Softmax(dim=1)(pre), 1)[1].cpu().detach().numpy()

        save_nii(pre.astype(np.int16), "pre-{index}".format(index = index), index)






if __name__ == '__main__':
    # make_print_to_file("./")
    torch.cuda.empty_cache()
    predct()
    os.system("shutdown")