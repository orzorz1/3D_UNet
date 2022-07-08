import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from data.LoadData import *
from torch.utils.data import DataLoader
import torch
from models.RA_UNet import RA_UNet_2
from modules.functions import dice_loss
import math
from commons.plot import save_nii
import random
import torchsummary
from commons.log import make_print_to_file

print(torch.cuda.is_available())
def train():
    batch_size = 8
    epochs = 20
    model = RA_UNet_2()
    try:
        model.load_state_dict(torch.load("UNet_3D.pth", map_location='cpu'))
    except FileNotFoundError:
        print("模型不存在")
    else:
        print("加载模型成功")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torchsummary.summary(model, (1,128,128,32),batch_size=batch_size, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for i in range(1,16):
        print("训练进度：{index}/15".format(index=i))
        dataset = load_dataset(i)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        for epoch in range(epochs):
            # training-----------------------------------
            model.train()
            train_loss = 0
            train_acc = 0
            for batch, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
                out = model(batch_x)
                loss = dice_loss(out, batch_y)
                # if loss <0.999:
                #     name = str(random.randint(1,9999))
                #     save_nii(batch_x.cpu().numpy().astype(np.int16)[0][0],'{name}X'.format(name=name))
                #     save_nii(batch_y.cpu().numpy().astype(np.int16)[0][0],'{name}Y'.format(name=name))
                #     save_nii(out.cpu().detach().numpy().astype(np.int16)[0][0], '{name}Out'.format(name=name))
                train_loss += loss.item()
                print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f'
                      % (epoch + 1, epochs, batch + 1, math.ceil(len(dataset) / batch_size),loss.item(),))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Train Loss: %.6f' % (train_loss / (math.ceil(len(dataset) / batch_size))))

            #evaluation---------------------
            model.eval()
            eval_loss = 0
            eval_acc = 0
            for batch, (batch_x, batch_y) in enumerate(val_loader):
                batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
                out = model(batch_x)
                # printXandY(out.detach().cpu().numpy()[0,0,:,:]*255, batch_y.detach().cpu().numpy()[0,0,:,:]*255)
                loss = dice_loss(out, batch_y)
                eval_loss += loss.item()
            print('Val Loss: %.6f' % (eval_loss / (math.ceil(len(dataset) / batch_size))))
        torch.save(model.state_dict(), "UNet_3D.pth")


if __name__ == '__main__':
    make_print_to_file("./")
    torch.cuda.empty_cache()
    train()
    os.system("shutdown")