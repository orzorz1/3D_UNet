import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from data.LoadData import *
from torch.utils.data import DataLoader
import torch
from models.UNet_3D import UNet_3D
from modules.functions import dice_loss, ce_loss
import math
from commons.plot import save_nii, draw, draw1
import torchsummary
from commons.log import make_print_to_file
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def train():
    loss_train = []
    loss_val = []
    batch_size = 12
    epochs = 50
    model = UNet_3D()
    try:
        model.load_state_dict(torch.load("UNet_3D-shuffle-14.pth", map_location='cpu'))
    except FileNotFoundError:
        print("模型不存在")
    else:
        print("加载模型成功")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torchsummary.summary(model, (1,64,64,32), batch_size=batch_size, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 40], 0.1)
    for i in range(15,30):
        print("训练进度：{index}/30".format(index=i))
        dataset = load_dataset(17, 30, i)
        val_data = load_dataset_one(16)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
        for epoch in range(epochs):
            # training-----------------------------------
            model.train()
            train_loss = 0
            for batch, (batch_x, batch_y, position) in enumerate(train_loader):
                batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
                out = model(batch_x)
                # loss, l, n = dice_loss(out, batch_y)
                # if l != 0:
                #     print("√",l,end="  ")
                loss = ce_loss(out, batch_y)
                train_loss += loss.item()
                print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f'
                      % (epoch + 1, epochs, batch + 1, math.ceil(len(dataset) / batch_size),loss.item(),))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()  # 更新learning rate
            print('Train Loss: %.6f' % (train_loss / (math.ceil(len(dataset) / batch_size))))
            loss_train.append(train_loss / (math.ceil(len(dataset) / batch_size)))

            #evaluation---------------------
            model.eval()
            eval_loss = 0
            for batch, (batch_x, batch_y, position) in enumerate(val_loader):
                batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(batch_y.to(device))
                out = model(batch_x)
                # loss, l, n = dice_loss(out, batch_y)
                loss = ce_loss(out, batch_y)
                eval_loss += loss.item()
                if batch == 8 and (epoch == 49 or epoch == 29 or epoch == 9):
                    save_nii(batch_x.cpu().numpy().astype(np.int16)[0][0],'{name}-{e}X'.format(name=i, e=epoch+1))
                    save_nii(batch_y.cpu().numpy().astype(np.int16)[0][0],'{name}-{e}Y'.format(name=i, e=epoch+1))
                    save_nii(out.cpu().detach().numpy().astype(np.int16)[0][0], '{name}-{e}Out'.format(name=i, e=epoch+1))
            print('Val Loss: %.6f' % (eval_loss / (math.ceil(len(dataset) / batch_size))))
            loss_val.append((eval_loss / (math.ceil(len(dataset) / batch_size))))
        torch.save(model.state_dict(), "UNet_3D-shuffle-{i}.pth".format(i=i))
        draw1(loss_train, "{i}-train".format(i=i))
        draw1(loss_val, "{i}-val".format(i=i))
        print(loss_train)
        print(loss_val)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    make_print_to_file("./")
    torch.cuda.empty_cache()
    train()
    os.system("shutdown")