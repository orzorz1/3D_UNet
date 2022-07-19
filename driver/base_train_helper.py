import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from models.UNet_3D import UNet_3D
from modules.functions import dice_loss, ce_loss
from commons.plot import save_nii, draw, draw1
from data.LoadData import *
from torch.utils.data import DataLoader
import torch
import math
from commons.plot import save_nii
import torchsummary
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)



class BaseTrainHelper(object):
    def __init__(self, model):
        self.model = model

    def train(self):
        loss_train = []
        loss_val = []
        batch_size = 15
        epochs = 50
        model = self.model
        try:
            model.load_state_dict(torch.load("UNet_3D-shuffle-20.pth", map_location='cpu'))
        except FileNotFoundError:
            print("模型不存在")
        else:
            print("加载模型成功")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        torchsummary.summary(model, (1,64,64,32), batch_size=batch_size, device="cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 40], 0.1)
        for i in range(21,31):
            print("训练进度：{index}/30".format(index=i))
            dataset = load_dataset(1, 90, i)
            val_data = load_dataset_one(101)
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
                    if batch == 2 and (epoch == 49 or epoch == 29 or epoch == 9):
                        save_nii(batch_x.cpu().numpy().astype(np.int16)[0][0],'{name}-{e}X'.format(name=i, e=epoch+1))
                        save_nii(batch_y.cpu().numpy().astype(np.int16)[0][0],'{name}-{e}Y'.format(name=i, e=epoch+1))
                        out = np.around(nn.Softmax(dim=1)(out).cpu().detach().numpy()[0])
                        save_nii(out[0], '{name}-{e}Out0'.format(name=i, e=epoch+1))
                        save_nii(out[1], '{name}-{e}Out1'.format(name=i, e=epoch+1))
                        save_nii(out[2], '{name}-{e}Out2'.format(name=i, e=epoch+1))

                print('Val Loss: %.6f' % (eval_loss / (math.ceil(len(dataset) / batch_size))))
                loss_val.append((eval_loss / (math.ceil(len(dataset) / batch_size))))
            torch.save(model.state_dict(), "UNet_3D-shuffle-{i}.pth".format(i=i))
            draw1(loss_train, "{i}-train".format(i=i))
            draw1(loss_val, "{i}-val".format(i=i))
            print(loss_train)
            print(loss_val)

    def predct(self):
        size = [64, 64, 32]
        batch_size = 1
        model = self.model
        try:
            model.load_state_dict(torch.load("UNet_3D-shuffle-27.pth", map_location='cpu'))
        except FileNotFoundError:
            print("模型不存在")
        else:
            print("加载模型成功")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for index in range(102, 106):
            val_data = load_dataset_test(index)
            val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
            model.eval()
            # path_x = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=index)
            # x = read_dataset(path_x)
            x = np.zeros((3, 512, 512, 128))
            predict = np.zeros_like(x)
            count = np.zeros_like(x)
            for batch, (batch_x, batch_y, position) in enumerate(val_loader):
                for i in range(len(position)):
                    position[i] = position[i].cpu().numpy().tolist()[0]
                print(position)
                batch_x, batch_y = torch.autograd.Variable(batch_x.to(device)), torch.autograd.Variable(
                    batch_y.to(device))
                out = model(batch_x)
                # out = np.around(nn.Softmax(dim=1)(out).cpu().detach().numpy()[0])
                # out = out[1]
                # out = torch.max(nn.Softmax(dim=1)(out), 1)[1].cpu().detach().numpy()[0]
                out = out.cpu().detach().numpy()[0]
                predict[0:3, position[0]:position[0] + size[0], position[1]:position[1] + size[1],
                position[2]:position[2] + size[2]] += out
                count[0:3, position[0]:position[0] + size[0], position[1]:position[1] + size[1],
                position[2]:position[2] + size[2]] += np.ones_like(out)
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

            pre = predict / count

            pre = pre[:, :, :, 8:128]
            print(pre.shape)
            pre = torch.tensor(pre)
            pre = torch.max(nn.Softmax(dim=0)(pre), 0)[1].cpu().detach().numpy()
            # pre=pre.transpose(1,2,3,0)

            save_nii(pre.astype(np.int16), "pre-{index}".format(index=index), index)

if __name__ == '__main__':
    # make_print_to_file("./")
    torch.cuda.empty_cache()
    NetWork = BaseTrainHelper(UNet_3D)
    NetWork.train()
    # NetWork.predct()
    # os.system("shutdown")