import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch


def dice_coeff(pred, target):
    smooth = 1.0
    num = pred.size(0)
    m1 = pred.view(num, -1) #Flatten
    m2 = target.view(num, -1) #Flatten
    intersection = (m1 * m2).sum()

    return (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def dice_loss(pred, target):
    smooth = 1.0
    num = pred.size(0)
    m1 = pred.view(num, -1).float() #Flatten
    m2 = target.view(num, -1).float() #Flatten
    intersection = (m1 * m2).sum()
    dice_coeff = abs(2.0 * intersection + smooth) / (abs(m1.sum()) + abs(m2.sum()) + smooth)
    l = 0
    n = 0
    for i in range(num):
        if m2[i].sum() != 0:
            l = m2[i].sum()
            n = i
    return 1-dice_coeff, l, n


# def dice_loss(target, predictive, ep=1e-8):
#     intersection = 2 * torch.sum(predictive * target) + ep
#     union = torch.sum(predictive) + torch.sum(target) + ep
#     loss = 1 - intersection / union
#     return loss

def bce_loss(pred, target):

    target = target.float()
    loss = torch.nn.BCEWithLogitsLoss()
    out = loss(pred, target)
    return out

def ce_loss(pred, target):

    target = target.float()
    target = target.squeeze(1)
    target = target.long()
    loss = torch.nn.CrossEntropyLoss()
    out = loss(pred, target)
    return out