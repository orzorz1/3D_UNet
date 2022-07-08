from torch.nn import functional as F
import torch

def label_to_onehot(img):
    out = F.one_hot(img)
    out = out.cpu().numpy()
    out = out.transpose(3, 0, 1, 2)
    out = torch.tensor(out)

    return out
