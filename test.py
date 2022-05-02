from tkinter import E
import torch
from models.efficientnet.model import EfficientNet

net = EfficientNet.from_pretrained('efficientnet-b4')
size = (380, 380)
layers = net(torch.zeros(1,3,size[0],size[1]))
print(net)