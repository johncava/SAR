import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
import numpy as np

from dataset import *

##
# ResNetUNet Reference: https://colab.research.google.com/github/usuyama/pytorch-unet/blob/master/pytorch_unet_resnet18_colab.ipynb#scrollTo=b8EJl0hcC5DH
##
def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )


class ResNetUNet(nn.Module):
  def __init__(self, n_class):
    super().__init__()

    self.base_model = torchvision.models.resnet18(pretrained=True)
    self.base_layers = list(self.base_model.children())

    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0_1x1 = convrelu(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convrelu(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convrelu(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convrelu(256, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convrelu(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

    self.conv_original_size = convrelu(1, 3, 3, 1)
    self.conv_original_size0 = convrelu(1, 64, 3, 1)
    self.conv_original_size1 = convrelu(64, 64, 3, 1)
    self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

    self.conv_last = nn.Conv2d(64, n_class, 1)

  def forward(self, input):
    x_original = self.conv_original_size0(input)
    x_original = self.conv_original_size1(x_original)

    x_input = self.conv_original_size(input)
    layer0 = self.layer0(x_input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)

    layer4 = self.layer4_1x1(layer4)
    x = self.upsample(layer4)
    layer3 = self.layer3_1x1(layer3)
    x = torch.cat([x, layer3], dim=1)
    x = self.conv_up3(x)

    x = self.upsample(x)
    layer2 = self.layer2_1x1(layer2)
    x = torch.cat([x, layer2], dim=1)
    x = self.conv_up2(x)

    x = self.upsample(x)
    layer1 = self.layer1_1x1(layer1)
    x = torch.cat([x, layer1], dim=1)
    x = self.conv_up1(x)

    x = self.upsample(x)
    layer0 = self.layer0_1x1(layer0)
    x = torch.cat([x, layer0], dim=1)
    x = self.conv_up0(x)

    x = self.upsample(x)
    x = torch.cat([x, x_original], dim=1)
    x = self.conv_original_size2(x)

    out = self.conv_last(x)
    out = torch.sigmoid(out)
    return out

sat_dataset = SARToImageDataset(root_dir='/scratch/jcava/SAR/QXSLAB_SAROPT',sar_dir='sar_256_oc_0.2',eo_dir='opt_256_oc_0.2')

batch_size = 32
dataset_loader = torch.utils.data.DataLoader(sat_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)


model = ResNetUNet(3).cuda()
epoch_loss = []
learning_rates = [1e-3,1e-3,1e-4,1e-4,1e-4,1e-4]
max_epochs = len(learning_rates)
import time
for epoch in range(max_epochs):
    start = time.time()
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rates[epoch], weight_decay=5e-4)
    for i, (x,y) in enumerate(dataset_loader):
        x = x.float().cuda()
        y = y.float().cuda() / 255
        pred = model(x) # * 255
        # print(pred.size())
        optimizer.zero_grad()
        loss = F.mse_loss(pred, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss.append(np.mean(losses))
    end = time.time()
    print('Epoch ' + str(epoch) + ': ' + str(end-start) + 's')

##
# Plot Loss
##
plt.plot(list(range(len(epoch_loss))), epoch_loss)
plt.title('MSE Loss vs Epochs')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.savefig('loss_v3.png')

print('Done')