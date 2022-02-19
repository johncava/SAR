import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
import torchvision
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

sat_dataset = SARToImageDataset(root_dir='/scratch/jcava/SAR/QXSLAB_SAROPT',sar_dir='sar_256_oc_0.2',eo_dir='opt_256_oc_0.2')

batch_size = 64
dataset_loader = torch.utils.data.DataLoader(sat_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)

###
# U-Net 
# Reference: https://amaarora.github.io/2020/09/13/unet.html
###
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=3, retain_dim=False, out_sz=(256,256)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        out = torch.sigmoid(out)
        return out

model = UNet(retain_dim=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epoch_loss = []
max_epochs = 10
for epoch in range(max_epochs):
    losses = []
    for i, (x,y) in enumerate(dataset_loader):
        x = x.float()
        y = y.float()
        pred = model(x) * 255
        # print(pred.size())
        optimizer.zero_grad()
        loss = F.mse_loss(pred, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    epoch_loss.append(np.mean(losses))

##
# Plot Loss
##
plt.plot(list(range(len(epoch_loss))), epoch_loss)
plt.title('MSE Loss vs Epochs')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.savefig('loss.png')

