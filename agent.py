import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, min_length=10):
        super(Model, self).__init__()
        plane1 = 16
        plane2 = 32
        plane3 = 64
        self.encoder = nn.Sequential(
                nn.Conv2d(3, plane1, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(plane1), # nn.LayerNorm(plane1*32*32)
                nn.ReLU(inplace=True),
                nn.Conv2d(plane1, plane1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(plane1), # nn.LayerNorm(plane1*32*32)
                nn.ReLU(inplace=True),

                nn.Conv2d(plane1, plane2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(plane2), # nn.LayerNorm(plane2*16*16)
                nn.ReLU(inplace=True),
                nn.Conv2d(plane2, plane2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(plane2), # nn.LayerNorm(plane2*16*16)
                nn.ReLU(inplace=True),

                nn.Conv2d(plane3, plane3, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(plane3), # nn.LayerNorm(plane3*8*8)
                nn.ReLU(inplace=True),
                nn.Conv2d(plane3, plane3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(plane3), # nn.LayerNorm(plane3*8*8)
                nn.ReLU(inplace=True)
            )

        self.lstm = nn.LSTMCell(plane3 * 8 * 8, 256)

        self.cls = nn.Sequential(
            nn.Linear(256,1),
            nn.Tanh()
        )

        self.min_length = 10

    def forward(self, imgs):
        B,S,C,H,W = imgs.shape
        imgs = imgs.view(-1,C,H,W)
        x = self.encoder(imgs)
        x = x.view(B,S,-1)
        x = x.transpose(0,1)

        # seq = []
        seq = x.split(x,1,dim=0)

        hxs = []
        hx,cx = self.lstm(seq[0], (h_0, c_0))
        for i in range(len(seq)-1):
            hx, cx = self.lstm(seq[i+1])
            if i >= self.min_length:
                hxs.append(hx)
        out = self.cls(torch.cat(hxs).transpose(0,1))

        return out

    def inf(self,img):
        x = self.encoder(img)
        x = x.view(1,-1)
        hx,cx= self.lstm(x, (self.hx,self.cx))

        self.hx = hx
        self.cx = cx

        out = self.cls(hx)
        return out


    def reset(self):
        self.hx = 0
        self.cx = 0