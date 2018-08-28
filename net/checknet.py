import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)

class P_Net(nn.Module):
    def __init__(self, istraining=True):
        super(P_Net, self).__init__()
        self.istraining = istraining
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)#用3*3做必须填充1
            , nn.PReLU()
            , nn.MaxPool2d(3, stride=2)#5*5*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, stride=1)#3*3*16
            , nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1) #1*1*32
            , nn.PReLU()
        )

        self.confidence = nn.Conv2d(32, 1, kernel_size=1, stride=1) #置信度
        self.offset = nn.Conv2d(32, 4, kernel_size=1, stride=1) #人脸框坐标偏移
        self.apply(weights_init)#权重初始化

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        off = self.offset(conv3)  # 偏移
        con = F.sigmoid(self.confidence(conv3))  # 置信度
        if self.istraining == True: #训练
            con = con.view(-1, 1) #NV
            off = off.view(-1, 4) #NV

        return con, off

class R_Net(nn.Module):
    def __init__(self, istraining=True):
        super(R_Net, self).__init__()
        self.istraining = istraining
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1)#用3*3做pool必须填充1
            , nn.PReLU()
            , nn.MaxPool2d(3, stride=2)#11*11*28
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(28, 48, kernel_size=3, stride=1)
            , nn.PReLU()
            , nn.MaxPool2d(3, stride=2)  # 4*4*48
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=2, stride=1) #3*3*64
            , nn.PReLU()
        )

        self.confidence = nn.Conv2d(64, 1, kernel_size=3, stride=1)#1*1*1
        self.offset = nn.Conv2d(64, 4, kernel_size=3, stride=1)##1*1*4
        self.apply(weights_init)#权重初始化

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        con = F.sigmoid(self.confidence(conv3))  # 置信度
        off = self.offset(conv3)  # 偏移
        if self.istraining == True: #训练
            con = con.view(-1, 1) #NV
            off = off.view(-1, 4) #NV

        return con, off

class O_Net(nn.Module):
    def __init__(self, istraining=True, island=False):
        super(O_Net, self).__init__()
        self.istraining = istraining
        self.island = island
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#用3*3做必须填充1
            , nn.PReLU()
            , nn.MaxPool2d(3, stride=2)#23*23*32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1)
            , nn.PReLU()
            , nn.MaxPool2d(3, stride=2)  # 10*10*64
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
            , nn.PReLU()
            , nn.MaxPool2d(2, stride=2) #4*4*64
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1) # 3*3*128
            , nn.PReLU()
        )

        self.confidence = nn.Conv2d(128, 1, kernel_size=3, stride=1)
        self.offset = nn.Conv2d(128, 4, kernel_size=3, stride=1)
        self.landoff = nn.Conv2d(128, 10, kernel_size=3, stride=1)
        self.apply(weights_init)#权重初始化

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        con = F.sigmoid(self.confidence(conv4))  # 置信度
        off = self.offset(conv4)  # 偏移
        if self.istraining == True: #训练
            con = con.view(-1, 1) #NV
            off = off.view(-1, 4) #NV

        if self.island:
            landoff = self.landoff(conv4)
            if self.istraining == True:  #训练
                landoff = landoff.view(-1, 10) #NV
            return con, off, landoff
        else:
            return con, off

