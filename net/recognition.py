import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import net.recog_dataset as data

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)

#输入图片采用48*48*3的
class RecognitionNet(Module):
    def __init__(self):
        super(RecognitionNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0)#23*23
            ,nn.BatchNorm2d(32)
            ,nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)  # 11*11
            , nn.BatchNorm2d(64)
            , nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 6*6
            , nn.BatchNorm2d(128)
            , nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)  # 4*4
            , nn.BatchNorm2d(256)
            , nn.PReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)  # 2*2
            , nn.BatchNorm2d(512)
            , nn.PReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 6, kernel_size=2, stride=1, padding=0)  # 1*1
        )
        self.apply(weights_init)  # 权重初始化

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        feature = self.conv5(conv4)
        out = self.conv6(feature)
        out = F.softmax(out.view(-1, 6), dim=1)
        return out, feature

class CenterLoss(nn.Module):
    def __init__(self, class_num, feature_num):
        super(CenterLoss, self).__init__()
        self.center = Parameter(torch.randn((class_num, feature_num, 2, 2)))

    def forward(self, longlabel, floatlabel, maxnum, feature):
        newcenter = self.center.index_select(dim=0, index=longlabel)
        count = torch.histc(floatlabel, bins=int(maxnum+1), min=0, max=int(maxnum))
        num = count.index_select(dim=0, index=longlabel)
        loss = torch.mean(torch.sqrt(torch.sum((feature - newcenter) ** 2)) / num)
        return loss

if __name__ == '__main__':
    net = RecognitionNet()
    mydata = data.RecDataSet(r"C:\Users\liev\Desktop\data\facedata\okfaces")
    centernet = CenterLoss(6, 512)
    if torch.cuda.is_available():
        net = net.cuda()
    net.train()
    centernet.train()
    net.load_state_dict(torch.load(r"C:\Users\liev\Desktop\myproject\face_recognition\params\net_params.pkl"))  # 导入训练参数
    centernet.load_state_dict(torch.load(r"C:\Users\liev\Desktop\myproject\face_recognition\params\center_params.pkl"))  # 导入训练参数

    optimer = optim.Adam(net.parameters()) #创建优化器
    centeroptimer = optim.Adam(centernet.parameters())  # 创建优化器
    loss_fun = nn.MSELoss()  # 创建loss
    plt.ion()

    for epoch in range(100000):
        train_xs, train_ys = mydata.get_data(mydata)
        train_xs = train_xs.reshape((10, 48, 48, 3))
        train_xs = np.transpose(train_xs, [0, 3, 1, 2])
        xs = Variable(torch.FloatTensor(train_xs))
        ys = Variable(torch.FloatTensor(train_ys))
        if torch.cuda.is_available():
            xs = xs.cuda()
            ys = ys.cuda()

        out, feature = net(xs)
        #softmax 损失
        loss = torch.mean(loss_fun(out, ys))  # 计算loss
        optimer.zero_grad()
        loss.backward()
        optimer.step()

        # centerloss
        cys = np.argmax(train_ys, axis=1).reshape(10)
        maxnum = np.max(cys)
        floatcys = Variable(torch.FloatTensor(cys))
        longcys = Variable(torch.LongTensor(cys))
        # if torch.cuda.is_available():
        #     floatcys = floatcys.cuda()
        #     longcys = longcys.cuda()
        feature = feature.cpu().data
        centerloss = centernet(longcys, floatcys, maxnum, feature)
        centeroptimer.zero_grad()
        centerloss.backward()
        centeroptimer.step()

        if epoch % 10 == 0:
            torch.save(net.state_dict(), r"C:\Users\liev\Desktop\myproject\face_recognition\params\net_params.pkl")  # 保存训练参数
            torch.save(centernet.state_dict(), r"C:\Users\liev\Desktop\myproject\face_recognition\params\center_params.pkl")  # 保存训练参数

            out = out.cpu().data.numpy()
            accuracy = np.mean(np.array(np.argmax(out, axis=1) == np.argmax(train_ys, axis=1), dtype=np.float32))
            print("epoch:", epoch, "loss:",loss.cpu().data.numpy(), "accuracy:", accuracy, "centerloss", centerloss.cpu().data.numpy())

