import torch, os
import torch.nn as nn
import net.dataset as smpl
import matplotlib.pyplot as plt
import tool.utils as utils
from torch.utils import data
import torch.optim as optim
import net.checknet as cnet
from torch.autograd import Variable

class Train(nn.Module):
    def __init__(self, net, iscuda, island, parampath, p_txtpath, n_txtpath, t_txtpath, p_imgpath, n_imgpath, t_imgpath):
        super(Train, self).__init__()
        self.p_txtpath = p_txtpath
        self.n_txtpath = n_txtpath
        self.t_txtpath = t_txtpath
        self.p_imgpath = p_imgpath
        self.n_imgpath = n_imgpath
        self.t_imgpath = t_imgpath
        self.net = net
        self.iscuda = iscuda
        self.island = island
        self.parampath = parampath
        # 训练数据集
        dataset = smpl.MyDataSet(self.island, self.p_txtpath, self.n_txtpath, self.t_txtpath
                                     , self.p_imgpath, self.n_imgpath, self.t_imgpath)
        self.dataloader = data.DataLoader(dataset, batch_size=512, shuffle=True)

    def train(self):
        if self.net == "pnet":
            net = cnet.P_Net(istraining=True)
        elif self.net == "rnet":
            net = cnet.R_Net(istraining=True)
        elif self.net == "onet":
            net = cnet.O_Net(istraining=True, island=self.island)
        else:
            print("net is error")
            return False

        # if os.path.exists(self.parampath):
        #     net.load_state_dict(torch.load(self.parampath))  # 导入训练参数
        net.train()
        if self.iscuda == True:
            net = net.cuda()
        optimer = optim.Adam(net.parameters())
        con_lossfun = nn.BCELoss()  # 置信度损失函数
        off_lossfun = nn.MSELoss()  # 坐标的损失函数
        landoff_lossfun = nn.MSELoss()  # 坐标的损失函数
        for epoch in range(100000000):
            # 训练置信度
            if self.island:
                imgdata, con, offset, landoff = utils.GetBatch(self.dataloader, True)
                landoff = Variable(landoff)
                if self.iscuda == True:
                    landoff = landoff.cuda()
            else:
                imgdata, con, offset = utils.GetBatch(self.dataloader, False)
            imgdata = Variable(imgdata)
            con = Variable(con)
            offset = Variable(offset)
            if self.iscuda == True:
                imgdata, con, offset = imgdata.cuda(), con.cuda(), offset.cuda()

            if self.island:
                con_out, off_out, landoff_out = net(imgdata)
            else:
                con_out, off_out = net(imgdata)
            #计算置信度损失
            conn_mask = torch.lt(con, 2)
            conn_ = con[conn_mask]
            conn_out = con_out[conn_mask]
            con_loss = con_lossfun(conn_out, conn_)

            # 计算坐标偏移损失
            off_mask = torch.gt(con, 0)
            off_ = offset[off_mask[:, 0]]
            offf_out = off_out[off_mask[:, 0]]
            off_loss = off_lossfun(offf_out, off_)

            if self.island:
                #计算5个特征点的损失
                landoff_ = landoff[off_mask[:, 0]]
                landoff_out_ = landoff_out[off_mask[:, 0]]
                land_loss = landoff_lossfun(landoff_out_, landoff_)

                loss = con_loss + off_loss * 0.5 + land_loss
            else:
                loss = con_loss + off_loss*0.5
            optimer.zero_grad()
            loss.backward()
            optimer.step()

            # 画图
            # plt.scatter(epoch, conloss, c='r', marker='.')
            # plt.scatter(epoch, offloss, c='y', marker='.')
            # plt.show()
            # plt.pause(1)
            if epoch % 10 == 0:
                conloss = con_loss.cpu().data.numpy()
                offloss = off_loss.cpu().data.numpy()
                loss_ = loss.cpu().data.numpy()
                if self.island:
                    landloss = land_loss.cpu().data.numpy()
                    print("epoch:", epoch, "loss:", loss_, "conloss:", conloss, "offloss:", offloss, "landloss:",
                          landloss)
                else:
                    print("epoch:", epoch, "loss:", loss_, "conloss:", conloss, "offloss:", offloss)

                if True:
                    torch.onnx.export(net, imgdata, r"C:\Users\liev\Desktop\myproject\face_recognition\params\yolonet.proto", verbose=True)
                torch.save(net.state_dict(), self.parampath)  # 保存训练参数
            break

if __name__ == '__main__':
    # P_NET
    pnet = Train("pnet", True, False, r"C:\Users\liev\Desktop\myproject\face_recognition\params\p_params.pkl"
                     , smpl.p_12txtpath
                     , smpl.n_12txtpath
                     , smpl.t_12txtpath
                     , smpl.p_12imgpath
                     , smpl.n_12imgpath
                     , smpl.t_12imgpath
                     )
    # R_NET
    # rnet = Train("rnet", True, False, r"C:\Users\liev\Desktop\myproject\face_recognition\params\r_params.pkl"
    #                  , smpl.p_24txtpath
    #                  , smpl.n_24txtpath
    #                  , smpl.t_24txtpath
    #                  , smpl.p_24imgpath
    #                  , smpl.n_24imgpath
    #                  , smpl.t_24imgpath
    #                  )

    # O_NET
    # onet = Train("onet", True, True, r"C:\Users\liev\Desktop\myproject\face_recognition\params\o_params.pkl"
    #              , smpl.p_48txtpath
    #              , smpl.n_48txtpath
    #              , smpl.t_48txtpath
    #              , smpl.p_48imgpath
    #              , smpl.n_48imgpath
    #              , smpl.t_48imgpath
    #              )
    #训练
    pnet.train()

