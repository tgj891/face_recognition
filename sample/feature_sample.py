import torch,os
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import net.recognition as recog
import net.recog_dataset as data
from PIL import Image

def GetFeature(features, num, count):
    feature = features[0] / np.sqrt(np.sum(features[0] ** 2))
    if num == 0:
        np.save(os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku\0", str(count[0]) +".npy"), feature)
        count[0] += 1
    elif num == 1:
        np.save(os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku\1", str(count[1]) + ".npy"), feature)
        count[1] += 1
    elif num == 2:
        np.save(os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku\2", str(count[2]) + ".npy"), feature)
        count[2] += 1
    elif num == 3:
        np.save(os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku\3", str(count[3]) + ".npy"), feature)
        count[3] += 1
    elif num == 4:
        np.save(os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku\4", str(count[4]) + ".npy"), feature)
        count[4] += 1
    elif num == 5:
        np.save(os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku\5", str(count[5]) + ".npy"), feature)
        count[5] += 1

if __name__ == '__main__':
    net = recog.RecognitionNet()
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    net.load_state_dict(torch.load(r"C:\Users\liev\Desktop\myproject\face_recognition\params\net_params.pkl"))
    countnum = np.zeros(6)
    dirs = os.listdir(r"C:\Users\liev\Desktop\myproject\face_recognition\pic_ku")
    for dir in dirs:
        filedir = os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\pic_ku", dir)
        filelists = os.listdir(filedir)
        for file in filelists:
            filepath = os.path.join(filedir, file)
            imgdata = np.array(Image.open(filepath), dtype=np.float32)/255.

            testxs = np.expand_dims(imgdata, axis=0)
            testxs = np.transpose(testxs, [0, 3, 1, 2])
            testxs = Variable(torch.FloatTensor(testxs))
            if torch.cuda.is_available():
                testxs = testxs.cuda()
            _, feature = net(testxs)
            feature = feature.cpu().data.numpy()
            feature = np.reshape(feature, (-1, 512*4))

            flag = GetFeature(feature, int(dir), countnum)

    print("Feature Get Ok!")
