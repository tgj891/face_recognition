import torch,os
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import net.recognition as recog
from PIL import Image

def FeatureMatch(feature):
    dirlist = os.listdir(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku")
    reslist = []
    for dir in dirlist:
        dirpath = os.path.join(r"C:\Users\liev\Desktop\myproject\face_recognition\recog_ku", dir)
        filelist = os.listdir(dirpath)
        for file in filelist:
            feature_ = np.load(os.path.join(dirpath, file))
            feature_ = np.reshape(feature_, (-1))
            feature = np.reshape(feature, (-1))

            cosine = np.sum(feature * feature_)
            reslist.append([cosine, int(dir)])

    res = np.array(reslist, dtype=np.float32)
    index = np.argmax(res[:, 0])
    if int(res[index, 1]) == 0:
        name = "bainiangzi"
    elif int(res[index, 1]) == 1:
        name = "xuxian"
    elif int(res[index, 1]) == 2:
        name = "xiaoqing"
    elif int(res[index, 1]) == 3:
        name = "jiejie"
    elif int(res[index, 1]) == 4:
        name = "ligongpu"
    elif int(res[index, 1]) == 5:
        name = "fahai"

    return res[index, 0], name

if __name__ == '__main__':
    net = recog.RecognitionNet()
    imgdata = Image.open(r"C:\Users\liev\Desktop\data\facedata\okfaces\fahai-707.jpg")
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    net.load_state_dict(torch.load(r"C:\Users\liev\Desktop\myproject\face_recognition\params\net_params.pkl"))

    imgdata = np.array(imgdata, dtype=np.float32)/255.
    imgdata = np.expand_dims(imgdata, axis=0)
    testxs = np.transpose(imgdata, [0, 3, 1, 2])
    testxs = Variable(torch.FloatTensor(testxs))
    if torch.cuda.is_available():
        testxs = testxs.cuda()
    _, outfeature = net(testxs)
    outfeature = outfeature.cpu().data.numpy()
    #相识度比较
    #做归一化
    outfeature = np.reshape(outfeature, (-1, 512*4))
    outfeature_ = outfeature / np.sqrt(np.sum(outfeature ** 2))
    consine, res = FeatureMatch(outfeature_)

    print("相似度:", consine, "识别结果:", res)
