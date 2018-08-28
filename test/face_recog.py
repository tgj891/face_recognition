import torch
import net.checknet as nets
import tool.utils as utils
from PIL import Image,ImageDraw
import time
import cv2
import numpy as np
import net.recognition as recog
from torch.autograd import Variable
import test.match as mt

if __name__ == '__main__':
    #net
    pnet = nets.P_Net(istraining=False)
    rnet = nets.R_Net(istraining=False)
    onet = nets.O_Net(istraining=False, island=True)
    renet = recog.RecognitionNet()

    if torch.cuda.is_available():
        pnet = pnet.cuda()
        rnet = rnet.cuda()
        onet = onet.cuda()
        renet = renet.cuda()
    pnet.eval()
    rnet.eval()
    onet.eval()
    renet.eval()
    pnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\p_params.pkl')) # 导入训练参数
    rnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\r_params.pkl'))  # 导入训练参数
    onet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\o_params_ok.pkl'))  # 导入训练参数
    renet.load_state_dict(torch.load(r"C:\Users\liev\Desktop\myproject\face_recognition\params\net_params.pkl")) #识别网络参数
    # 输入图片
    img = Image.open(r"C:\Users\liev\Desktop\data\facedata\sample\test\新白娘子传奇01 342.jpg")
    total_start_t = time.time()
    # P网络
    start_time = time.time()
    pboxs = utils.PnetDetect(pnet, img, imgshow=False)
    pnet_end_time = time.time()
    # R网络
    rboxs = utils.RnetDetect(rnet, img, pboxs, imgshow=False)
    rnet_end_time = time.time()
    # O网络
    oboxs = utils.OnetDetect(onet, img, rboxs, imgshow=False, show_conf=False)

    # 人脸识别
    for box in oboxs:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        side = np.maximum(w, h)

        _x1 = cx - side / 2
        _y1 = cy - side / 2
        _x2 = cx + side / 2
        _y2 = cy + side / 2
        tempimg = img.crop((_x1, _y1, _x2, _y2))
        tempimg = tempimg.resize((48, 48), Image.ANTIALIAS)

        imgdata = np.array(tempimg, dtype=np.float32) / 255.
        imgdata = np.expand_dims(imgdata, axis=0)
        testxs = np.transpose(imgdata, [0, 3, 1, 2])
        testxs = Variable(torch.FloatTensor(testxs))
        if torch.cuda.is_available():
            testxs = testxs.cuda()
        _, outfeature = renet(testxs)
        outfeature = outfeature.cpu().data.numpy()
        # 相识度比较
        # 做归一化
        outfeature = np.reshape(outfeature, (-1, 512 * 4))
        outfeature_ = outfeature / np.sqrt(np.sum(outfeature ** 2))
        consine, res = mt.FeatureMatch(outfeature_)

        h_img = ImageDraw.Draw(img)
        h_img.rectangle((x1, y1, x2, y2), outline="red")
        h_img.text((x1, y1), res+"-"+str(consine), "red")

    img.show()




