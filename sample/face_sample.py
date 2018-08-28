import torch
import net.checknet as nets
import tool.utils as utils
from PIL import Image,ImageDraw
import time
import cv2,os
import numpy as np

if __name__ == '__main__':
    #net
    pnet = nets.P_Net(istraining=False)
    rnet = nets.R_Net(istraining=False)
    onet = nets.O_Net(istraining=False, island=True)
    if torch.cuda.is_available():
        pnet = pnet.cuda()
        rnet = rnet.cuda()
        onet = onet.cuda()
    pnet.eval()
    rnet.eval()
    onet.eval()
    pnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\p_params.pkl')) # 导入训练参数
    rnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\r_params.pkl'))  # 导入训练参数
    onet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\o_params_ok.pkl'))  # 导入训练参数

    filelist = os.listdir(r"C:\Users\liev\Desktop\data\facedata\sample\bailiangzhi")
    count = 0
    for file in filelist:
        print(file)
        # 输入图片
        img = Image.open(os.path.join(r"C:\Users\liev\Desktop\data\facedata\sample\bailiangzhi", file))
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
        for box in oboxs:
            count += 1
            w = box[2] - box[0]
            h = box[3] - box[1]
            side = np.maximum(w, h)
            cx = box[0] + w/2
            cy = box[1] + h/2
            x1 = cx - side/2
            y1 = cy - side/2
            x2 = cx + side / 2
            y2 = cy + side / 2
            tempimg = img.crop((x1, y1, x2, y2))
            tempimg = tempimg.resize((48, 48), Image.ANTIALIAS)
            tempimg.save(r"C:\Users\liev\Desktop\data\facedata\recognition_face\-B%d.jpg"%(count))
