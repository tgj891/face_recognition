import torch
import net.checknet as nets
import tool.utils as utils
from PIL import Image
import numpy as np
import time
import cv2

if __name__ == '__main__':
    #net
    pnet = nets.P_Net(istraining=False)
    rnet = nets.R_Net(istraining=False)
    onet = nets.O_Net(istraining=False, island=False)
    if torch.cuda.is_available():
        pnet = pnet.cuda()
        rnet = rnet.cuda()
        onet = onet.cuda()
    pnet.eval()
    rnet.eval()
    onet.eval()
    pnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\p_params_ok.pkl')) # 导入训练参数
    rnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\r_params_ok.pkl'))  # 导入训练参数
    onet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\o_params_ok.pkl'))  # 导入训练参数

    # cap = cv2.VideoCapture(r"C:\Users\liev\Desktop\myproject\face_recognition\test\test.mp4")
    cap = cv2.VideoCapture(0)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            # if count % 1 == 0:
            # 转换通道, 转成Image 格式
            b, g, r = cv2.split(frame)
            img = cv2.merge([r, g, b])
            img = Image.fromarray(img.astype(np.uint8))

            # P网络
            pboxs = utils.PnetDetect(pnet, img, imgshow=False)
            # R网络
            rboxs = utils.RnetDetect(rnet, img, pboxs, imgshow=False)
            # O网络
            oboxs = utils.OnetDetect(onet, img, rboxs, imgshow=True, show_conf=False, isuse=True)
            img = np.array(img, dtype=np.uint8)

            # #转换通道BGR
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            cv2.imshow("img", img)
            # else:
            #     cv2.imshow("img", frame)
        else:
            continue

        k = cv2.waitKey(1)
        # q键退出
        if (k & 0xff == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
