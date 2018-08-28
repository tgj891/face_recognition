import torch, time
import net.checknet as nets
import tool.utils as utils
from PIL import Image
import numpy as np

if __name__ == '__main__':
    net = nets.P_Net(istraining=False)
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    net.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\p_params.pkl')) # 导入训练参数
    #输入图片
    img = Image.open(r"./5.jpg")

    #P网络
    start_tim = time.time()
    pboxs = utils.PnetDetect(net, img, imgshow=True)
    end_tim = time.time()

    print("pnet_time:", end_tim - start_tim)