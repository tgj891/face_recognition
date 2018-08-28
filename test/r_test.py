import torch, time
import net.checknet as nets
import tool.utils as utils
from PIL import Image

if __name__ == '__main__':
    #net
    pnet = nets.P_Net(istraining=False)
    rnet = nets.R_Net(istraining=False)
    if torch.cuda.is_available():
        pnet = pnet.cuda()
        rnet = rnet.cuda()
    pnet.eval()
    rnet.eval()
    pnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\p_params.pkl')) # 导入训练参数
    rnet.load_state_dict(torch.load(r'C:\Users\liev\Desktop\myproject\face_recognition\params\r_params.pkl'))  # 导入训练参数

    #输入图片
    img = Image.open(r"C:\Users\liev\Desktop\data\face_match_pic\14ce36d3d539b6009c2f88f9ea50352ac75cb79f.jpg")

    # P网络
    start_tim = time.time()
    pboxs = utils.PnetDetect(pnet, img, imgshow=False)
    pnet_tim = time.time()
    # R网络
    rboxs = utils.RnetDetect(rnet, img, pboxs,imgshow=True, show_conf=False)
    rnet_tim = time.time()

    print("pnet_time:", pnet_tim-start_tim, "rnet_time:", rnet_tim-pnet_tim)


