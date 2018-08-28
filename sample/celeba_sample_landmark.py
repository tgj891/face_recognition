import numpy as np
import os
from tool.utils import IouDo
from PIL import Image, ImageFilter
from PIL import ImageDraw

img_path = r"D:\facedata\img_celeba\img_celeba"
label_path = r"D:\facedata\list_bbox_landmark_m.txt"
handel_path = r"D:\face_check\face_data\facedata"
# img_path = r"D:\facedata\test"
# label_path = r"D:\facedata\test.txt"
# handel_path = r"D:\face_check\face_data\test"

def mkdir(size):
    rootpath = os.path.join(handel_path, str(size))
    if not os.path.exists(rootpath):
        os.mkdir(rootpath)

    p_dirpath = os.path.join(rootpath, "positive")
    if not os.path.exists(p_dirpath):
        os.mkdir(p_dirpath)

    n_dirpath = os.path.join(rootpath, "negative")
    if not os.path.exists(n_dirpath):
        os.mkdir(n_dirpath)

    t_dirpath = os.path.join(rootpath, "part")
    if not os.path.exists(t_dirpath):
        os.mkdir(t_dirpath)

    return rootpath, p_dirpath, n_dirpath, t_dirpath

def sample_handle(size):
    imgcount = 0

    r_path, p_path, n_path, t_path = mkdir(size)  # 创建目录
    p_file = open(r_path + "/positive.txt", "w")
    n_file = open(r_path + "/negative.txt", "w")
    t_file = open(r_path + "/part.txt", "w")
    for index, lines in enumerate(open(label_path).readlines()):
        if index < 2:
            continue
        strs = lines.strip().split(" ")
        strs = list(filter(bool, strs))
        print(strs)
        filename = strs[0]
        #原始坐标
        x1 = float(strs[1])
        y1 = float(strs[2])
        w = float(strs[3])
        h = float(strs[4])
        x2 = x1 + w
        y2 = y1 + h

        #5个特征点的位置
        fx1 = float(strs[5])
        fy1 = float(strs[6])
        fx2 = float(strs[7])
        fy2 = float(strs[8])
        fx3 = float(strs[9])
        fy3 = float(strs[10])
        fx4 = float(strs[11])
        fy4 = float(strs[12])
        fx5 = float(strs[13])
        fy5 = float(strs[14])

        if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
            continue

        cx = x1 + w*0.5 #中心点
        cy = y1 + h*0.5
        side = np.maximum(w, h)
        img = Image.open(os.path.join(img_path, filename))
        width, high = img.size
        # r_img = ImageDraw.Draw(img)
        # r_img.rectangle((x1,y1,x2,y2))
        for count in range(5):
            # 随机浮动产生正方形正、负、部分样本
            offset_side = np.random.uniform(-0.2, 0.2) * side
            offset_x = np.random.uniform(-0.2, 0.2) * w/2
            offset_y = np.random.uniform(-0.2, 0.2) * h/2
            _cx = cx + offset_x
            _cy = cy + offset_y
            _side = side + offset_side
            _x1 = np.maximum(_cx - _side * 0.5, 0)
            _y1 = np.maximum(_cy - _side * 0.5, 0)
            _x2 = _x1 + _side
            _y2 = _y1 + _side

            #计算偏移值
            offset_x1 = (x1 - _x1) / _side
            offset_y1 = (y1 - _y1) / _side
            offset_x2 = (x2 - _x2) / _side
            offset_y2 = (y2 - _y2) / _side

            offset_fx1 = (fx1 - _x1) / _side
            offset_fy1 = (fy1 - _y1) / _side
            offset_fx2 = (fx2 - _x1) / _side
            offset_fy2 = (fy2 - _y1) / _side
            offset_fx3 = (fx3 - _x1) / _side
            offset_fy3 = (fy3 - _y1) / _side
            offset_fx4 = (fx4 - _x1) / _side
            offset_fy4 = (fy4 - _y1) / _side
            offset_fx5 = (fx5 - _x1) / _side
            offset_fy5 = (fy5 - _y1) / _side

            #计算IOU
            #[x1, y1, x2, y2, 置信度]
            box = np.array([x1, y1, x2, y2, 0])
            boxs = np.array([[_x1, _y1, _x2, _y2, 0]])
            per = IouDo(box, boxs, mode="UNIUM")
            per = per[0]
            #截取图片
            #img.show()
            # r_img.rectangle((_x1, _y1, _x2, _y2))
            # img.show()
            tempimg = img.crop((_x1, _y1, _x2, _y2))
            tempimg = tempimg.resize((size, size),Image.ANTIALIAS)
            imglist = []
            imglist.append(tempimg)
            #图片模糊处理
            filterimg = tempimg.filter(ImageFilter.BLUR)
            imglist.append(filterimg)
            for _tempimg in imglist:
                if per > 0.65: #正样本
                    imgcount += 1
                    _tempimg.save("{0}/{1}.jpg".format(p_path, imgcount))
                    p_file.write("{0}.jpg 1 {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n".format(imgcount, offset_x1, offset_y1, offset_x2, offset_y2
                                    ,offset_fx1,offset_fy1,offset_fx2,offset_fy2,offset_fx3,offset_fy3,offset_fx4,offset_fy4,offset_fx5,offset_fy5))

                elif per < 0.3: #负样本
                    imgcount += 1
                    _tempimg.save("{0}/{1}.jpg".format(n_path, imgcount))
                    n_file.write("{0}.jpg 0 {1} {2} {3} {4} 0 0 0 0 0 0 0 0 0 0\n".format(imgcount, 0, 0, 0, 0))

                elif (per > 0.4) and (per < 0.65): #部分样本
                    imgcount += 1
                    _tempimg.save("{0}/{1}.jpg".format(t_path, imgcount))
                    t_file.write("{0}.jpg 2 {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n".format(imgcount, offset_x1, offset_y1, offset_x2, offset_y2
                                    ,offset_fx1,offset_fy1,offset_fx2,offset_fy2,offset_fx3,offset_fy3,offset_fx4,offset_fy4,offset_fx5,offset_fy5))

        #再创建负样本
        for i in range(10):
            offset_side = np.random.uniform(-0.2, 0.2) * side
            _side = side + offset_side
            _x1 = np.random.uniform(0, width - _side)
            _y1 = np.random.uniform(0, high - _side)
            _x2 = _x1 + _side
            _y2 = _y1 + _side

            # 计算偏移值
            offset_x1 = (x1 - _x1) / _side
            offset_y1 = (y1 - _y1) / _side
            offset_x2 = (x2 - _x2) / _side
            offset_y2 = (y2 - _y2) / _side

            # 计算IOU
            # [x1, y1, x2, y2, 置信度]
            box = np.array([x1, y1, x2, y2, 0])
            boxs = np.array([[_x1, _y1, _x2, _y2, 0]])
            per = IouDo(box, boxs, mode="UNIUM")
            per = per[0]
            # 截取图片
            tempimg = img.crop((_x1, _y1, _x2, _y2))
            tempimg = tempimg.resize((size, size),Image.ANTIALIAS)
            imglist = []
            imglist.append(tempimg)
            filterimg = tempimg.filter(ImageFilter.BLUR)
            imglist.append(filterimg)
            for _tempimg in imglist:
                if per < 0.3:
                    imgcount += 1
                    _tempimg.save("{0}/{1}.jpg".format(n_path, imgcount))
                    n_file.write("{0}.jpg 0 {1} {2} {3} {4} 0 0 0 0 0 0 0 0 0 0\n".format(imgcount, 0, 0, 0, 0))

    p_file.close()
    n_file.close()
    t_file.close()

if __name__ == '__main__':
    # sample_handle(12)
    sample_handle(24)
    sample_handle(48)