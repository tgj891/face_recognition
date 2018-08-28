import numpy as np
import torch, time
from PIL import Image, ImageDraw, ImageFilter
from torch.autograd import Variable
import matplotlib.pyplot as plt

def BoxsAndConfidence(img, org_boxs, confi, offset):
    for index in range(confi.shape[0]):
        # 建议框坐标
        _x1 = org_boxs[index][0]
        _y1 = org_boxs[index][1]
        _x2 = org_boxs[index][2]
        _y2 = org_boxs[index][3]
        w = _x2 - _x1
        h = _y2 - _y1
        offx1 = offset[index][0]
        offy1 = offset[index][1]
        offx2 = offset[index][2]
        offy2 = offset[index][3]
        # 真实框坐标
        x1 = _x1 + offx1 * w
        y1 = _y1 + offy1 * h
        x2 = _x2 + offx2 * w
        y2 = _y2 + offy2 * h
        _img = ImageDraw.Draw(img)
        _img.rectangle((x1, y1, x2, y2), outline="red")
        _img.text((x1, y1), str(confi[index]), "black")
    img.show()

def iou(box, boxes, isMin = False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr

def IouDo(box, boxs, mode="UNIUM"):
    #print(boxs)
    _x1 = boxs[:, 0]
    _y1 = boxs[:, 1]
    _x2 = boxs[:, 2]
    _y2 = boxs[:, 3]

    x1 = np.maximum(box[0], _x1)
    y1 = np.maximum(box[1], _y1)
    x2 = np.minimum(box[2], _x2)
    y2 = np.minimum(box[3], _y2)

    w = np.maximum(0, x2-x1)
    h = np.maximum(0, y2-y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxs_area = (_x2-_x1)*(_y2-_y1)
    inter_area = w * h  # 交集
    unium_area = box_area + boxs_area - inter_area #并集

    if mode == "UNIUM":
        per = np.true_divide(inter_area, unium_area)
    else:
        per = np.true_divide(inter_area, np.minimum(box_area, boxs_area))

    return per

def GetBatch(dataloader, island=False):
    iters = iter(dataloader)
    if island:
        data, confidence, offset, landoff = iters.next()
        return data, confidence, offset, landoff
    else:
        data, confidence, offset = iters.next()
        return data, confidence, offset

def NmsDo(boxs, mode):
    keeplists = [] #保留列表
    boxsarry = np.array(boxs, dtype=np.float32)

    while boxsarry.shape[0] > 0:
        maxcon_index = np.argmax(boxsarry[:, 4])
        box = boxsarry[maxcon_index] #置信度最大的框
        keeplists.append(box) #添加到保留列表中
        boxs1 = np.delete(boxsarry, maxcon_index, 0) #删除第maxcon_index行
        if boxs1.shape[0] == 0:
            break
        index1= np.where(IouDo(box, boxs1, mode) < 0.3) #保留iou小于0.3的框
        boxsarry = boxs1[index1]
    return keeplists

def PnetDetect(net, img, imgshow=False):
    width, high = img.size
    scale = 1  # 缩放比列
    copyimg = img
    boxslist = []  # 框列表
    sidelen = np.minimum(width, high)
    imglist = []
    while sidelen > 12:  # 最小尺寸
        imgdata = np.array(copyimg, dtype=np.float32) / 255
        imgdata = np.expand_dims(imgdata, axis=0)  # 扩展维度变成4维
        imgdata = imgdata.transpose([0, 3, 1, 2])  # 轴变换NCHW
        imgdata = torch.FloatTensor(imgdata)
        imgdata = Variable(imgdata)
        if torch.cuda.is_available():
            imgdata = imgdata.cuda()

        confidence, offset = net(imgdata)  # 置信度和偏移
        confidence = confidence[0][0].cpu().data.numpy()
        offset = offset[0].cpu().data.numpy()
        # confidence = np.squeeze(confidence)  # 降维(h, w) ,去掉维度为1的
        # offset = np.squeeze(offset)  # 降维(c, h, w)
        off_x1 = offset[0]
        off_y1 = offset[1]
        off_x2 = offset[2]
        off_y2 = offset[3]
        # 筛选置信度大于阈值的框
        indexs = np.where(confidence > 0.6)  # np.where()[0] 表示行的索引 1是列
        indexs = np.stack(indexs, axis=1)
        if indexs.shape[0] > 0:
            # 反算坐标
            # for index in indexs:
            _x1 = (indexs[:, 1] * 2) / scale
            _y1 = (indexs[:, 0] * 2) / scale
            _x2 = (indexs[:, 1] * 2 + 12) / scale
            _y2 = (indexs[:, 0] * 2 + 12) / scale
            w = _x2 - _x1
            h = _y2 - _y1
            offx1 = off_x1[indexs[:, 0], indexs[:, 1]]
            offy1 = off_y1[indexs[:, 0], indexs[:, 1]]
            offx2 = off_x2[indexs[:, 0], indexs[:, 1]]
            offy2 = off_y2[indexs[:, 0], indexs[:, 1]]
            conf = confidence[indexs[:, 0], indexs[:, 1]]
            # 真实框坐标
            x1 = _x1 + offx1 * w
            y1 = _y1 + offy1 * h
            x2 = _x2 + offx2 * w
            y2 = _y2 + offy2 * h
            # h_img.rectangle((x1, y1, x2, y2), outline="red")
            boxslist.extend(np.stack([x1, y1, x2, y2, conf], axis=1))

        # 图片缩放
        scale *= 0.7
        _width = int(width * scale)
        _high = int(high * scale)
        copyimg = img.resize((_width, _high), Image.ANTIALIAS)
        sidelen = np.minimum(_width, _high)

    #NMS
    oklist = NmsDo(boxslist, "UNIUM")
    if imgshow == True:
        h_img = ImageDraw.Draw(img)
        for box in oklist:
            h_img.rectangle((box[0],box[1],box[2],box[3]), outline="red")
            h_img.text((box[0],box[1]), str(box[4]), "black")
        img.show()
    return oklist

def RnetDetect(net, img, boxs, imgshow=False, show_conf=False):
    imglist = []
    rboxslist = []
    boxss = np.array(boxs)
    if boxss.shape[0] == 0:
        return []
    #把框变成24*24的正方形
    x1 = boxss[:, 0]
    y1 = boxss[:, 1]
    x2 = boxss[:, 2]
    y2 = boxss[:, 3]
    w = x2 - x1
    h = y2 - y1
    sidelen = np.maximum(w, h)
    cx = x1 + w / 2
    cy = y1 + h / 2
    _x1 = cx - sidelen / 2
    _y1 = cy - sidelen / 2
    _x2 = cx + sidelen / 2
    _y2 = cy + sidelen / 2

    _boxs = np.stack([_x1, _y1, _x2, _y2], axis=1)
    for i in range(_boxs.shape[0]):
        cropimg = img.crop((_boxs[i,0], _boxs[i,1], _boxs[i,2], _boxs[i,3]))
        imgdata = cropimg.resize((24, 24), Image.ANTIALIAS)
        # cropimg.show()
        cropimg = np.array(imgdata, dtype=np.float32)/255
        cropimg = cropimg.transpose([2, 0, 1]) #CHW
        imglist.append(cropimg)

    #传入R网络
    imgdata = np.array(imglist)
    imgdata = torch.FloatTensor(imgdata)
    imgdata = Variable(imgdata)
    if torch.cuda.is_available():
        imgdata = imgdata.cuda()

    confidence, offset = net(imgdata)
    confidence = confidence.view(-1, 1) #N * 1
    offset = offset.view(-1, 4) #N * 4
    confidence = confidence.cpu().data.numpy()
    offset = offset.cpu().data.numpy()
    if show_conf:
        BoxsAndConfidence(img, _boxs, confidence, offset)
    # 筛选置信度大于阈值的框
    indexs = np.where(confidence > 0.7)  # np.where()[0] 表示行的索引 1是列
    indexs = np.stack(indexs, axis=1)
    if indexs.shape[0] > 0:
        # 反算坐标
        # for index in indexs:
        # 建议框坐标
        _x1 = _boxs[indexs[:,0], 0]
        _y1 = _boxs[indexs[:,0], 1]
        _x2 = _boxs[indexs[:,0], 2]
        _y2 = _boxs[indexs[:,0], 3]
        w = _x2 - _x1
        h = _y2 - _y1
        offx1 = offset[indexs[:,0],0]
        offy1 = offset[indexs[:,0],1]
        offx2 = offset[indexs[:,0],2]
        offy2 = offset[indexs[:,0],3]
        conf = confidence[indexs[:,0],indexs[:,1]]
        # 真实框坐标
        x1 = _x1 + offx1 * w
        y1 = _y1 + offy1 * h
        x2 = _x2 + offx2 * w
        y2 = _y2 + offy2 * h
        rboxslist.extend(np.stack([x1, y1, x2, y2, conf],axis=1))

    #做NMS
    oklist = NmsDo(rboxslist, "UNIUM")
    if imgshow == True:
        h_img = ImageDraw.Draw(img)
        for box in oklist:
            h_img.rectangle((box[0], box[1], box[2], box[3]), outline="red")
            h_img.text((box[0], box[1]), str(box[4]), "black")
        img.show()
    return oklist

def OnetDetect(net, img, boxs, imgshow=False, show_conf=False, isuse=False):
    imglist = []
    oboxslist = []
    boxss = np.array(boxs)
    if boxss.shape[0] == 0:
        return []

    #把框变成48*48的正方形
    x1 = boxss[:, 0]
    y1 = boxss[:, 1]
    x2 = boxss[:, 2]
    y2 = boxss[:, 3]
    w = x2 - x1
    h = y2 - y1
    sidelen = np.maximum(w, h)
    cx = x1 + w / 2
    cy = y1 + h / 2
    _x1 = cx - sidelen / 2
    _y1 = cy - sidelen / 2
    _x2 = cx + sidelen / 2
    _y2 = cy + sidelen / 2

    _boxs = np.stack([_x1, _y1, _x2, _y2], axis=1)
    for box in _boxs:
        cropimg = img.crop((box[0], box[1], box[2], box[3]))
        imgdata = cropimg.resize((48, 48), Image.ANTIALIAS)
        cropimg = np.array(imgdata, dtype=np.float32)/255.
        cropimg = cropimg.transpose([2, 0, 1]) #CHW
        imglist.append(cropimg)

    #传入O网络
    imgdata = np.array(imglist)
    imgdata = torch.FloatTensor(imgdata)
    imgdata = Variable(imgdata)
    if torch.cuda.is_available():
        imgdata = imgdata.cuda()

    if net.island:
        confidence, offset, landoff = net(imgdata)
        landoff = landoff.view(-1, 10)  # N * 10
        landoff = landoff.cpu().data.numpy()
    else:
        confidence, offset = net(imgdata)

    confidence = confidence.view(-1, 1) #N * 1
    offset = offset.view(-1, 4) #N * 4
    confidence = confidence.cpu().data.numpy()
    offset = offset.cpu().data.numpy()
    # 筛选置信度大于阈值的框
    if show_conf:
        BoxsAndConfidence(img, _boxs, confidence, offset)
    indexs = np.where(confidence > 0.95)  # np.where()[0] 表示行的索引 1是列
    indexs = np.stack(indexs, axis=1)
    if indexs.shape[0] > 0:
        # 反算坐标
        # for index in indexs:
        # 建议框坐标
        _x1 = _boxs[indexs[:,0],0]
        _y1 = _boxs[indexs[:,0],1]
        _x2 = _boxs[indexs[:,0],2]
        _y2 = _boxs[indexs[:,0],3]
        w = _x2 - _x1
        h = _y2 - _y1
        offx1 = offset[indexs[:,0],0]
        offy1 = offset[indexs[:,0],1]
        offx2 = offset[indexs[:,0],2]
        offy2 = offset[indexs[:,0],3]
        # 真实框坐标
        x1 = _x1 + offx1 * w
        y1 = _y1 + offy1 * h
        x2 = _x2 + offx2 * w
        y2 = _y2 + offy2 * h
        conf = confidence[indexs[:,0], indexs[:,1]]
        #5个特征点的坐标
        if net.island:
            landoffx1 = landoff[indexs[:,0],0]
            landoffy1 = landoff[indexs[:,0],1]
            landoffx2 = landoff[indexs[:,0],2]
            landoffy2 = landoff[indexs[:,0],3]
            landoffx3 = landoff[indexs[:,0],4]
            landoffy3 = landoff[indexs[:,0],5]
            landoffx4 = landoff[indexs[:,0],6]
            landoffy4 = landoff[indexs[:,0],7]
            landoffx5 = landoff[indexs[:,0],8]
            landoffy5 = landoff[indexs[:,0],9]

            fx1 = _x1 + landoffx1 * w
            fy1 = _y1 + landoffy1 * h
            fx2 = _x1 + landoffx2 * w
            fy2 = _y1 + landoffy2 * h
            fx3 = _x1 + landoffx3 * w
            fy3 = _y1 + landoffy3 * h
            fx4 = _x1 + landoffx4 * w
            fy4 = _y1 + landoffy4 * h
            fx5 = _x1 + landoffx5 * w
            fy5 = _y1 + landoffy5 * h

            oboxslist.extend(np.stack([x1, y1, x2, y2, conf
                            , fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5],axis=1))
        else:
            oboxslist.extend(np.stack([x1, y1, x2, y2, conf],axis=1))

    #做NMS
    oklist = NmsDo(oboxslist, "OTHER")
    if imgshow == True:
        h_img = ImageDraw.Draw(img)
        if net.island:
            for box in oklist:
                h_img.rectangle((box[0], box[1], box[2], box[3]), outline="red")
                # h_img.point([(box[5], box[6]),(box[7], box[8]),(box[9], box[10]),(box[11], box[12]),(box[13], box[14])]
                #             ,fill=(255, 0, 0))
        else:
            for box in oklist:
                h_img.rectangle((box[0], box[1], box[2], box[3]), outline="red")

        if not isuse:
            img.show()
    return oklist

#图片清晰处理
def PicLight():
    im02 =Image.open(r"D:/ftp/originalPics/2003/01/17/big/img_610.jpg")
    im =im02.filter(ImageFilter.BLUR)
    im.show()


#imgF.filter(ImageFilter.BLUR) 均值滤波(普通模糊化)
# ImageFilter.GaussianBlur (高斯模糊)
def PicShow():

    im02 =Image.open(r"D:\facedata\img_celeba\img_celeba\000002.jpg")
    img = ImageDraw.Draw(im02)
    img.rectangle((72,94,95+221,71+306), outline="red")
    im02.show()

#测试程序
if __name__ == '__main__':
    a = np.array([
                [[2, 5], [10, 14]]
                , [[1, 2], [3, 4]]
                ], dtype=np.float32)

    b = np.array([
        [[2], [14]]
        , [[1], [4]]
    ], dtype=np.float32)

    xx = []
    # b = np.where(a[:]>2)
    # c = np.stack(b, axis=1)
    # print(a.shape)
    # print(b)
    # print(c)
    xx.append(a)
    xx.append(b)
#     print(a.shape)
#     b = np.delete(a, 0, 0)
#     print(b)
#     print(b.shape)

# if __name__ == '__main__':
#     PicLight()

