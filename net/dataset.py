import torch,os
import numpy as np
from PIL import Image
from torch.utils import data

# p_12txtpath = r"D:\face_check\face_data\test\12\positive.txt"
# n_12txtpath = r"D:\face_check\face_data\test\12\negative.txt"
# t_12txtpath = r"D:\face_check\face_data\test\12\part.txt"
p_12txtpath = r"C:\Users\liev\Desktop\data\facedata\12\positive.txt"
n_12txtpath = r"C:\Users\liev\Desktop\data\facedata\12\negative.txt"
t_12txtpath = r"C:\Users\liev\Desktop\data\facedata\12\part.txt"
p_24txtpath = r"C:\Users\liev\Desktop\data\facedata\24\positive.txt"
n_24txtpath = r"C:\Users\liev\Desktop\data\facedata\24\negative.txt"
t_24txtpath = r"C:\Users\liev\Desktop\data\facedata\24\part.txt"
p_48txtpath = r"C:\Users\liev\Desktop\data\facedata\48\positive.txt"
n_48txtpath = r"C:\Users\liev\Desktop\data\facedata\48\negative.txt"
t_48txtpath = r"C:\Users\liev\Desktop\data\facedata\48\part.txt"

#图片地址
# p_12imgpath = r"D:\face_check\face_data\test\12\positive"
# n_12imgpath = r"D:\face_check\face_data\test\12\negative"
# t_12imgpath = r"D:\face_check\face_data\test\12\part"
p_12imgpath = r"C:\Users\liev\Desktop\data\facedata\12\positive"
n_12imgpath = r"C:\Users\liev\Desktop\data\facedata\12\negative"
t_12imgpath = r"C:\Users\liev\Desktop\data\facedata\12\part"
p_24imgpath = r"C:\Users\liev\Desktop\data\facedata\24\positive"
n_24imgpath = r"C:\Users\liev\Desktop\data\facedata\24\negative"
t_24imgpath = r"C:\Users\liev\Desktop\data\facedata\24\part"
p_48imgpath = r"C:\Users\liev\Desktop\data\facedata\48\positive"
n_48imgpath = r"C:\Users\liev\Desktop\data\facedata\48\negative"
t_48imgpath = r"C:\Users\liev\Desktop\data\facedata\48\part"

class MyDataSet(data.Dataset):
    def __init__(self, islandmarks, p_path, n_path, t_path, p_imgpath, n_imgpath, t_imgpath):
        super(MyDataSet, self).__init__()
        self.p_path = p_path
        self.n_path = n_path
        self.t_path = t_path
        self.islandmarks = islandmarks
        self.p_imgpath = p_imgpath
        self.n_imgpath = n_imgpath
        self.t_imgpath = t_imgpath

        self.dataset = []
        p_file = None
        n_file = None
        t_file = None
        if p_path:
            p_file = open(p_path, "r")
        if n_path:
            n_file = open(n_path, "r")
        if t_path:
            t_file = open(t_path, "r")

        #正负部分样本插入添加，减少相关性
        if p_file:
            pdata = p_file.readlines()
            self.dataset.extend(np.random.choice(pdata, size=30000))
        if n_file:
            ndata = n_file.readlines()
            self.dataset.extend(np.random.choice(ndata, size=90000))
        if t_file:
            tdata = t_file.readlines()
            self.dataset.extend(np.random.choice(tdata, size=30000))

    def __getitem__(self, index):
        "{0}.jpg 0 x1 y1 x2 y2 置信度"
        strs = self.dataset[index].strip().split(" ")
        if strs[1] == '0':#负样本
            imgdata = np.array(Image.open(os.path.join(self.n_imgpath, strs[0])), dtype=np.float32) / 255.
        elif strs[1] == '1':#正样本
            imgdata = np.array(Image.open(os.path.join(self.p_imgpath, strs[0])), dtype=np.float32) / 255
        elif strs[1] == '2':#部分样本
            imgdata = np.array(Image.open(os.path.join(self.t_imgpath, strs[0])), dtype=np.float32) / 255

        imgdata = imgdata.transpose([2, 0, 1]) #CHW
        img_data = torch.FloatTensor(imgdata)
        confidence = torch.FloatTensor(np.array([float(strs[1])]))
        offset = torch.FloatTensor(np.array([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])]))
        if self.islandmarks == True:
            landoff = torch.FloatTensor(np.array([float(strs[6]), float(strs[7]), float(strs[8]), float(strs[9])
                        , float(strs[10]), float(strs[11]), float(strs[12]),float(strs[13]), float(strs[14]), float(strs[15])]))
            return img_data, confidence, offset, landoff
        else:
            return img_data, confidence, offset

    def __len__(self):
        return len(self.dataset)

#测试代码
if __name__ == '__main__':
    mydata = MyDataSet(False, p_12txtpath, n_12txtpath, t_12txtpath, p_12imgpath, n_12imgpath, t_12imgpath)
    dataloader = data.DataLoader(mydata, batch_size=10, shuffle=True)
    iters = iter(dataloader)
    imgdata, con, offset = iters.next()
    # conn_mask = torch.lt(con, 2)
    # conn_ = con[conn_mask]
    #
    # off_mask = torch.gt(con, 0)
    # off_ = offset[off_mask[:, 0]]
    print(con)
    # print(str)
