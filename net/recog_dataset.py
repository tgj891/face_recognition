import torch,os
import numpy as np
from PIL import Image
from torch.utils import data

class RecDataSet(data.Dataset):
    def __init__(self, path):
        super(RecDataSet, self).__init__()
        self.path = path
        self.dataset = []
        filelist = os.listdir(self.path)
        for file in filelist:
            self.dataset.append(file)

    def __getitem__(self, index):
        # bainiangzhi,xuxian,xiaoqing,jiejie,ligongpu,fahai
        # 0,1,2,3,4,5
        strs = self.dataset[index].strip().split("-")
        imgdata = np.array(Image.open(os.path.join(self.path, self.dataset[index])), dtype=np.float32)/255.

        label = np.zeros([6], dtype=np.float32)
        if strs[0] == "bainiangzi":
            label[0] = 1
        elif strs[0] == "xuxian":
            label[1] = 1
        elif strs[0] == "xiaoqing":
            label[2] = 1
        elif strs[0] == "jiejie":
            label[3] = 1
        elif strs[0] == "ligongpu":
            label[4] = 1
        elif strs[0] == "fahai":
            label[5] = 1
        return imgdata, label

    def __len__(self):
        return len(self.dataset)

    def get_data(self, mydata):
        dataloader = data.DataLoader(mydata, batch_size=10, shuffle=True)
        iters = iter(dataloader)
        xs, ys = iters.next()
        return xs.data.numpy(), ys.data.numpy()
#测试代码
if __name__ == '__main__':
    mydata = RecDataSet(r"C:\Users\liev\Desktop\data\facedata\okfaces")
    xs, ys = mydata.get_data(mydata)
    print(type(xs))
    print(xs)
    print(ys)
