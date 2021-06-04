from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import nibabel as nib
import numpy as np
import random
import skimage.transform as skTrans
class CrossDataSet(Dataset):
    def __init__(self, root, txt_name, train_flag=False, transformer=None):
        self.images_list = self.get_file(txt_name)
        self.train_flag=train_flag
        self.root = root
        self.transform = transformer
        self.slice = 16
    def get_file(self,txt_name):
        file=open(txt_name,'r')
        list=file.readlines()
        return list
    def __getitem__(self, index):

        if (self.train_flag==True): index=index*2
        file_name=self.root+'/'+self.images_list[index].strip()
        img = nib.load(file_name).get_fdata()
        spos=img.shape[-1]
        r=random.randint(0,spos-self.slice-1)
        img=img[:,:,r:r+self.slice]
        #img = img.
        if (self.train_flag):
            label_name = self.root + '/' + self.images_list[index+1].strip()
            label = nib.load(label_name).get_fdata()
            label = label[:,:,r:r+self.slice]
        if (self.transform!=None): img = self.transform(img)
        img = skTrans.resize(img, (img.shape[0]//4, img.shape[1]//4, img.shape[2]), order=1, preserve_range=True).astype(np.float32)
        label = skTrans.resize(label , (label.shape[0] // 4, label.shape[1] // 4, label.shape[2]), order=1, preserve_range=True)
        label[label>0]=1
        label=label.astype(np.int64)
        label = np.expand_dims(label, axis=0)
        img = np.expand_dims(img, axis=0)

        if (self.train_flag):
            return img,label,file_name,label_name
        else:
            return img,file_name

    def __len__(self):
        if (self.train_flag):
            return len(self.images_list)//2
        else:
            return len(self.images_list)
if __name__ == '__main__':
    dataSet=CrossDataSet(root='D:\\Download\\cross\\source_training',txt_name='./data/train_source.txt',train_flag=True)
    img,label,file_name,label_name=dataSet.__getitem__(0)
    print(file_name)
    print(label_name)
    print(img.shape)
    print(type(img))
    print(img.dtype)
    print(label.shape)
    print(type(label))
    #print(label.dtype)
    #label[label>0]=1.
    print(np.sum(label==0))