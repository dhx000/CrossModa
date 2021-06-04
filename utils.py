import os
import numpy as np
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

def get_file(file_path,txt_name):
    path_list=os.listdir(file_path)
    print(path_list)
    file_object=open(txt_name,'w')
    for line in path_list:
        file_object.write(line+'\n')
    file_object.close()
def get_mu_sigma(file_path):
    # target domain : 194.01,236.61
    # source domain : 289.94,354.87
    # source domain :
    path_list=os.listdir(file_path)
    mean_list=[]
    var_list=[]
    num=0
    print(path_list)
    for line in path_list:
        num = num + 1
        if (num % 2 == 0 ): continue

        name=path+'\\'+line
        print(name)
        img=nib.load(name)
        mean=np.mean(img.dataobj)
        var=np.std(img.dataobj)
        mean_list.append(mean)
        var_list.append(var)
    mean_array=np.array(mean_list)
    var_array=np.array(var_list)
    return np.mean(mean_array),np.mean(var_array)

if __name__ == '__main__':
    path='D:\\Download\\cross\\source_training'
    mu,sigma=get_mu_sigma(path)

    print(mu,sigma)