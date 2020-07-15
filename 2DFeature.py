#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:26:40 2020

@author: liubing
"""
import scipy.io as sio
import numpy as np
import cv2
#from tqdm import tqdm
from sklearn.decomposition import PCA

img=sio.loadmat('Indian_pines_corrected.mat')
img=img['indian_pines_corrected']
m,n,b=img.shape
img=img.reshape(-1,b)

pca=PCA(n_components=3)     #加载PCA算法，设置降维后主成分数目为2
img=pca.fit_transform(img)#对样本进行降维
img=img.reshape(m,n,3)



kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7, 7))
kernel4 = cv2.getStructuringElement(cv2.MORPH_CROSS,(9, 9))

#img0=img[:,:,0]
#img1=img[:,:,1]
#img2=img[:,:,2]
m,n,b=img.shape
def get_feature(img):
    close4=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel4)
    close4=close4.reshape(m,n,1)
    close3=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3) 
    close3=close3.reshape(m,n,1)
    close2=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2) 
    close2=close2.reshape(m,n,1)
    close1=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    close1=close1.reshape(m,n,1)
    open1=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1) 
    open1=open1.reshape(m,n,1)
    open2=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2) 
    open2=open2.reshape(m,n,1)
    open3=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3) 
    open3=open3.reshape(m,n,1)
    open4=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel4)
    open4=open4.reshape(m,n,1)
    img=img.reshape(m,n,1)
    feature=np.concatenate((close4,close3,close2,close1,img,open1,open2,open3,open4),2)
    return feature

#f0=get_feature(img0)
#f1=get_feature(img1)
#f2=get_feature(img2)
#feature=np.concatenate((f0,f1,f2),2)
feature=[]
for i in range(b):
    temp=img[:,:,i]
    temp=np.squeeze(temp)
    f=get_feature(temp)
    if i==0:
        feature=f
    else:
        feature=np.concatenate((feature,f),2)

v_min=feature.min()
v_max=feature.max()
feature=(feature-v_min)/(v_max-v_min)

gt=sio.loadmat('Indian_pines_gt.mat')
gt=gt['indian_pines_gt']



def Patch(data,height_index,width_index,PATCH_SIZE):
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    print(patch.shape)
    return patch

img = feature
[m,n,b]=img.shape
label_num=gt.max()
data=[]
label=[]
PATCH_SIZE=14


#padding the hyperspectral images
img_temp=np.zeros((m+2*PATCH_SIZE,n+2*PATCH_SIZE,b),dtype=np.float32)
img_temp[PATCH_SIZE:(m+PATCH_SIZE),PATCH_SIZE:(n+PATCH_SIZE),:]=img[:,:,:]

for i in range(PATCH_SIZE):
    img_temp[i,:,:]=img_temp[2*PATCH_SIZE-i-1,:,:]
    img_temp[m+PATCH_SIZE+i,:,:]=img_temp[m+i-1,:,:]

for i in range(PATCH_SIZE):
    img_temp[:,i,:]=img_temp[:,2*PATCH_SIZE-i-1,:]
    img_temp[:,n+PATCH_SIZE+i,:]=img_temp[:,n+i-1,:]

img=img_temp


gt_temp=np.zeros((m+2*PATCH_SIZE,n+2*PATCH_SIZE),dtype=np.int8)
gt_temp[PATCH_SIZE:(m+PATCH_SIZE),PATCH_SIZE:(n+PATCH_SIZE)]=gt[:,:]
gt=gt_temp

[m,n,b]=img.shape

            
gt_index=[]
for i in range(PATCH_SIZE,m-PATCH_SIZE):
    for j in range(PATCH_SIZE,n-PATCH_SIZE):
        if gt[i,j]==0:
            continue
        else:
            temp_data=Patch(img,i,j,PATCH_SIZE)
            temp_label=np.zeros((1,label_num),dtype=np.int8)
            temp_label[0,gt[i,j]-1]=1
            data.append(temp_data)
            label.append(temp_label)
            gt_index.append((i-PATCH_SIZE)*145+j-PATCH_SIZE)
            
import pickle
f=open('gt_IP.data','wb')
pickle.dump(gt_index, f)
f.close()
            
data=np.array(data)
data=np.squeeze(data)
data=data.reshape(-1,28,28,27)
data=data.swapaxes(2,3)
data=data.swapaxes(1,2)
data=data.reshape(-1,28*28*27)
label=np.array(label)
label=np.squeeze(label)
import h5py
f=h5py.File('IP28-28-27.h5','w')
f['data']=data
f['label']=label
f.close()

'''
f=h5py.File('IP_EMP.h5','w')
f['data']=feature
f.close()
'''
