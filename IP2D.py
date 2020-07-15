#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:11:12 2020

@author: liubing
"""
import argparse
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
sys.path.insert(0, "lib")

from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json
import h5py

def get_label(gc, train_images):
    num = len(train_images)/1000
    y = np.zeros(1)
    for i in range(num):
        temp = gc.predict(train_images[i*1000:(i+1)*1000,:])
        y = np.concatenate((y,temp))
    temp = gc.predict(train_images[num*1000:,:])
    y = np.concatenate((y,temp))
    return y[1:]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config=load_json('IP.json')
    
    gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.

    f=h5py.File('IP28-28-27.h5','r')
    train_images=f['data'][:]
    train_labels=f['label'][:]
    f.close()
    #f=open('paviaU.data','rb')
    #train_images=pickle.load(f)
    #train_labels=pickle.load(f)
    #f.close()
    train_labels = np.argmax(train_labels,1)
  
    indices = np.arange(train_images.shape[0])
    shuffled_indices = np.random.permutation(indices)
    images = train_images[shuffled_indices]
    labels = train_labels[shuffled_indices]
    n_classes = labels.max() + 1
    i_labeled = []
    for c in range(n_classes):
        i = indices[labels==c][:5]##change sample number
        i_labeled += list(i)
    X_train = images[i_labeled]
    X_train = X_train.reshape(-1,27,28,28)
    y_train = labels[i_labeled]
    
    #y_train = np.argmax(y_train,1)
    #train_labels = np.argmax(train_labels,1)

    X_train_enc = gc.fit_transform(X_train, y_train)
    # X_enc is the concatenated predict_proba result of each estimators of the last layer of the GCForest model
    # X_enc.shape =
    #   (n_datas, n_estimators * n_classes): If cascade is provided
    #   (n_datas, n_estimators * n_classes, dimX, dimY): If only finegrained part is provided
    # You can also pass X_test, y_test to fit_transform method, then the accracy on test data will be logged when training.
    # X_train_enc, X_test_enc = gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test)
    # WARNING: if you set gc.set_keep_model_in_mem(True), you would have to use
    # gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test) to evaluate your model.
    train_images = train_images.reshape(-1,27,28,28)
    y_pred = get_label(gc, train_images)
    acc = accuracy_score(train_labels, y_pred)
    print(acc)
    
