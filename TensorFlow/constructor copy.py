from PIL import Image
import numpy as np
import os
import cv2


data=[]
labels=[]
cats=os.listdir("../raw-img/gatto")
for cat in cats:
    imag=cv2.imread("../raw-img/gatto/"+cat)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

dogs=os.listdir("../raw-img/cane")
for dog in dogs:
    imag=cv2.imread("../raw-img/cane/"+dog)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)
    
cows=os.listdir("../raw-img/mucca")
for cow in cows:
    imag=cv2.imread("../raw-img/mucca/"+cow)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

chickens=os.listdir("../raw-img/gallina")
for chicken in chickens:
    imag=cv2.imread("../raw-img/gallina/"+chicken)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

people=os.listdir("../raw-img/persone/")
for person in people:
    _p=os.listdir(f"../raw-img/persone/{person}")
    for p in _p:
        imag=cv2.imread(f"../raw-img/persone/{person}/"+p)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))
        data.append(np.array(resized_image))
        labels.append(0)

rooms=os.listdir("../raw-img/stanze")
for room in rooms:    
    _r=os.listdir(f"../raw-img/stanze/{room}")
    for r in _r:
        imag=cv2.imread(f"../raw-img/stanze/{room}/"+r)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((50, 50))
        data.append(np.array(resized_image))
        labels.append(1)


animals=np.array(data)
labels=np.array(labels)

np.save("owo",animals)
np.save("owo_labels",labels)