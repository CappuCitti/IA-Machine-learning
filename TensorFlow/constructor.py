from PIL import Image
import numpy as np
import os
import cv2


data=[]
labels=[]
cats=os.listdir("raw-img/gatto")
for cat in cats:
    imag=cv2.imread("raw-img/gatto/"+cat)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

dogs=os.listdir("raw-img/cane")
for dog in dogs:
    imag=cv2.imread("raw-img/cane/"+dog)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)
    
cows=os.listdir("raw-img/mucca")
for cow in cows:
    imag=cv2.imread("raw-img/mucca/"+cow)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)

chickens=os.listdir("raw-img/gallina")
for chicken in chickens:
    imag=cv2.imread("raw-img/gallina/"+chicken)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)


animals=np.array(data)
labels=np.array(labels)

np.save("animals",animals)
np.save("labels",labels)