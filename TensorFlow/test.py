import numpy as np
from keras.models import model_from_json

NUMBER = 3
PATH = f"./modelli/{str(NUMBER)}/"

f = open(PATH + "model.json", "r")
jsonfile = f.read()
f.close()
model = model_from_json(jsonfile)
model.load_weights(PATH + "model.h5")
model.summary()


def convert_to_array(img):
    # im = cv2.imread(img)
    # img = Image.fromarray(img, 'RGB')
    # image = img.resize((50, 50))
    # return np.array(image)
    return img
def get_animal_name(label):
    if label==0:
        return "gatto"
    if label==1:
        return "cane"
    if label==2:
        return "mucca"
    if label==3:
        return "gallina"
    if label==4:
        return "persona"
def predict_animal(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print(animal)
    print("The predicted Animal is a "+animal+" with accuracy =    "+str(acc))


# predict_animal("test/cat1.jpg")
