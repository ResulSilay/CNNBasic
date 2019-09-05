import os
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from random import shuffle
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import seaborn as sns

TEST_DIR = './images/test/Cat'
DIR = './images/train/Cat'
IMG_SIZE = 300

def load_training_data():
    DIR = './images/train/Cat'
    train_data = []
    for img in os.listdir(DIR):
        label = label_img("cat")
        path = os.path.join(DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            train_data.append([np.array(img), label])

    DIR = './images/train/Dog'            
    for img in os.listdir(DIR):
        label = label_img("dog")
        path = os.path.join(DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            train_data.append([np.array(img), label])
            
    shuffle(train_data)
    return train_data

def load_test_data():
    test_data = []
    TEST_DIR = './images/test/Cat'
    for img in os.listdir(TEST_DIR):
        label = label_img("cat")
        path = os.path.join(TEST_DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            test_data.append([np.array(img), label])
     
    TEST_DIR = './images/test/Dog'
    for img in os.listdir(TEST_DIR):
        label = label_img("dog")
        path = os.path.join(TEST_DIR, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            test_data.append([np.array(img), label])      
            
    shuffle(test_data)
    return test_data


def label_img(name):
    word_label = name
    if word_label == 'dog': return np.array([1, 0])
    elif word_label == 'cat' : return np.array([0, 1])


from keras.preprocessing import image
import pydot
from keras.utils import plot_model
def IMAGE(model,img):
    img = image.load_img(path=img,grayscale=True,target_size=(IMG_SIZE,IMG_SIZE,1))
    img = image.img_to_array(img)
    
    img = np.expand_dims(img, axis=0)
    img_pred = model.predict(img)
    print("Pred*: ",img_pred)
    
    img_class = model.predict_classes(img)
    print("Pred: ",img_class)
    prediction = img_class[0]
    
    tip = 'Köpek' if int(prediction) == 0 else 'Kedi'
    
    img = img.reshape((IMG_SIZE,IMG_SIZE))
    plt.imshow(img)
    plt.title(tip)
    plt.show()  
    
    #print("Datas:",img_pred)
    print("Class: ",prediction)  
    print("Type: ", tip)  
    print("---------------------")

def LOAD():
    model = load_model('model.h5')
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    test_data = load_test_data()    

    #plt.imshow(test_data[10][0], cmap = 'gist_gray')    
    testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    testLabels = np.array([i[1] for i in test_data])
        
    pred = model.predict(testImages)
    loss, acc = model.evaluate(testImages, testLabels, verbose = 1)

    IMAGE(model,'test_d.jpg')
    IMAGE(model,'test_c.jpg')
    
    #y_pred = np.around(testLabels)
    #y_pred = (y_pred > 0.5) 
    #F1=f1_score(testImages, y_pred,average='weighted')
    #print("F1: "+str(F1))
    print("Acc: "+ str(acc * 100))
    
    #print(classification_report(testLabels,pred))
    cm = confusion_matrix(testLabels.argmax(axis=1),pred.argmax(axis=1))
    print("CM: ")
    print(cm)

    df_cm = pd.DataFrame(cm, index = [i for i in "01"], columns = [i for i in "01"])
    plt.figure(figsize = (2,2))
    sns.heatmap(df_cm, annot=True)
    plt.savefig("cm_heat_mp.png")

    keras.utils.vis_utils.pydot = pydot
    plot_model(model, to_file='model.png')
    plot_model(model, to_file='model.png')
    

def CNN():
    train_data = load_training_data()
    #plt.imshow(train_data[43][0], cmap = 'gist_gray')
    trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    trainLabels = np.array([i[1] for i in train_data])
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])    
    hist = model.fit(trainImages, trainLabels, batch_size = 50, epochs = 3, verbose = 1)
    model.save("model.h5")
    
    plt.plot(hist.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("epoch_loss.png")
    plt.show()
    

#CNN()
LOAD()