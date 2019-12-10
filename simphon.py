from PIL import Image
import numpy as np
import os
import glob
import re
import keras
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#定义函数：从数据集中读取图片数据并转化为特征矩阵，取前150张图片做测试集，其余做训练集
def read_img(location):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    label_name = []
    dirs = os.listdir(location)
    label = 0
    count = 0
    for i in dirs: #loop all directory
        print(i)
        n = 0
        label_name.append(i) #save folder name in var label_name
        x_s = 256
        y_s = 256
        for pic in glob.glob(location+'\\'+i+'\*.jpg'):
            im = Image.open(pic) #open data
            im = im.resize((x_s, y_s), Image.ANTIALIAS)
            im = np.array(im) #store im as numpy array
            if(im.shape[0]==256 and im.shape[1]==256):
                r = im[:,:,0]
                g = im[:,:,1]
                b = im[:,:,2]
                if(n<100):
                    x_test.append([r,g,b]) #save in x_test
                    y_test.append([label]) #save in y_test
                else : #remaining data set as training data
                    x_train.append([r,g,b]) #save in x_train
                    y_train.append([label]) #save in y_train
                n = n + 1
                count = count + 1
        label = label + 1 #increment label
    print(label_name)
    print(dirs)
    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
#通过定义的函数读数据，并定义训练数据，训练标签，测试数据，测试标签
path='D:\\simpsons_dataset'
img_rows = 256 #num of image height
img_cols = 256 #num of image width
num_class = 7 #num of classes/labels
x_train,y_train,x_test,y_test = read_img(path)
#对训练数据和测试数据的值做线性变化，提高机器学习的速率，并将标签转化为向量，以便用交叉熵计算loss值：
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)
#输出训练训练特征矩阵、训练标签向量、测试特征矩阵、测试标签向量的维度：
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#定义CNN神经网络模型：
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_class, activation='softmax'))
#编译模型：用交叉熵作为损失值，随机梯度下降作为优化器，预测的准确率用以定义模型的好坏。
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
#训一次模型并保存：模型一个批次处理32个样本，迭代1次，用测试集数据做验证。
model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, validation_data=(x_test, y_test))
model.save('Simpson.h5')
#循环进行模型训练，每一次循环迭代一次训练，保存并读取模型，循环十次
for i in range(0,20):
    print('The '+str(i)+' th Iteration')
    model=load_model('Simpson.h5')
    model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, validation_data=(x_test, y_test))
    model.save('Simpson.h5')
    K.clear_session()