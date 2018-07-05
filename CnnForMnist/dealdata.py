from keras.datasets import mnist
from keras.utils import np_utils#对数据预处理的模块
import numpy as np
#load data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#
x_train_4d = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test_4d = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

x_train_4d = x_train_4d/255
x_test_4d = x_test_4d/255
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onthot =np_utils.to_categorical(y_test)

#建立模型
from CnnForMnist.cnn_models import get_cnnmodel
model = get_cnnmodel()
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x=x_train_4d,y=y_train_onehot,batch_size=300,epochs=10,verbose=2,validation_split=0.2)