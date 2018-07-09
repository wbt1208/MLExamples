from keras.datasets import cifar10
from keras.utils import np_utils
from  CnnforCifar import plotimg
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
# print(x_train.shape)     (50000, 32, 32, 3)
# print(y_train.shape)     (50000, 1)
# print(x_test.shape)      (10000, 32, 32, 3)
# print(y_test.shape)    2  (10000, 1)
#plotimg.show(x_train[3])
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
y_test_onehot = np_utils.to_categorical(y_test)
y_train_onehot = np_utils.to_categorical(y_train)
from CnnforCifar.cnn_model_for_cifar import get_model
model = get_model()
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train_onehot,batch_size=128,epochs=10,verbose=2,validation_split=0.2)
model.save_weights('cnn_model_for_cifar_2')
#model.load_weights('cnn_model_for_cifar_1')
score = model.evaluate(x_test,y_test_onehot,verbose=1)
print(score)