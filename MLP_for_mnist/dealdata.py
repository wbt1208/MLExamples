from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd
from MLP_for_mnist.plotimage import PlotImage
(x_train,y_train),(x_test,y_test) = mnist.load_data()#加载数据
#数据处理
#对训练集归一化处理
x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
#print(x_train.shape,x_test.shape)
#对label,onehot处理
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)
#print(y_train_onehot[:5])
#搭建训练模型
from MLP_for_mnist.mlp_models import mlpmodel
model = mlpmodel()
print(model.summary())
train_history = model.fit(x=x_train,y=y_train_onehot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)
score = model.evaluate(x_test,y_test_onehot)
predict = model.predict_classes(x_test)
print(predict)
print(score)
img = PlotImage((x_test.reshape(10000,28,28)*255)[:20])
img.plot_images_labels_predicts(labels=y_test[:20],predicts=predict[:20])
pd.crosstab(y_test,predict,rownames=['label'],colnames=['predict'])
# history = ShowHistory(train_history,'acc','val_acc')
# history.show()





# x = []
# y = []
# for i in range(28):
#     for j in range(28):
#         if j in [12,15,16 ]:
#             y.append(255)
#         else:
#             y.append(0)
#     x.append(y)
#     y=[]
# img = PlotImage(x)
# img.show()
# img = PlotImage(x_train[0:25])
# lables = y_train[0:25]
# img.plot_images_labels_predicts(labels=lables,predicts=[])
