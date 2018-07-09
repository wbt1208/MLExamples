from keras.datasets import mnist
(x_train_image,y_train_image),\
    (x_test_image,y_test_image) = mnist.load_data()
print("x_train_shape: ",x_train_image.shape)
print("y_train_shape: ",y_train_image.shape)
print("x_test_shape: ",x_test_image.shape)
print("y_test_shape: ",y_test_image.shape)
'''
x_train_shape:  (60000, 28, 28)
y_train_shape:  (60000,)
x_test_shape:  (10000, 28, 28)
y_test_shape:  (10000,)

'''