from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,MaxPooling2D,Flatten
import keras

def get_model():
    #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1.1, patience=5, verbose=0, mode='auto',
    #                                 epsilon=0.0001, cooldown=0, min_lr=0)
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(32,32,3)))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(units=2500,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=1500,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10,activation='softmax'))
    return model