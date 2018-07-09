from keras.models import Sequential
from keras.layers import Dense,Dropout
def mlpmodel():
    model = Sequential()
    model.add(Dense(units=1000,input_dim=784,kernel_initializer="normal",activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model