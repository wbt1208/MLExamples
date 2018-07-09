import matplotlib.pyplot  as plt
class PlotImage():
    def __init__(self,image):
        self.image = image
        self.num = len(image)
    def show(self):
        fig = plt.gcf()
        fig.set_size_inches(2,2)
        plt.imshow(self.image,cmap = 'binary')
        plt.show()
    def plot_images_labels_predicts(self,labels,predicts):
        fig = plt.gcf()
        fig.set_size_inches(12,14)
        if self.num >25 :self.num=25
        for i in range(0,self.num):
            ax = plt.subplot(5,5,i+1)
            ax.imshow(self.image[i],cmap='binary')
            title  = "label=" + str(labels[i])
            if len(predicts)>0:
                title+=",predict="+str(predicts[i])
            ax.set_title(title)
        plt.show()
class ShowHistory():
    def __init__(self,train_history,train,vallidation):
        self.train_history = train_history
        self.train = train
        self.vallidation = vallidation
    def show(self):
        plt.plot(self.train_history.history[self.train])
        plt.plot(self.train_history.history[self.vallidation])
        plt.title("train history")
        plt.ylabel(self.train)
        plt.xlabel('Epoch')
        plt.show()



