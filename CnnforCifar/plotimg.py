import matplotlib.pyplot as plt
def show(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image)
    plt.show()