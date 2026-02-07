import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

def visualize_data(points) :

    x = points[:,0]
    y = points[:,1]
    fig = plt.figure()
    plt.plot(x,y,marker = '.',linestyle = '')
    plt.show()


if __name__ == '__main__':
    pass