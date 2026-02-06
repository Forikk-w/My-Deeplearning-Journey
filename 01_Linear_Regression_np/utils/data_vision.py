import matplotlib.pyplot as plt
import numpy as np
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
#默认传入列向量
def data_vision(x_vector, y_vector, x_range ,y_range ,if_scatter) :
    '''
        x : 横坐标
        y : 纵坐标
        x_range ： x坐标轴范围
        y_range ： y坐标轴范围
        if_scatter : 是否绘制散点图
    '''

    #转为行向量 [[x1,x2,...]]  取第一项
    x = x_vector.reshape(1,-1) [0]
    y = y_vector.reshape(1,-1) [0]

    fig = plt.figure()
    if if_scatter : #是否绘制散点图
        plt.plot(x,y,linestyle = '',marker = '.') #使用plot画散点图一起渲染 效率更高
    else : plt.plot(x,y)

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()

if __name__ == '__main__':

    #e.g.
    x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
    y = np.array([101,99,98,1,2,3,4,100,0,9]).reshape(-1,1)
    data_vision(x,y,[0,10],[0,150],False)