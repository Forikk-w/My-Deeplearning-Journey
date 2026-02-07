import numpy as np
from utils.sigmoid_function import sigmoid
def logical_regression(x,y,alpha,iterations) :
    '''
        x.shape = (len(x),5)
        x_features = 5 (包含一个常数项)
    '''
    m = len(x)
    #初始化参数
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    #迭代
    for i in range(iterations) :
        theta = gradient_descent(x, y, theta, alpha, m)

    return theta

def gradient_descent(x, y, theta, alpha, m) :
    #预测概率矩阵 (m,1)
    h = sigmoid(np.dot(x,theta))

    theta = theta - alpha * np.dot(x.T, h - y) / m
    return theta

if __name__ == '__main__':

    pass