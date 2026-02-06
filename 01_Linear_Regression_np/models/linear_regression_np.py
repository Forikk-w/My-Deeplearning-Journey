import numpy as np
def linear_regression(x,y,degree,alpha,iterations) :
    '''
        将自变量化为矩阵 ：
            e.g. degree = 2
            X = [[...] [... ][...]]
                第i个[...]中存储 x ^ i的值
            m_theta.T  点乘 X = [_y1,_y2,...] 预测值向量
    '''
    X = np.ones((len(x),1),dtype=int)

    for i in range(1,degree + 1) :
        X = np.concatenate([X,x ** i] ,axis = 1)

    #初始化参数 列向量
    m_theta = np.zeros((degree + 1,1))

    for i in range(iterations) :
        m_theta = gradient_descent(m_theta,X,y,alpha)


    return m_theta

#批量梯度下降
def gradient_descent(t,X,y,alpha) :

    return t - alpha * (np.dot(np.dot(X.T,X),t) - np.dot(X.T,y))


if __name__ == '__main__':
    x = np.array([1,2,3,4,5]).reshape(-1,1)
    y = np.array([2,4,6,8,10]).reshape(-1,1)
    t = linear_regression(x,y,1,0.01,100)
    print(t)
