import numpy as np
def linear_regression(x,y,degree,alpha,iterations) :
    '''
        将自变量化为矩阵 ：
            e.g. degree = 2
            X = [[...] [... ][...]]
                第i个[...]中存储 x ^ i的值

        归一化 ： 在复杂度较高的多项式情况下容易发生梯度爆炸
                需要平衡高次数自变量对梯度下降的影响

        返回后： 反归一化预测值 计算Loss
    '''
    X = np.ones((len(x),1),dtype=int)

    for i in range(1,degree + 1) :
        X = np.concatenate([X,x ** i] ,axis = 1)

    #归一化 传入矩阵 以及 每个列向量的均值与方差
    X_mean = np.mean(X,axis = 0)
    X_mean[0] = 0
    X_std = np.std(X,axis = 0)
    X_std[0] = 1 #对常数项特殊处理 不做归一化
    y_mean = np.mean(y,axis = 0)
    y_std = np.mean(y,axis = 0)

    X_norm = data_normalization(X,X_mean,X_std)
    y_norm = data_normalization(y,y_mean,y_std)

    #初始化参数 列向量
    m_theta = np.zeros((degree + 1,1))

    for i in range(iterations) :
        m_theta = gradient_descent(m_theta,X_norm,y_norm,alpha)

    #返回参数 训练集的平均值和标准差以反归一化
    return m_theta,X_mean,X_std,y_mean,y_std

#批量梯度下降
def gradient_descent(t,X,y,alpha) :
    #细节 集合律减少计算量

    return t - alpha * np.dot(X.T , np.dot(X,t) - y) / len(X)

#    return t - alpha * (np.dot(np.dot(X.T,X),t) - np.dot(X.T,y)) / len(X)

#Z-score normalization 归一化
def data_normalization(X,mean,std) :
    #常数项不做归一化

    return (X - mean) / (std + 1e-8)



if __name__ == '__main__':
    #测试用例模型表现良好
    x = np.array([1,2,3,4,5]).reshape(-1,1)
    y = np.array([3,5,7,9,11]).reshape(-1,1)
    t = linear_regression(x,y,1,0.001,1000)
    print(t)
