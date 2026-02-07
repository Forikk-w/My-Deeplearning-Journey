import numpy as np
def r2_computation(y_predicted,test_y,y_mean) :
    '''
        计算R^2决定系数 衡量模型质量
    '''
    #总离差平方和
    ss1 = np.sum((test_y - y_mean) ** 2)
    #残差平方和
    ss2 = np.sum((test_y - y_predicted) ** 2)

    return 1 - ss2 / ss1



if __name__ == '__main__':
    pass