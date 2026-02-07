import numpy as np


def sigmoid(data) :
    '''
        逻辑函数 转化为概率
    '''
    p_data = 1 / (1 + np.exp(-data))

    return p_data

if __name__ == '__main__':

    data = np.array([[0,0,1,2,3]])
    print(sigmoid(data))