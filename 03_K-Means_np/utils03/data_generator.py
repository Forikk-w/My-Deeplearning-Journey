import numpy as np
from utils03.data_vision import visualize_data
def generate_points(k = 3) :
    '''
        生成（1，1） （3，10） （10，2）附近的三个簇 分别20个
    '''
    cluster = np.array([[1,1],[3,10],[10,2]])
    points = np.array([]).reshape(-1,2)
    for i in range(20) :
        points = np.concatenate([points,cluster],axis = 0)

    #添加自然噪声
    points = points + np.random.normal(0,1,points.shape)
    #打乱索引
    rs = np.random.RandomState(22)
    indexs = np.arange(len(points))
    rs.shuffle(indexs)
    points = points[indexs]

    return points



if __name__ == '__main__':

    visualize_data(generate_points(k = 3))