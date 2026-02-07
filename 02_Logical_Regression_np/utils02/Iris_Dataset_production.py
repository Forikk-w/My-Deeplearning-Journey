from sklearn.datasets import load_iris
import numpy as np
def produce_iris_dataset(train_size = 80) :
    '''
        train_size : 需要划分的训练集数量 (train_size < 100)
        Iris_Dataset : 150 in total
                    labels : 0  1  2
                    X.shape = (150,4) Y.shape = (150,)
        本次逻辑回归完成二分类任务 即取标签为 0 1的数据
    '''

    iris  = load_iris()
    #取出标签为0 1 的部分
    x_total = iris.data[:100]
    y_total = iris.target[:100].reshape(-1,1)

    #原数据集有顺序 现需打乱
    #打乱索引
    rs = np.random.RandomState(20)
    index = np.arange(len(x_total))
    rs.shuffle(index)
    #应用新索引
    x_total = x_total[index]
    y_total = y_total[index]

    x_train = x_total[:train_size]
    y_train = y_total[:train_size]
    x_test = x_total[train_size:]
    y_test = y_total[train_size:]

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':

    x_train, y_train, x_test, y_test = produce_iris_dataset()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    # print(x_train)
    print(y_train)
    print(y_test)