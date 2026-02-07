from utils.Iris_Dataset_production import produce_iris_dataset
from utils.data_normalization import normalize_data,denormalize_data
from models.logical_regression_np import logical_regression
from utils.sigmoid_function import sigmoid
import numpy as np
def run() :
    '''
        基于鸢尾花数据集的二分类逻辑回归实现
        1.通过sklearn获取数据集
        2.数据预处理
        3.模型训练
        4.评估模型
    '''
    #生成数据集
    x_train,y_train,x_test,y_test = produce_iris_dataset(train_size=80)
    #获取训练集均值和标准差
    x_mean = np.mean(x_train,axis = 0)
    x_std = np.std(x_train,axis = 0)

    #数据预处理 归一化
    x_train_norm, temp=  normalize_data(x_train, x_mean, x_std)

    #训练
    alpha = 0.03
    iterations = 1000
    theta = logical_regression(x_train_norm, y_train,alpha,iterations)

    #归一化测试集
    x_test_norm, temp = normalize_data(x_test, x_mean, x_std)

    y_predicted = np.dot(x_test_norm, theta)
    #转化为概率
    y_predicted = sigmoid(y_predicted)
    prediction = (y_predicted >= 0.5).astype(int)

    #验证正确率
    accuracy = np.mean(prediction == y_test) * 100
    print(f"模型准确率 : {accuracy:.2f}%\n")




if __name__ == '__main__':
    print("02 Logical Regression by numpy")
    print(f"{'-'*25}")

    run()