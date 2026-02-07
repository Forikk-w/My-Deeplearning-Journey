import numpy as np
from utils.data_generator import generate_data
from models.linear_regression_np import linear_regression
from utils.loss_computation import compute_loss
from utils.R2_determination import r2_computation
from utils.data_vision import data_vision
def run() :
    '''
        1.生成数据集
        2.归一化训练模型
        3.反归一化计算prediction , loss
    '''
    degree = int(input("请输入多项式复杂度:"))
    train_x, train_y, test_x, test_y, k = generate_data(degree = degree)
    print()

    #输出理想模型
    print(k[0],end = "")
    for i in range(1,len(k)) :
        print(f" + {k[i]} * x^{i} ",end = "")
    print()

    #开始训练
    alpha = 0.01 #学习率
    iterations = 2000# 迭代次数
    trained_k,X_mean,X_std,y_mean,y_std = linear_regression(train_x,train_y, degree,alpha, iterations)

    #将测试集转化为矩阵
    test_X = np.ones((len(test_x),1),dtype=int)

    for i in range(1,degree + 1) :
        test_X = np.concatenate([test_X,test_x ** i] ,axis = 1)

    #计算Loss
    Loss,y_predicted = compute_loss(trained_k,test_X,X_mean,X_std,y_mean,y_std,test_y)
    print(f"\nLoss : {Loss}\n")
    # print(test_y.reshape(1,-1)[0][:10],'\n',y_predicted.reshape(1,-1)[0][:10])

    #计算决定系数 R^2
    print(f"R^2 = {r2_computation(y_predicted,test_y,np.mean(test_y))}\n")

if __name__ == '__main__':
    print("01 Linear Regression by numpy")
    print(f"{'-'*25}")

    run()


