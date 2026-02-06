import numpy as np
def compute_loss(theta,X_test,X_mean,X_std,y_mean,y_std,test_y) :
    '''
        反归一化计算loss
        本质相当于转换坐标系  :坐标中心点为均值 坐标轴按标准差比例收缩
    '''
    m = len(X_test)
    #归一化
    X_test = (X_test - X_mean) / (X_std + 1e-8)
    y_predicted_norm = np.dot(X_test,theta)
    #反归一化
    y_predicted = y_predicted_norm * (y_std + 1e-8) + y_mean

    singe_loss = (y_predicted - test_y) ** 2
    loss = np.sum(singe_loss)

    return loss / m,y_predicted


if __name__ == '__main__':
    pass