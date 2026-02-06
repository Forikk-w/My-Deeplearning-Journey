import numpy as np
#from data_vision import data_vision as vision
def generate_data(degree = 2,random_seed = 12):
    '''
        生成训练集和测试集 （随机种子确定）
        - 训练集：x_train, y_train
        - 测试集：x_test, y_test

        生成数据模型 ： e.g.  多项式degree = 2 : y = k2 * x^2 + k1 * x + k0

    '''
    np.random.seed(random_seed)

    x_train = np.random.randint(-50,50,100).reshape(-1,1)  #生成列向量 范围：-50 ~ 50 ,int
    x_test = np.random.randint(-50,50,50).reshape(-1,1)

    #生成目标模型的系数 在其基础上得到训练集
    k = np.array([],dtype = int)
    for i in range(degree + 1) :
        k = np.append(k,np.random.randint(-10,11))
        y_train = k[-1] * x_train ** i  + np.random.normal(0,100,(100,1))
        y_test = k[-1] * x_test ** i

    #print(k)
    return x_train, y_train, x_test, y_test, k.T






if __name__ == '__main__':

    d_x,d_y,t_x,t_y, k= generate_data()
    print("Data generated successfully!")

  #  vision(t_x, t_y,[-60,61],[-10000,10000],True)