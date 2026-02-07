import numpy as np
def normalize_data(x_train, x_mean, x_std, y_train = [], y_mean = 0, y_std = -1) :

    x_train_norm = (x_train - x_mean) / (x_std + 1e-8)
    if(y_std != -1) : #区分是否只传入一个需要归一化的数据集
        y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
    else : y_train_norm = np.array([])
    x_train_norm = np.concatenate([np.ones(len(x_train_norm)).reshape(-1,1), x_train_norm], axis = 1)

    return x_train_norm,y_train_norm

def denormalize_data(data, data_mean, data_std) :

    data_denormalized = data * data_std + data_mean

    return data_denormalized

if __name__ == '__main__':

    x = np.arange(10).reshape(-1,1)
    y = np.arange(10,20).reshape(-1,1)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    x,y= normalize_data(x, x_mean, x_std, y, y_mean, y_std)
    print(x)
    print(y)