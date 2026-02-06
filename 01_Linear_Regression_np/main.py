from utils.data_generator import generate_data
from models.linear_regression_np import linear_regression
def run() :
    degree = int(input("请输入多项式复杂度:"))
    train_x, train_y, test_x, test_y, k = generate_data(degree = degree)
    print()

    #输出理想模型
    print(k[0],end = "")
    for i in range(1,len(k)) :
        print(f" + {k[i]} * x^{i} ",end = "")

    #开始训练
    alpha = 0.003 #学习率
    iterations = 100 # 迭代次数
    trained_k = linear_regression(train_x,train_y, degree,alpha, iterations)

    # 输出训练模型
    print(trained_k[0], end="")
    for i in range(1, len(trained_k)):
        print(f" + {trained_k[i]} * x^{i} ", end="")


if __name__ == '__main__':
    print("01 Linear Regression by numpy")
    print(f"{'-'*25}\n")

    run()


