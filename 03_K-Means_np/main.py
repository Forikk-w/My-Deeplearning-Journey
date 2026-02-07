from utils03.data_generator import generate_points
from models.k_means import K_Means
def run() :
    '''
        使用K-Means算法实现双特征三聚类问题
    '''
    #三聚类
    k = 3
    #获取点集
    points = generate_points(k = k)

    #定义K-means类
    iterations = 1000
    km_algorithm = K_Means(k = k, points = points, iterations = iterations)

    km_algorithm.begin_algorithm()
    print(f"中心点为{km_algorithm.get_center_points()}")


if __name__ == '__main__':
    print("03 K-Means by numpy")
    print(f"{'-' * 25}")

    run()