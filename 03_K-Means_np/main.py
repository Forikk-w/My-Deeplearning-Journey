from utils03.data_generator import generate_points
from utils03.data_vision import visualize_data
from models.k_means import K_Means
def run() :
    '''
        使用K-Means算法实现双特征三聚类问题
    '''
    #三聚类
    k = 3
    #获取点集
    points = generate_points(k = k)
    visualize_data(points)

    #定义K-means类
    iterations = 1000
    km_algorithm = K_Means(k = k, points = points, iterations = iterations)

    km_algorithm.begin_algorithm()
    #获取中心点
    print(f"中心点为{km_algorithm.get_center_points()}")

    #获取聚类标签
    labels = km_algorithm.get_clusters()
    visualize_data(points[labels == 0])
    visualize_data(points[labels == 1])
    visualize_data(points[labels == 2])

if __name__ == '__main__':
    print("03 K-Means by numpy")
    print(f"{'-' * 25}")

    run()