import numpy as np
class K_Means :

    def __init__(self, k, points, iterations) :
        '''
        :param k: 聚类个数
        :param points: 数据集
        :param iteraions: 迭代次数
        '''
        self.k = k
        self.points = points

        self.iterations = iterations

    def begin_algorithm(self):
        # 初始化中心点
        self.center = self.initialize_center_points()
        #迭代更新中心点 即更新聚类
        for i in range(self.iterations) :
            self.update_center_points()


    def initialize_center_points(self):

        #取随机数据点为起始中心点
        indexs = np.random.permutation(len(self.points)) #随机索引
        return self.points[:self.k]

    def update_center_points(self):
        '''
            取出一个当前中心点target
            计算各点到target的距离 更新clusters[center points]

        '''
        #初始化假设都在第一个中心点的簇中
        self.dist = self.calculate_dist(self.center[0])
        self.clusters = np.full((len(self.points), 2), self.center[0]) #存储每个点对应的中心点

        #寻找中心点对应的簇
        for i in range(1,self.k) :
            dist = self.calculate_dist(self.center[i])
            indexs = np.where(dist < self.dist)[0]
            self.clusters[indexs] = self.center[i]
            self.dist[indexs] = dist[indexs]

        #更新中心点
        for i in range(self.k) :
            indexs = np.where(self.clusters == self.center[i])[0]
            self.center[i] = [np.mean(self.points[indexs][0], axis = 0), np.mean(self.points[indexs][1], axis = 0)]

    def calculate_dist(self, target):
        '''
            计算到points到target的距离dist
            如果dist >= pre_dist 更新clusters
        '''
        return  np.sum((self.points - target.reshape(1,-1)) ** 2 ,axis=1)  #返回行向量

    def get_center_points(self):

        return self.center

    def get_clusters(self):

        return self.clusters


if __name__ == '__main__':

    pass
