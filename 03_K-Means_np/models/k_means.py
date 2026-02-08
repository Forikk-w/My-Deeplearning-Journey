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
        rs = np.random.RandomState(45)
        indexs = rs.permutation(len(self.points)) #随机索引
        return self.points[indexs[:self.k]]

    def update_center_points(self):
        '''
            取出一个当前中心点target
            计算各点到target的距离 更新clusters[center points]

        '''

        #    self.points : (60, 2)   self.center : (3, 2)   ---> dist : (60, 3)   ---> center_points  : (60, )
        # points : (60, 2) ---> (60, 1, 2)     - center : (3, 2)
        # 广播 : diff : (60, 3, 2) points的第二个维度扩张 重复第一个维度(, 2)   center的第三个维度扩张 重复第二个维度(3, 2)
        diff = self.points[:, np.newaxis] - self.center
        #求平方和 消除坐标轴axis = 2 此时 dists : (60,3)
        dists = np.sum(diff ** 2, axis = 2)

        #获取对应距离最小值的索引 labels : (60, )
        self.labels = np.argmin(dists, axis = 1)

        #更新中心点
        for i in range(self.k) :
            clusters = self.points[self.labels == i]
            #确保存在这个聚类 避免后续divided by zero
            if len(clusters) > 0 :
                self.center[i] = np.mean(clusters, axis = 0)





        '''
        #初始化假设都在第一个中心点的簇中
        self.dist = self.calculate_dist(self.center[0])
        self.clusters = np.full((len(self.points), 1), self.center[0]) #存储每个点对应的中心点索引

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
            
            
        def calculate_dist(self, target_index):
    
           # 计算到points到target的距离dist
           # 如果dist >= pre_dist 更新clusters
            
        return  np.sum((self.points - self.points[target_index].reshape(1,-1)) ** 2 ,axis=1)  #返回行向量
            
        '''



    def get_center_points(self):

        return self.center

    def get_clusters(self):

        return self.labels

    #
    # def get_clusters(self):
    #
    #     return self.clusters


if __name__ == '__main__':

    pass
