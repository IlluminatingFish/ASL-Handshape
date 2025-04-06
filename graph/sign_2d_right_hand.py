import sys

sys.path.extend(['../'])
from graph import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
#inward_ori_index = [(5, 6), (5, 7),(5,9),(5,11),(5,13),(7,8),(9,10),(11,12),(13,14)]
inward_ori_index = [(5, 6), (6, 7),(7,8),(8,9),(10,11),(11,12),(12,13),(14,15),(15,16),(16,17),(18,19),(19,20),(20,21),(22,23),(23,24),(24,25),(5,10),(5,14),(5,18),(5,22)]
inward = [(i - 5, j - 5) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
