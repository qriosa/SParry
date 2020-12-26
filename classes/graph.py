import numpy as np

from utils.settings import INF
from utils.debugger import Logger

# set logging
logger = Logger(__name__)

class Graph(object):
    """
    function: 
        a graph class.

    parameters:
        filename: str, must, the graph data file. (more info please see the developer documentation).
        directed: bool, the graph is directed ot not.
    
    attributes:
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        CSR: tuple, a 3-tuple of integers as (V, E, W) about the CSR of graph data. (more info please see the developer documentation).
        src, des, w: tuple, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
        MAXW: int, the max weight of the edges.
        MINW: int, the min weight of the edges.
        MAXD: int, the max degree(In degree and Out degree) of all the vertices.
        MAXU: int, one of the vertices with the max degree.
        MIND: int, the min degree(In degree and Out degree) of all the vertices.
        MINU: int, one of the vertices with the min degree.
        degree: list, save the degree of each vertex.
        msg: str, the message about the read func.
        filename: str, the file name of the graph.
    
    method:
        read: read the graph from file.
        reshape: convert data to numpy.int32.

    return: 
        class, Result object. (see the 'SPoon/classes/graph.py/Graph') 
    """

    def __init__(self):

        logger.info("set a new Graph Object.")

        # 预定义变量 图中的点和边
        self.n = -1
        self.m = -1

        # 图是否有方向
        self.directed = "Unknown"

        # 归一化为 graph
        # CSR 图数据
        # [src, des, w] 里面的每个边都将按照单向边来处理，并不会自己翻倍
        self.graph = None

        # 图数据类型 采用的算法 默认 dij
        self.mathod = "dij"
        self.delta = 3 # 若是 delta 则默认 delta 是 3

        # 最大边权
        self.MAXW = -1 
        # 最小边权
        self.MINW = INF
        # 最大度(仅有出度 因为无向图需要用两个有向边表示)
        self.MAXD = -1 
        # 最大度的点(之一)
        self.MAXU = -1 
        # 最小度
        self.MIND = INF 
        # 最小度的点(之一)
        self.MINU = -1 

        # 各个点的度
        self.degree = []


        # 打印的提示信息
        self.msg = "Welcome to use SPoon.\nIf you want see the detail of the Graph, you can set the parameter 'detail' of method 'read' as 'True'."
        
    def setmsg(self):
        """
        function: 
            set msg.

        parameters: 
            None, but 'self'.

        return:
            None, no return.        
        """
        
        self.msg = f"""
[+] the number of vertices in the Graph:\tn = {self.n}, 
[+] the number of edges in the Graph:\t\tm = {self.m}, 
[+] the max edge weight in the Graph:\t\tMAXW = {self.MAXW}, 
[+] the min edge weight in the Graph:\t\tMINW = {self.MINW}, 
[+] the max out degree in the Graph:\t\tdegree({self.MAXU}) = {self.MAXD}, 
[+] the min out degree in the Graph:\t\tdegree({self.MINU}) = {self.MIND}, 
[+] the average out degree of the Graph:\tavgDegree = {self.m/self.n},
[+] the directed of the Graph:\t\t\tdirected = {self.directed}, 
[+] the method of the Graph:\t\t\tmethod = {self.method}.
"""

    def reshape(self):
        
        """
        function: 
            convert data to numpy.int32.

        parameters: 
            None, but 'self'.

        return:
            None, no return.
        """

        logger.info(f"converting the graph to numpy.int32...")

        self.n = np.int32(self.n) # 结点数量
        self.m = np.int32(self.m) # 边的数量

        # graph
        # 不论是 CSR 还是 edgeSet 都是三元组
        self.graph[0] = np.copy(self.graph[0]).astype(np.int32)
        self.graph[1] = np.copy(self.graph[1]).astype(np.int32)
        self.graph[2] = np.copy(self.graph[2]).astype(np.int32)

        # delta
        self.delta = np.int32(self.delta)

        self.MAXW = np.int32(self.MAXW) # 最大边权
        self.MINW = np.int32(self.MINW) # 最小边权
        self.MAXD = np.int32(self.MAXD) # 最大度
        self.MAXU = np.int32(self.MAXU) # 最大度的点(之一)
        self.MIND = np.int32(self.MIND) # 最小度
        self.MINU = np.int32(self.MINU) # 最小度的点(之一)

