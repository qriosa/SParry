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
        class, Result object. (see the 'sparry/classes/graph.py/Graph').
    """

    def __init__(self):

        logger.info("set a new Graph Object.")

        # n and m
        self.n = -1
        self.m = -1

        # the direct of the graph
        self.directed = "Unknown"

        # all named as graph, include CSR and edgeSet
        self.graph = None

        # algorithm
        self.mathod = "dij"
        self.delta = 3 # delta of delta-stepping

        # max edge weight
        self.MAXW = -1 
        # min edge weight
        self.MINW = INF
        # max out degree
        self.MAXD = -1 
        # one of the max degree vertices
        self.MAXU = -1 
        # min out degree
        self.MIND = INF 
        # one of the min degree vertices
        self.MINU = -1 

        # the degree of vertices
        self.degree = []


        # printable msg
        self.msg = "Welcome to use sparry.\nIf you want see the detail of the Graph, you can set the parameter 'detail' of method 'read' in 'pretreat.py' as 'True'."
        
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
[+] the number of vertices in the Graph:\t\tn = {self.n}, 
[+] the number of edges(directed) in the Graph:\t\tm = {self.m}, 
[+] the max edge weight in the Graph:\t\t\tMAXW = {self.MAXW}, 
[+] the min edge weight in the Graph:\t\t\tMINW = {self.MINW}, 
[+] the max out degree in the Graph:\t\t\tdegree({self.MAXU}) = {self.MAXD}, 
[+] the min out degree in the Graph:\t\t\tdegree({self.MINU}) = {self.MIND}, 
[+] the average out degree of the Graph:\t\tavgDegree = {self.m/self.n},
[+] the directed of the Graph:\t\t\t\tdirected = {self.directed}, 
[+] the method of the Graph:\t\t\t\tmethod = {self.method}.
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

        self.n = np.int32(self.n) # vertex number
        self.m = np.int32(self.m) # edge number

        # graph
        # no matter CSR or edgeSet
        self.graph[0] = np.copy(self.graph[0]).astype(np.int32)
        self.graph[1] = np.copy(self.graph[1]).astype(np.int32)
        self.graph[2] = np.copy(self.graph[2]).astype(np.int32)

        # delta
        self.delta = np.int32(self.delta)

        self.MAXW = np.int32(self.MAXW)
        self.MINW = np.int32(self.MINW) 
        self.MAXD = np.int32(self.MAXD) 
        self.MAXU = np.int32(self.MAXU) 
        self.MIND = np.int32(self.MIND) 
        self.MINU = np.int32(self.MINU) 

