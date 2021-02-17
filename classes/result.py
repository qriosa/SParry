import numpy as np

from utils.debugger import Logger

logger = Logger(__name__)

class Result(object):
    """
    function: 
        to store the result of different algorithm. 

    parameters:
        dist: list, the shortest path distance answer for algorithm.
        timeCostNum: float, a float data of time cost of getting the answer, so it can use to calculate.
        timeCost: str, a str data of time cost of getting the answer.
        memoryCost: str, memory cost of getting the answer.
        graph: class Graph, must, the graph data that you want to get the shortest path.
            (more info please see the developer documentation).
    
    method:
        display: 
            show the detail of this calculation.
        drawPath: 
            draw the path from vertices to the sources.
        calcPath:  
            calc the path through the graph and dist.
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    def __init__(self, 
                dist = None, 
                timeCost = None, 
                memoryCost = None, 
                graph = None): # class Graph

        logger.info("class.Result initializing.")

        self.dist = np.array(dist).flatten() # dist
        self.path = None # path
        self.timeCostNum = timeCost
        self.timeCost = str(timeCost * 100000 // 100 / 1000) + ' sec' # time cost str(timeCost * 100000 // 100 / 1000) + ' sec'
        self.memoryCost = memoryCost # Memory overhead
        
        self.graph = graph # graph

    
    def display(self):
        """
        function: 
            show the detail of the graph, parameters and calc time.

        parameters:
            None, but 'self'.
        
        return: 
            str, the msg info.       
        """

        return f"{self.graph.msg}\n\n[+] calc the shortest path timeCost = {self.timeCost}"
    
    def drawPath(self):
        """
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.        
        """

        if self.path is None:
            self.calcPath()
        
        from utils.showPath import draw
        
        # only show one source vertex
        for i in range(self.dist.size):
            if self.dist[i] == 0:
                s = i % self.graph.n
                
        draw(path = self.path, s = s, graph = self.graph)

    def calcPath(self):
        """
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.
        """
   
        logger.info("turning to func calcPath.")

        if(self.dist is None):
            raise Exception("can not calc path without dist")
        
        if(self.graph is None):
            raise Exception("can not calc path without graph")

        # calc path with dist and graphic data
        if(self.graph.method == "edge"):
            self.calcPathFromEdgeSet()
        else:
            self.calcPathFromCSR()

    def calcPathFromCSR(self):
        """
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.        
        """

        V, E, W = self.graph.graph[0], self.graph.graph[1], self.graph.graph[2]
        n = self.graph.n
        self.path = np.full((self.dist.size, ), -1)
        sNum = self.dist.size // n # the number of source vertex

        for i in range(n):
            for j in range(V[i], V[i + 1]):
                for k in range(sNum):
                    kn = k * n
                    if(self.path[E[j] + kn] == -1 and self.dist[E[j] + kn] == self.dist[i + kn] + W[j]):
                        self.path[E[j] + kn] = i 
        
    def calcPathFromMatrix(self):
        """
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.         
        """

        raise Exception("there should not be a matrix graph data.")

        n = np.array(self.graph).shape[0]

        self.path = np.full((self.dist.size, ), -1)
        sNum = self.dist.size // n 

        for i in range(n):
            for j in range(n):
                for k in range(sNum):
                    kn = k * n
                    if(self.path[j + kn] == -1 and self.dist[j + kn] == self.dist[i + kn] + self.matrix[i][j]):
                        self.path[j + kn] = i 

    def calcPathFromEdgeSet(self):
        """
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.         
        """

        src, des, w = self.graph.graph[0], self.graph.graph[1], self.graph.graph[2]
        m = self.graph.m

        self.path = np.full((self.dist.size, ), -1)
        n = self.graph.n

        sNum = self.dist.size // n 

        for i in range(m):
            for k in range(sNum):
                kn = k * n
                if(self.path[des[i] + kn] == -1 and self.dist[des[i] + kn] == self.dist[src[i] + kn] + w[i]):
                    self.path[des[i] + kn] = src[i]       
