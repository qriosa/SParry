class Result(object):
    """
    function: 
        to store the result of different algorithm.

    parameters:
        dist: list, the shortest path distance answer for algorithm.
        timeCostNum: float, a float data of time cost of getting the answer, so it can use to calculate.
        timeCost: str, a str data of time cost of getting the answer.
        memoryCost: str, memory cost of getting the answer.
    
    method:
        calcPath: calc the path through the graph and dist.
    
    return: Result object.
    """
    def __init__(self, dist = None, timeCost = None, memoryCost = None):
        # 关于 n 的值 之前说的点的编号 从 0 ~ n + 1 处理这个问题应该不是这里的事情
        self.dist = dist # 距离记录
        self.path = None # 路径记录
        self.timeCostNum = timeCost
        self.timeCost = str(timeCost * 100000 // 100 / 1000) + ' sec' # 时间花费 保留两位小数  str(timeCost * 100000 // 100 / 1000) + ' sec'
        self.memoryCost = memoryCost # 内存的开销
        
        # 待补充 更多关于图的特点
    
    def calcPath(self,CSR=None,matrix=None,edgeSet=None):
        """
        function: 
            to get the path.

        parameters:
            CSR: the CSR graph data. (more info please see the developer documentation).
            matrix: the matrix graph data.
            edgeSet: the edgeSet graph data.
        
        return: 
            None, no return.
        """
        if(self.dist == None):
            raise Exception("can not calc path without dist")
        # calc path with dist and graphic data
        if(CSR != None):
            pass
        elif(matrix != None):
            pass
        elif(edgeSet != None):
            src, des, w = edgeSet[0], edgeSet[1], edgeSet[2] 
        else:
            raise Exception("can not calc path only with dist")
