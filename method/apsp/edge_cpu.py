from time import time
import numpy as np

from classes.result import Result
from method.sssp.edge_cpu import edge as edge_sssp
from utils.debugger import Logger

logger = Logger(__name__)

def edge(para):
    """
    function: 
        use edgeSet in CPU to solve the APSP.  (more info please see the developer documentation) .
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.debug("turning to func edge-cpu-apsp")

    t1 = time()

    n, pathRecordBool = para.graph.n, para.pathRecordBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = edge_sssp(para)
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result