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
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) . 
    """

    logger.info("turning to func edge-cpu-apsp")

    t1 = time()

    edgeSet, n, m, pathRecordBool = para.edgeSet, para.n, para.m, para.pathRecordBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = edge_sssp(para)
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, msg = para.msg, graph = para.edgeSet, graphType = 'edgeSet')

    if pathRecordBool:
        result.calcPath()

    return result