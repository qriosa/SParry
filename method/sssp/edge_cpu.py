from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

logger = Logger(__name__)

def edge(para):
    """
    function: 
        use edgeSet in CPU to solve the SSSP.  (more info please see the developer documentation).
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    logger.debug("turning to func edge-cpu-sssp")

    t1 = time()

    edgeSet, n, m, s, pathRecordBool = para.graph.graph, para.graph.n, para.graph.m, para.srclist, para.pathRecordBool
    src, des, val = edgeSet[0], edgeSet[1], edgeSet[2]

    # exit flag
    flag = 1

    # dist
    dist = np.full((n, ), INF).astype(np.int32)
    dist[s] = 0 
 
    while True:
        # if there is not a vertex updated, then break
        if flag == 0:
            break

        flag = 0

        for i in range(len(src)):
            u, v, w = src[i], des[i], val[i]
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                flag = 1


    timeCost = time() - t1

    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)
    
    if pathRecordBool:
        result.calcPath()

    return result

