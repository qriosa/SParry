from time import time
import numpy as np

from classes.result import Result
from utils.settings import INF
from utils.debugger import Logger

logger = Logger(__name__)

def edge(para):
    """
    function: 
        use edgeSet in CPU to solve the SSSP.  (more info please see the developer documentation) .
    
    parameters:  
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) . 
    """

    logger.info("turning to func edge-cpu-sssp")

    t1 = time()

    edgeSet, n, m, s, pathRecordBool = para.edgeSet, para.n, para.m, para.srclist, para.pathRecordBool
    src, des, val = para.edgeSet[0], para.edgeSet[1], para.edgeSet[2]

    # 退出标识
    flag = 1

    # dist
    dist = np.full((n, ), INF).astype(np.int32)
    dist[s] = 0 
    # print(dist.shape)  
    while True:
        # 如果没有点的距离发生改变，则退出遍历
        if flag == 0:
            break

        flag = 0

        # edge = (u, v, w): u -> v = w
        for i in range(len(src)):
            u, v, w = src[i], des[i], val[i]
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                flag = 1


    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, msg = para.msg, graph = para.edgeSet, graphType = 'edgeSet')
    
    if pathRecordBool:
        result.calcPath()

    return result

