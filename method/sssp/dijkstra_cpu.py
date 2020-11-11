from queue import PriorityQueue
from time import time
import numpy as np

from utils.settings import INF
from utils.debugger import Logger
from classes.result import Result

logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in CPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.info("turning to func dijkstra-cpu-sssp")

    t1 = time()

    CSR, n, s, pathRecordBool = para.CSR, para.n, para.srclist, para.pathRecordBool

    V, E, W = CSR[0], CSR[1], CSR[2]

    # 优先队列
    q = PriorityQueue()

    # 距离数组
    dist = np.full((n,), INF).astype(np.int32)
    dist[s] = 0

    # vis 数组
    vis = np.full((n, ), 0).astype(np.int32)

    # 开始计算
    q.put((0, s))#放入s点

    while q.empty() == False:
        p = q.get()[1]

        if vis[p] == 1: #如果当前节点松弛已经过了，则不需要再松弛了
            continue

        vis[p] = 1

        for j in range(V[p], V[p + 1]):
            if dist[E[j]] > dist[p] + W[j]:
                dist[E[j]] = dist[p] + W[j]
                q.put((dist[E[j]], E[j]))

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, msg = para.msg, graph = para.CSR, graphType = 'CSR')

    if pathRecordBool:
        result.calcPath()

    return result