from time import time
import numpy as np
from queue import PriorityQueue
from multiprocessing import Process, cpu_count, Manager, Pool

from classes.result import Result
from utils.debugger import Logger
from utils.settings import INF
from method.sssp.dijkstra_cpu import dijkstra as dij_sssp

logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in CPU to solve the APSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """
    logger.info("turning to func dijkstra-cpu-apsp")

    if para.useMultiPro == True:
        return dijkstra_multi(para)
    else:
        return dijkstra_single(para)


def dijkstra_single(para):
    """
    function: 
        use dijkstra algorithm in A SINGLE CPU core to solve the APSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.info("turning to func dijkstra-cpu-apsp single-process")

    t1 = time()

    CSR, n, pathRecordBool = para.CSR, para.n, para.pathRecordBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = dij_sssp(para) 
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, msg = para.msg, graph = para.CSR, graphType = 'CSR')

    if pathRecordBool:
        result.calcPath()

    return result


# share_V, share_E, share_W, n = None, None, None, 

def dijkstra_multi_sssp(args):
    """
    function: 
        use dijkstra algorithm to solve a sssp as a process. 
    
    parameters:  
        V, array, the CSR[0]. 
        E, array, the CSR[1]. 
        W, array, the CSR[2]. (see the document)
        s, int, the source vertex.
        n, int, the number of vertices.
    
    return: 
        dist, array, the distance array. 
    """

    V, E, W, s, n = args[0], args[1], args[2], args[3], args[4]

    # 优先队列
    q = PriorityQueue()

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
    
    return dist


def dijkstra_multi(para):
    """
    function: 
        use dijkstra algorithm in ALL CPU cores to solve the APSP PARALLEL. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.info("turning to func dijkstra-cpu-apsp multi-process")

    t1 = time()

    CSR, n, pathRecordBool = para.CSR, para.n, para.pathRecordBool

    # share_V = CSR[0]
    # share_E = CSR[1]
    # share_W = CSR[2]
    
    tt = time()

    share_V = Manager().Array('i', CSR[0])
    share_E = Manager().Array('i', CSR[1])
    share_W = Manager().Array('i', CSR[2])

    print("convert time cost: ", time() - tt)

    # del CSR

    tt = time()
    xs = []
    for s in range(n):
        xs.append([share_V, share_E, share_W, s, n])
    print("append time cost: ", time() - tt)

    cores = cpu_count()
    with Pool(cores) as pool:
        dist = np.array(pool.map(dijkstra_multi_sssp, xs))
    
    print("finish calc")
    
    # cores = cpu_count()
    # myProcesses = [Process(target = dijkstra_multi_sssp, args=(share_V, share_E, share_W, share_dist, s, n)) for s in range(n)]

    # for myProcess in myProcesses:
    #     myProcess.start()
    
    # for myProcess in myProcesses:
    #     myProcess.join()
            
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, msg = para.msg, graph = para.CSR, graphType = 'CSR')

    if pathRecordBool:
        result.calcPath()

    return result