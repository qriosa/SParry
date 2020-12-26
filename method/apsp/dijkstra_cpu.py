from time import time
import numpy as np
from queue import PriorityQueue
from multiprocessing import Process, cpu_count, Queue
from multiprocessing.sharedctypes import RawArray

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
    logger.debug("turning to func dijkstra-cpu-apsp")

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

    logger.debug("turning to func dijkstra-cpu-apsp single-process")

    t1 = time()

    CSR, n, pathRecordBool = para.graph.graph, para.graph.n, para.pathRecordBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = dij_sssp(para) 
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result


def dijkstra_multi_sssp(V, E, W, n, sources, distQ):
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

    # 优先队列
    q = PriorityQueue()

    # 任务调度
    while sources.empty() == False:

        # 获取一个源点
        s = sources.get()

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
    
        distQ.put((s, dist))


def dijkstra_multi(para):
    """
    function: 
        use dijkstra algorithm in ALL CPU cores to solve the APSP PARALLEL. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter')
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.debug("turning to func dijkstra-cpu-apsp multi-process")
    if para.namename == "__main__":

        t1 = time()
        q = Queue()

        CSR, n, pathRecordBool = para.graph.graph, para.graph.n, para.pathRecordBool

        shared_V = RawArray('i', CSR[0])
        shared_E = RawArray('i', CSR[1])
        shared_W = RawArray('i', CSR[2])

        del CSR
        
        # 源点队列
        sources = Queue()

        for i in range(n):
            sources.put(i)

        
        # 创建 核心数 这么多个进程 通过队列实现任务调度
        cores = cpu_count()
        myProcesses = [Process(target = dijkstra_multi_sssp, args = (shared_V, shared_E, shared_W, n, sources, q)) for _ in range(cores)]

        for myProcess in myProcesses:
            myProcess.start()
        
        for myProcess in myProcesses:
            if myProcess.is_alive():
                myProcess.join()
        
        dist = [None for i in range(n)]

        while q.empty() == False:
            temp = q.get()
            dist[temp[0]] = temp[1]

        dist = np.array(dist)

        timeCost = time() - t1

        # 结果
        result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

        if pathRecordBool:
            result.calcPath()

        return result
    else:
        print("herherherherhehr")
        print(__name__)
        print("jjjjjjjjjjjj")
        print(para.namename)
        print("qqqqqqqqqqqqqq")