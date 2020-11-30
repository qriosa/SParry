from queue import PriorityQueue
from threading import Thread, Lock
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

    return dij_serial(para)

    # if para.useThreading == False:
    #     return dij_serial(para)
    # else:
    #     return dij_concurrent(para)

def dij_serial(para):
    """
    function: 
        use dijkstra algorithm in CPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    # logger.info("turning to func dij_serial-sssp")

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


# 放弃多线程了
# class MyThread:
#     def __init__(self, V, E, W, dist, vis, s):

#         # V E W dist vis
#         self.V = V
#         self.E = E
#         self.W = W
#         self.dist = dist
#         self.vis = vis

#         self.tail = 1 # queue 是否为空不可信 就自己来实现
#         self.head = 0
        
#         # lock the dist, and the tail head
#         self.lock_dist = Lock() 
#         self.lock_index = Lock()

#         # 优先队列
#         self.q = PriorityQueue()
#         self.q.put((0, s)) #放入s点
        

#     def getSSSP(self):
        
#         while True:
            
#             # 队列空了
#             if self.head >= self.tail:
#                 break
            
#             p = self.q.get(block=False) # 非阻塞请求
            
#             p = p[1]

#             tocontinue = 0

#             with self.lock_index:

#                 if self.vis[p] == 1:
#                     # 之所以在这里才判定出队 是怕当前线程再放点进去 这里判定当前线程就不可能再放进去了
#                     # with self.lock_index:
#                     #     self.head += 1 # 元素 -1
#                     self.head += 1 # 元素 -1

#                     # continue
#                     tocontinue = 1
                
#                 self.vis[p] = 1
                
                
#             if tocontinue == 1:
#                 continue

#             # 点数需要小于 n 才有意义
            
#             for j in range(self.V[p], self.V[p + 1]):
#                 # with self.lock_dist:
#                 #     if self.dist[self.E[j]] > self.dist[p] + self.W[j]:
#                 #         self.dist[self.E[j]] = self.dist[p] + self.W[j]
#                 #         self.q.put((self.dist[self.E[j]], self.E[j]))
                        
#                 #         with self.lock_index:
#                 #             self.tail += 1 # 元素 +1
                
#                 if self.dist[self.E[j]] > self.dist[p] + self.W[j]:
#                     self.dist[self.E[j]] = self.dist[p] + self.W[j]
#                     self.q.put((self.dist[self.E[j]], self.E[j]))
                    
                    
#                     self.tail += 1 # 元素 +1
            
#             # 之所以在这里才判定出队 是怕当前线程再放点进去 这里判定当前线程就不可能再放进去了
#             # with self.lock_index:
#             #     self.head += 1 # 元素 -1
#             self.head += 1 # 元素 -1
            

            
            

# def dij_concurrent(para):
#     """
#     function: 
#         use dijkstra algorithm in CPU (with concurrent) to solve the SSSP. 
    
#     parameters:  
#         class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
#     return: 
#         class, Result object. (see the 'SPoon/classes/result.py/Result')     
#     """

#     logger.info("turning to func dij_concurrent-sssp")

#     t1 = time()

#     CSR, n, s, pathRecordBool = para.CSR, para.n, para.srclist, para.pathRecordBool

#     V, E, W = CSR[0], CSR[1], CSR[2]

#     # vis 数组
#     vis = np.full((n, ), 0).astype(np.int32)

#     # 距离数组
#     dist = np.full((n,), INF).astype(np.int32)
#     dist[s] = 0    

#     # 实例化线程类
#     myThread = MyThread(V, E, W, dist, vis, s)

#     # 线程数量
#     threadNum = 8
#     ts = [Thread(target = myThread.getSSSP) for i in range(threadNum)]

#     # 启动线程
#     for ti in ts:
#         ti.start()

#     # 等待线程执行完毕
#     for ti in ts:
#         ti.join()    

#     timeCost = time() - t1

#     # 结果
#     result = Result(dist = myThread.dist, timeCost = timeCost, msg = para.msg, graph = para.CSR, graphType = 'CSR')

#     if pathRecordBool:
#         result.calcPath()

#     return result
