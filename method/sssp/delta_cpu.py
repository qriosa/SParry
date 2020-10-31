from time import time
import numpy as np

from utils.settings import INF
from classes.result import Result

# CSR 结点数 源点 delta 最大边权
def delta_stepping(para):
    """
	function: use delta_stepping algorithm in CPU to solve the SSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        s: the source vertex, can be [None, list, number].(more info please see the developer documentation).
        delta: the delta of this algorithm.
        MAXN: the max value of all the edges in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .  

    """

    # 起始时间
    t1 = time()

    CSR, n, s, delta, MAXN, pathRecordingBool = para.CSR, para.n, para.srclist, para.delta, para.MAXN, para.pathRecordingBool

    V, E, W = CSR[0], CSR[1], CSR[2]
    
    # dist
    dist = np.full((n, ), INF).astype(np.int32)
    dist[s] = 0   

    B = []

    #初始化源点距离信息
    isin = np.full((n, ), -1).astype(np.int32) # 标记当前点在哪个桶里面 

    maxidB = MAXN * (n - 2) // delta + 1 #计算出桶的最大上限

    for i in range(maxidB):
        B.append([])

    B[0].append(s) # 源点放入第一个桶中
    isin[s] = 0


    idB = 0 #当前桶的id 从0开始，直到后面没有了
    while True:
        
        tag = 0 

        for i in range(idB, maxidB):
            if B[i] != []: # 向后找到第一个非空的
                tag = 1
                idB = i # 快速跳过中间的空桶
                break

        if tag == 0: # 全桶都空了
            break
        
        head = 0 # 已经用了多少点
        tail = len(B[idB]) # 桶中一共多少点

        while head < tail:

            p = B[idB][head]

            isin[p] = -1 

            for i in range(V[p], V[p + 1]):
                if W[i] <= delta:
                    if dist[E[i]] > dist[p] + W[i]:

                        dist[E[i]] = dist[p] + W[i]
                        newId = dist[E[i]] // delta

                        if isin[E[i]] != -1 and isin[E[i]] != newId:
                            B[isin[E[i]]].remove(E[i])
                        
                        if isin[E[i]] != newId:
                            isin[E[i]] = newId
                            
                            B[isin[E[i]]].append(E[i])

                            if newId == idB:
                                tail += 1

            head += 1

        for p in B[idB]:

            for i in range(V[p], V[p + 1]):
                if W[i] > delta:
                    if dist[E[i]] > dist[p] + W[i]:

                        dist[E[i]] = dist[p] + W[i]
                        newId = dist[E[i]] // delta

                        if isin[E[i]] != -1 and isin[E[i]] != newId:
                            B[isin[E[i]]].remove(E[i])
                        
                        
                        isin[E[i]] = newId
                        
                        B[isin[E[i]]].append(E[i])
                            
        idB += 1 # 下一个桶

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
