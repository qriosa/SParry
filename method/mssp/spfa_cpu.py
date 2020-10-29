from time import time
import numpy as np

from classes.result import Result
from method.sssp.spfa_cpu import spfa as spfa_sssp



def spfa(CSR,n,srclist,pathRecordingBool):
    """
	function: use Bellman-Ford algorithm in CPU to solve the MSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        srclist: the source list, can be list.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .      
    """
    global Va,Ea,Wa
    start_time = time()
    Va=CSR[0]
    Ea=CSR[1]
    Wa=CSR[2]
    dist=[]
    for st in srclist:
        resi = spfa_sssp(CSR,n,st)
        dist.append(resi.dist)
    end_time = time()
    timeCost = end_time - start_time
    result = Result(dist = dist, timeCost = timeCost)
    return result

# def spfa_iterator(n,st):
#     global Va,Ea,Wa
#     Que=[]
#     dist=[0x7f7f7f7f for i in range(0,n+1)]
#     inQ=[0 for i in range(0,n+1)]
#     Que.append(st)
#     dist[st]=0
#     inQ[st]=1
#     head=0
#     while(len(Que)-head>0):
#         nowVer=Que[head]
#         head=head+1
#         inQ[nowVer]=0
#         for i in range(Va[nowVer],Va[nowVer+1]):
#             if(dist[Ea[i]]>dist[nowVer]+Wa[i]):
#                 dist[Ea[i]]=dist[nowVer]+Wa[i]
#                 if(inQ[Ea[i]]==0):
#                     inQ[Ea[i]]=1
#                     Que.append(Ea[i])