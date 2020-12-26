from time import time
import numpy as np
import copy

from classes.result import Result
from utils.debugger import Logger
from method.sssp.spfa_cpu import spfa as spfa_sssp

logger = Logger(__name__)

def spfa(para):
    """
    function: 
        use spfa algorithm in CPU to solve the MSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter') 
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """

    logger.debug("turning to func spfa-cpu-mssp")

    CSR = para.graph.graph
    n = para.graph.n 
    srclist = copy.deepcopy(para.srclist)
    pathRecording = para.pathRecordBool

    start_time = time()
    Va=CSR[0]
    Ea=CSR[1]
    Wa=CSR[2]
    dist=[]
    for st in srclist:
        para.srclist = st
        resi = spfa_sssp(para)
        dist.append(resi.dist)
    para.srclist = srclist
    end_time = time()
    timeCost = end_time - start_time
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecording:
        result.calcPath()

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