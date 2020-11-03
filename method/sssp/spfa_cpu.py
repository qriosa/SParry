from time import time
import numpy as np

from utils.settings import INF
from utils.debugger import Logger
from classes.result import Result

logger = Logger(__name__)

def spfa(para):
    """
    function: 
        use spfa algorithm in CPU to solve the APSP. 
    
    parameters:  
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) .
    """

    logger.info("turning to func spfa-cpu-apsp")

    CSR = para.CSR
    n = para.n 
    s = para.srclist
    pathRecording = para.pathRecordingBool
    start_time = time()
    Va=CSR[0]
    Ea=CSR[1]
    Wa=CSR[2]
    Que=[]
    dist=[INF for i in range(0,n)]
    inQ=[0 for i in range(0,n)]
    Que.append(s)
    dist[s]=0
    inQ[s]=1
    head=0
    #MultiThread
    while(len(Que)-head>0):
        nowVer=Que[head]
        head=head+1
        inQ[nowVer]=0
        for i in range(Va[nowVer],Va[nowVer+1]):
            if(dist[Ea[i]]>dist[nowVer]+Wa[i]):
                dist[Ea[i]]=dist[nowVer]+Wa[i]
                if(inQ[Ea[i]]==0):
                    inQ[Ea[i]]=1
                    Que.append(Ea[i])
    
    dist=np.array(dist)
    end_time = time()
    timeCost = end_time - start_time
    result = Result(dist = dist, timeCost = timeCost)
    # if(pathRecording):
    #     result.calcPath(CSR=CSR)
    return result