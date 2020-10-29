from time import time
import numpy as np

from utils.settings import INF
from classes.result import Result

def spfa(CSR,n,s,pathRecording=False):
    """
	function: use Bellman-Ford algorithm in CPU to solve the SSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        srclist: the source list, can be number.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .     
    """
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