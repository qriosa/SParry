from time import time
import numpy as np

from classes.result import Result
from method.sssp.spfa_cpu import spfa as spfa_sssp



def spfa(para):
    """
	function: use Bellman-Ford algorithm in CPU to solve the APSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .    
    """
    CSR = para.CSR
    n = para.n 
    pathRecording = para.pathRecordingBool
    
    start_time = time()
    Va=CSR[0]
    Ea=CSR[1]
    Wa=CSR[2]
    dist=[]
    for st in range(n):
        para.srclist = st
        resi = spfa_sssp(para)
        dist.append(resi.dist)
    para.srclist = None
    dist = np.array(dist)
    end_time = time()
    timeCost = end_time - start_time
    result = Result(dist = dist, timeCost = timeCost)
    return result
