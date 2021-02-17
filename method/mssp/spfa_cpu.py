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
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter'). 
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result'). 
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
