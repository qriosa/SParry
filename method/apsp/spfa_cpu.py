from time import time
import numpy as np

from utils.debugger import Logger
from classes.result import Result
from method.sssp.spfa_cpu import spfa as spfa_sssp

logger = Logger(__name__)

def spfa(para):
    """
    function: 
        use spfa algorithm in CPU to solve the APSP.
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func spfa-cpu-apsp")

    n = para.graph.n 
    pathRecording = para.pathRecordBool
    
    start_time = time()
    dist=[]
    for st in range(n):
        para.srclist = st
        resi = spfa_sssp(para)
        dist.append(resi.dist)
    para.srclist = None
    dist = np.array(dist)
    end_time = time()
    timeCost = end_time - start_time
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecording:
        result.calcPath()

    return result
