import numpy as np

from utils.dispatcher import dispatch
from utils.readGraph import read
from utils.debugger import Logger

# set logging test update lcx added by wenake
logger = Logger(__name__)

def getINF():
    """
    function: 
        return the INF of this tool.
    
    parameters: 
        None, no parameter.
    
    return:
        int, the INF in this tools.
    """

    from utils.settings import INF as inf

    return inf


def calc(graph = None, useCUDA = True, useMultiPro = False, pathRecordBool = False, srclist = None, block = None, grid = None, namename = None):
    
    """
    function: 
        a calculate interface.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: str, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        directed: bool, directed or not. only valid in read graph from file.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.

    return:
        class, Result object. (see the 'SPoon/classes/result.py/Result') 
    """
    # 跳转到 dispatch 函数进行分发
    # we only accept graphic data in edgeSet format 

    logger.info(f"entering calc func...")
     
    return dispatch(graph = graph, useCUDA = useCUDA, useMultiPro = useMultiPro, pathRecordBool = pathRecordBool, srclist = srclist, block = block, grid = grid, namename = namename)
    


if __name__ == "__main__":
    calc(namename = __name__)