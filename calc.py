#encoding=utf-8
from utils.dispatcher import dispatch
from utils.readGraph import read
from utils.debugger import Logger

# set logging test update lcx added by wenake
logger = Logger(__name__)

def INF():
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


def main(graph = None, graphType = None, method = 'dij', useCUDA = True, pathRecordBool = False, srclist = None, block=None, grid=None):
    
    """
    function: 
        a calculate interface.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: str, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: bool, use CUDA to speedup or not.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.

    return:
        class, Result object. (more info please see the developer documentation).  
    """
    # 跳转到 dispatch 函数进行分发
    # we only accept graphic data in edgeSet format 

    logger.info(f"go to func 'dispatch', method is {method}, useCUDA is {useCUDA}, pathRecord is {pathRecordBool}, srclist is {srclist}")
  
    if(type(graph) == str):
        graphObj=read(graph)
        if(graphType=='edgeSet'):
            result = dispatch(graphObj.edgeSet, graphType, method, useCUDA, pathRecordBool, srclist, graphObj.msg, block, grid)
        else:
            result = dispatch(graphObj.CSR, 'CSR', method, useCUDA, pathRecordBool, srclist, graphObj.msg, block, grid)
    else:
        result = dispatch(graph, graphType, method, useCUDA, pathRecordBool, srclist, "", block, grid)
     
    return result
    


if __name__ == "__main__":
    main()