#encoding=utf-8
from utils.dispatcher import dispatch
from utils.readGraph import read
from utils.debugger import Logger

# set logging test update lcx added by wenake
logger = Logger(__name__)

def INF():
    """
    function: return the INF of this tool.
    
    parameters: None.
    
    return:
        the INF in this tools.
    """

    from utils.settings import INF as inf

    return inf


def main(graph = None, graphType = None, method = 'dij', useCUDA = True, pathRecordBool = False, srclist = None, grid=None, block=None):
    
    """
    function: 
        a calculate interface.
    
    parameters: 
        graph: the graph data that you want to get the shortest path.
        graphType: type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: use CUDA to speedup or not.
        pathRecordBool: record the path or not.
        srclist: the source list, can be [None, list, number].(more info please see the developer documentation).
    
    return:
        this func will return a Result(class). (more info please see the developer documentation) .  
    """
    # 跳转到 dispatch 函数进行分发
    # we only accept graphic data in edgeSet format 
    if(type(graph) == str):
        graphObj=read(graph)
        if(graphType=='edgeSet'):
            result = dispatch(graphObj.edgeSet, graphType, method, useCUDA, pathRecordBool, srclist, grid, block)
        else:
            result = dispatch(graphObj.CSR, 'CSR', method, useCUDA, pathRecordBool, srclist, grid, block)
    else:
        result = dispatch(graph, graphType, method, useCUDA, pathRecordBool, srclist, grid, block)
    
    logger.info(f"go to func 'dispatch', method is {method}, useCUDA is {useCUDA}, pathRecord is {pathRecordBool}, srclist is {srclist}")
    
    return result
    


if __name__ == "__main__":
    main()