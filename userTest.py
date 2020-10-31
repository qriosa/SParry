#encoding=utf-8
from utils.dispatcher import dispatch
from utils.readGraph import read
from utils.debugger import Logger
from utils.check import check

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


def readAnsMatrix(filename):


def main(inputGraph = None, graphType = None, outputGraph=None, method=None, useCUDA=True):
    """
    function: 
        an interface to prove the correctness of this algorithm with the standard input&output graph data that user provides
        this algorithm will calculate an answer to the input graph and compare answer with the output graph in APSP problem
        hence, we could prove the correctness of this algorithm
        we will test all of out calculating flow in this algorithm if the method is defined as None
        since this function is only for correctness proving, so there's no need to consider the performance with other config parameters
    
    parameters: 
        inputGraph: the graph data that you want to get the shortest path. [CSR edgeSet Matrix]
        graphType: type of the input graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        outputGraph:the graph data of answer that you want to get the shortest path. [only in matrix format]
        method: the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: use CUDA to speedup or not.
    
    return:
        no return but print the result of proving 
    """
    assert (inputGraph == None or outputGraph==None or graphType==None), "必须指定输入数据和标准输出，以及数据格式"

    # 跳转到 dispatch 函数进行分发
    # we only accept inputGraph data in edgeSet format 
    if(type(inputGraph) == str):
        graphObj=read(graph)
        if(graphType=='CSR'):
            inputGraph = graphObj.CSR
        else:
            inputGraph = graphObj.edgeSet
            graphType = "edgeSet"
    
    if(type(outputGraph) == str):
        outputGraph=readAnsMatrix(outputGraph)
    
    pltform=['cpu','GPU']
    # if define method as None , we will test all calculating path we have
    if(method == None):
        methods=['dij','spfa','delta','edge']
        usage = [True, False]
        for msz in methods:
            for use in usage:
                result = dispatch(inputGraph, graphType, msz, use, False, None, None, None)
                print(check(result.matrix, outputGraph, f'answer[{msz} {pltform[use]}]','stdout'))
    else:
        result = dispatch(inputGraph, graphType, method, useCUDA, False, None, None, None)
        print(check(result.matrix, outputGraph, f'answer[{methods} {pltform[useCUDA]}]','stdout'))


if __name__ == "__main__":
    main()