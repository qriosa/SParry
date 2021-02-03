import numpy as np
import os

from classes.graph import Graph
from utils.settings import INF
from utils.debugger import Logger

logger = Logger(__name__)

def read(CSR = None, matrix = None, edgeSet = None, filename = "", method = "dij", detail = False, directed = "Unknown", delta = 3):
    """
    function: 
        convert a graph from [CSR/matrix/edgeSet/file] 
            to Graph object(see the 'SPoon/classes/graph.py/Graph')
            as the paremeter 'graph' of the func calc(see the 'SPoon/calc.py/calc')
    
    parameters: 
        CSR: tuple, optional, a 3-tuple of integers as (V, E, W) about the CSR of graph data.
            (more info please see the developer documentation).
        matrix: matrix/list, optional, the adjacency matrix of the graph.
            (more info please see the developer documentation).
        edgeSet: tuple, optional, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
            (more info please see the developer documentation).
        filename: str, optional, the name of the graph file, optional, and the graph should be list as edgeSet.
            (more info please see the developer documentation).
        method: str, optional, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
            (more info please see the developer documentation).
        detail: bool, optional, default 'False', whether you need to read the detailed information of this picture or not.
            If you set True, it will cost more time to get the detail information.
        directed: bool, optional, default 'False', the graph is drected or not.
            It will be valid, only the 'filename' parameter is set.
        delta: int, optional, default 3, the delta of the delta-stepping algorithom.
            It will be valid, only you choose method is delta-stepping.

        ATTENTION:
            CSR, matrix, edgeSet and filename cann't be all None. 
            You must give at least one graph data.
            And the priority of the four parameters is:
                CSR > matrix > edgeSet > filename.

    return:
        class, Graph object. (see the 'SPoon/classes/graph.py/Graph')     
    """
    
    """
    only use to read graph, and translate to CSR, detail is calc the detail info of the graph.
    delta default is 3
    priority: CSR -> matrix -> edgeSet -> filename
    """

    logger.info("entering pretreat.read func.")

    # can not be all None
    if(type(CSR) == type(None) and type(matrix) == type(None) and type(edgeSet) == type(None) and filename == ""):
        raise Exception("CSR, matrix, edgeSet and filename can not be all NULL!") 

    CSR = np.array(CSR)
    edgeSet = np.array(edgeSet)
    matrix = np.array(matrix)

    g = Graph()
    # directed or not
    g.directed = directed
    # delta's delta val
    g.delta = delta

    # set method
    if(method == "dij" or method == "spfa" or method == "delta" or method == "edge"):
        g.method = method
    else:
        raise Exception("method can only be one of ['dij', 'spfa', 'delta', 'edge'], not " + method)

    # check type
    # CSR 
    if((CSR.shape == ()) == False):
        if(type(CSR) != np.ndarray):
            raise Exception("CSR can only be list or numpy.ndarray")

        # need to transfer CSR to edgeSet
        if g.method == "edge":

            logger.info("processing CSR to edgeSet")

            V = CSR[0]
            E = CSR[1]
            W = CSR[2]
            
            # set it to g
            g.n = len(V)-1
            g.m = len(E)

            g.degree = np.zeros(g.n, dtype = np.int32)

            src = []
            des = []
            val = []
            edgeSet=[]
            MAXW = -1

            for u in range(g.n):
                # change degree
                d = V[u + 1] - V[u]
                g.degree[u] = d

                # max degree
                if d > g.MAXD:
                    g.MAXD = d
                    g.MAXU = u
                # min degree
                if d < g.MIND:
                    g.MIND = d
                    g.MINU = u

                for ind in range(V[u],V[u + 1]):
                    
                    src.append(u)
                    des.append(E[ind])
                    val.append(W[ind])

                    # max edge weight
                    g.MAXW = max(g.MAXW, W[ind])
                    # min edge  weight
                    g.MINW = min(g.MINW, W[ind])
            
            # edgeSet
            g.graph = [src, des, val]
            # msg
            g.setmsg()

        # keep CSR 
        else:
            logger.info("processing CSR")

            g.n = len(CSR[0]) - 1 
            g.m = len(CSR[1]) 
            g.graph = CSR

            # get detail info of the graph
            if detail:
                # get the degree
                g.degree = np.zeros(g.n, dtype = np.int32)
                # average degree
                g.avgDegree = g.m / g.n 

                for i in range(g.n):
                    # degree
                    d = CSR[0][i+1] - CSR[0][i]
                    g.degree[i] = d

                    # max degree
                    if d > g.MAXD:
                        g.MAXD = d
                        g.MAXU = i
                    # min degree
                    if d < g.MIND:
                        g.MIND = d
                        g.MINU = i
                    
                    for j in range(CSR[0][i], CSR[0][i+1]):
                        g.MAXW = max(g.MAXW, CSR[2][j])
                        g.MINW = min(g.MINW, CSR[2][j])
                
                # msg
                g.setmsg()

    # edgeSet
    elif((edgeSet.shape == ()) == False):
        if(type(edgeSet) != np.ndarray):
            raise Exception("edgeSet can only be list or numpy.ndarray")
        # the length must be equal
        if not (len(edgeSet[0]) == len(edgeSet[1]) == len(edgeSet[2])):
            raise Exception("invalid edgeSet data")
        
        # keep edge
        if g.method == "edge":
            logger.info("processing edgeSet")

            # from 0 to n, so the number of the vertex need to add one.
            g.n = max(max(edgeSet[0]), max(edgeSet[1])) + 1
            g.m = len(edgeSet[0])
            g.graph = edgeSet

            # detail
            if detail:
                # average degree
                g.avgDegree = g.m / g.n 

                g.degree = np.zeros(g.n, dtype = np.int32)

                src = g.graph[0]
                des = g.graph[1]
                w = g.graph[2]

                for i in range(g.m):
                    g.MAXW = max(g.MAXW, w[i])
                    g.MINW = min(g.MINW, w[i])

                    g.degree[src[i]] += 1 
                
                g.MAXD = g.degree.max()
                g.MAXU = g.degree.argmax()
                g.MIND = g.degree.min()
                g.MINU = g.degree.argmin()

                # msg
                g.setmsg()

        # change to CSR
        else:
            logger.info("processing edgeSet to CSR")

            src = edgeSet[0]
            des = edgeSet[1]
            w = edgeSet[2]

            g.n = max(max(src), max(des)) + 1
            g.m = len(src)

            #  V E W
            V = np.zeros(g.n + 1, dtype = np.int32) # so need add one
            E = np.full((g.m,), -1).astype(np.int32)
            W = np.full((g.m,), INF).astype(np.int32)

            # degree
            g.degree = np.zeros(g.n, dtype = np.int32)

            for i in range(g.m):
                g.degree[src[i]] += 1

            for u in range(g.n):
                V[u + 1] = V[u] + g.degree[u]
                
                if g.MAXD < g.degree[u]:
                    g.MAXD = g.degree[u]
                    g.MAXU = u

                if g.MIND > g.degree[u]:
                    g.MIND = g.degree[u]
                    g.MINU = u
            

            edgeOfV = np.zeros(g.n, dtype = np.int32)
            for i in range(g.m):

                u = src[i]

                E[V[u] + edgeOfV[u]] = des[i]
                W[V[u] + edgeOfV[u]] = w[i]

                edgeOfV[u] += 1

                g.MAXW = max(g.MAXW, w[i])
                g.MINW = min(g.MINW, w[i])

            g.graph = [V, E, W]

            # msg
            g.setmsg()        

    # matrix
    elif((matrix.shape == ()) == False):
        if(matrix.shape[0] != matrix.shape[1]):
            raise Exception("Matrix must be matrix, not a wrong shape " + str(matrix.shape))
        
        g.n = matrix.shape[0]
        g.m = 0

        g.degree = np.zeros(g.n, dtype = np.int32)

        # change to edgeSet
        if(g.method == "edge"):

            logger.info("processing matrix to edgeSet")
            
            src = []
            des = []
            w = []

            for i in range(g.n):
                for j in range(g.n):
                    if(matrix[i][j] < INF):
                        # i -> j
                        src.append(i)
                        des.append(j)
                        w.append(matrix[i][j])

                        g.m += 1
                        g.degree[i] += 1

                        g.MAXW = max(g.MAXW, w[-1])
                        g.MINW = min(g.MINW, w[-1])
                
                if g.MAXD < g.degree[i]:
                    g.MAXD = g.degree[i]
                    g.MAXU = i

                if g.MIND > g.degree[i]:
                    g.MIND = g.degree[i]
                    g.MINU = i

            # graph
            g.graph = [src, des, w]

            # msg
            g.setmsg()

        # change to CSR
        else:
            logger.info("processing matrix to CSR")

            #  V E W
            V = np.zeros(g.n + 1, dtype = np.int32) # add one as Virtual
            E = []
            W = []

            for i in range(g.n):
                for j in range(g.n):
                    if matrix[i][j] < INF:
                        g.degree[i] += 1
                        g.m += 1

                        E.append(j)
                        W.append(matrix[i][j])

                        g.MAXW = max(g.MAXW, W[-1])
                        g.MINW = min(g.MINW, W[-1])

                V[i + 1] = V[i] + g.degree[i] 

                if g.MAXD < g.degree[i]:
                    g.MAXD = g.degree[i]
                    g.MAXU = i
               
                if g.MIND > g.degree[i]:
                    g.MIND = g.degree[i]
                    g.MINU = i
            
            # graph
            g.graph = [V, E, W]

            # msg
            g.setmsg()

    # filename 
    elif(filename != ""):
        if(os.path.exists(filename) == False):
            raise Exception("no such a file or dictionary named " + filename)
        # direct not true is undirect 
        # If we don't know, then show Unknown 
        if(g.directed != True):
            g.directed = False

        # open file
        try:
            with open(filename, 'r') as f:
                content = f.read()    
        except:
            raise Exception("fail to open file " + filename)

        # split to list and get rid of "\n" in the end
        contenList = content.split("\n")
        contenList.remove("")

        nm_str = contenList[0].split(" ")
        g.n = np.int32(nm_str[0]) 
        g.m = np.int32(nm_str[1])
        
        # directed
        if g.directed == False:
            g.m *= 2
        # degree
        g.degree = np.zeros(g.n, dtype=np.int32)

        lines = contenList[1:]
        src = []
        des = []
        w = []

        if g.directed:
            for line in lines:
                edge = line.split(" ")
                temp1 = np.int32(edge[0])
                temp2 = np.int32(edge[1])
                temp3 = np.int32(edge[2])

                src.append(temp1)
                des.append(temp2)
                w.append(temp3)

                g.degree[temp1] += 1

                g.MINW = min(g.MINW, temp3)
                g.MAXW = max(g.MAXW, temp3)
        else:
            for line in lines:
                edge = line.split(" ")
                temp1 = np.int32(edge[0])
                temp2 = np.int32(edge[1])
                temp3 = np.int32(edge[2])

                src.append(temp1)
                des.append(temp2)
                w.append(temp3)

                src.append(temp2)
                des.append(temp1)
                w.append(temp3)

                g.degree[temp1] += 1
                g.degree[temp2] += 1

                g.MINW = min(g.MINW, temp3)
                g.MAXW = max(g.MAXW, temp3)

        # read as edgeSet
        if g.method == "edge":
            logger.info("processing filelist to edgeSet")

            if detail:
                g.MAXD = g.degree.max()
                g.MAXU = g.degree.argmax()
                g.MIND = g.degree.min()
                g.MINU = g.degree.argmin()

                # msg
                g.setmsg()
            
            g.graph = [src, des, w]

        # read as CSR       
        else:
            logger.info("processing filelist to CSR")

            #  V E W
            V = np.zeros(g.n + 1, dtype=np.int32) 
            E = np.full((g.m,), -1).astype(np.int32)
            W = np.full((g.m,), INF).astype(np.int32)
            
            V[g.n] = g.m
        
            for u in range(g.n):
                
                V[u + 1] = V[u] + g.degree[u]
                
                if g.MAXD < g.degree[u]:
                    g.MAXD = g.degree[u]
                    g.MAXU = u
              
                if g.MIND > g.degree[u]:
                    g.MIND = g.degree[u]
                    g.MINU = u       
            
            edgeOfV = np.zeros(g.n, dtype = np.int32)

            for i in range(g.m):
                u = src[i]

                E[V[u] + edgeOfV[u]] = des[i]
                W[V[u] + edgeOfV[u]] = w[i]

                edgeOfV[u] += 1

            #  graph
            g.graph = [V, E, W]

            # msg
            g.setmsg()

    else:
        raise Exception("What happend??")

    # reshape
    g.reshape()

    # g
    return g

if __name__ == "__main__":
    """
    CSR = None, 
    matrix = None, 
    edgeSet = None, 
    filename = "", 
    method = "dij",
    detail = False, 
    directed = "Unknown", 
    delta = 3
    """
    filename = "data10_20.txt"

    CSR = np.array([np.array([ 0,  3,  5,  9, 15, 19, 24, 27, 32, 36, 40]), np.array([1, 2, 5, 0, 4, 0, 3, 7, 4, 2, 4, 6, 7, 5, 7, 1, 2, 3, 9, 0, 3, 8,
       9, 9, 3, 8, 8, 2, 3, 3, 7, 7, 5, 6, 6, 9, 4, 5, 5, 8]), np.array([9, 5, 8, 9, 8, 5, 2, 9, 3, 2, 2, 6, 6, 2, 6, 8, 3, 2, 2, 8, 2, 4,
       4, 8, 6, 1, 5, 9, 6, 6, 4, 4, 4, 1, 5, 7, 2, 4, 8, 7])])
    
    edgeSet = [[0, 1, 0, 2, 0, 5, 1, 4, 2, 3, 2, 7, 2, 4, 3, 4, 3, 6, 3, 7, 3, 5, 3, 7, 4, 9, 5, 8, 5, 9, 5, 9, 6, 8, 6, 8, 7, 7, 8, 9], [1, 0, 2, 0, 5, 0, 4, 1, 3, 2, 7, 2, 4, 2, 4, 3, 6, 3, 7, 3, 5, 3, 7, 3, 9, 4, 8, 5, 9, 5, 9, 5, 8, 6, 8, 6, 7, 7, 9, 8], [9, 9, 5, 5, 8, 8, 8, 8, 2, 2, 9, 9, 3, 3, 2, 2, 6, 6, 6, 6, 2, 2, 6, 6, 2, 2, 4, 4, 4, 4, 8, 8, 1, 1, 5, 5, 4, 4, 7, 7]]
    
    matrix = [0,1,2,3],[1,0,2,3],[2,2,0,4],[3,3,4,0]

    g = read(matrix = matrix, detail = True, directed=False, method = "edge")
    print(g.msg)
    # print(g.degree)