# created by wenake 2020.10.16
from utils.settings import INF
from utils.debugger import Logger

import numpy as np

logger = Logger(__name__)

def CSR2Matrix(CSR):
    """
    function: 
        transfer CSR graph to matrix graph.
    
    parameters: 
        CSR: tuple, CSR graph.
    
    return: 
        matrix, matrix graph.

    may I ask user-sama input a correct CSR format graph data please? thanks
    """

    logger.info("transfering CSR to Matrix.")

    V = CSR[0]
    E = CSR[1]
    W = CSR[2]
    n = len(V)-1
    mat = [[INF for i in range(n)] for j in range(n)]
    for u in range(n):
        for ind in range(V[u],V[u+1]):
            mat[u][E[ind]]=W[ind]
    return np.int32(n), np.int32(len(E)), np.array(mat,dtype=np.int32)

def CSR2edgeSet(CSR):
    """
    function: 
        transfer CSR graph to edgeSet graph.
    
    parameters: 
        CSR: tuple, CSR graph.
    
    return: 
        tuple, edgeSet graph.
    """

    logger.info("transfering CSR to edgeSet.")

    V = CSR[0]
    E = CSR[1]
    W = CSR[2]
    n = len(V)-1
    src = []
    des = []
    val = []
    edgeSet=[]
    for u in range(n):
        for ind in range(V[u],V[u+1]):
            # edgeSet.append((u,E[ind],W[ind]))
            src.append(u)
            des.append(E[ind])
            val.append(W[ind])
    edgeSet = [src, des, val]
    return np.int32(n), np.int32(len(E)), np.array(edgeSet,dtype=np.int32)

def matrix2CSR(mat):
    """
    function: 
        transfer matrix graph to CSR graph.
    
    parameters: 
        mat: matrix, matrix graph.
    
    return: 
        tuple, CSR graph.
    """

    logger.info("transfering Matrix to CSR.")

    n = len(mat)
    V = [0 for i in range(n+1)]
    E = []
    W = []
    for u in range(n):
        tot = 0
        for v in range(n):
            w=mat[u][v]
            # V[u+1]=V[u]
            if(w < INF):
                tot += 1
                V[u+1]=V[u+1]+1
                E.append(v)
                W.append(w)
        V[u+1] = V[u] + tot
    V[n]=len(W)
    return np.int32(n),np.int32(len(E)-1),[np.array(V,dtype=np.int32), np.array(E,dtype=np.int32), np.array(W,dtype=np.int32)]

def matrix2edgeSet(mat):
    """
    function: 
        transfer matrix graph to edgeSet graph.
    
    parameters: 
        mat: matrix, matrix graph.
    
    return: 
        tuple, edgeSet graph.
    """

    logger.info("transfering Matrix to edgeSet.")

    src = []
    des = []
    val = []
    edgeSet=[]
    n = len(mat)
    edgeSet = []
    for u in range(n):
        for v in range(n):
            w=mat[u][v]
            if(w < INF):
                # edgeSet.append([u,v,w])
                src.append(u)
                des.append(v)
                val.append(w)
    edgeSet = [src, des, val]                
    return np.int32(n),np.int32(len(edgeSet)),np.array(edgeSet,np.int32)

def edgeSet2Matrix(edgeSet):
    """
    function: 
        transfer edgeSet graph to matrix graph.
    
    parameters: 
        edgeSet: tuple, edgeSet graph.
    
    return: 
        matrix, matrix graph.    
    """

    logger.info("transfering edgeSet to Matrix.")

    m = len(edgeSet)
    n = 0
    for ind in range(len(edgeSet[0])):
        n=max(n,edgeSet[0][ind])
        n=max(n,edgeSet[1][ind])
    n=n+1
    mat = np.full((n,n),INF).astype(np.int32)
    # print(mat.shape)
    for ind in range(len(edgeSet[0])):
        u, v, w=edgeSet[0][ind], edgeSet[1][ind], edgeSet[2][ind]
        # print(u,v)
        mat[u,v]=min(mat[u,v],w)
        mat[v,u]=min(mat[v,u],w)
    return np.int32(n),np.int32(m),mat

def edgeSet2CSR(edgeSet):
    """
    function: 
        transfer edgeSet graph to CSR graph.
    
    parameters: 
        edgeSet: tuple, edgeSet graph.
    
    return: 
        tuple, CSR graph.
    """

    logger.info("transfering edgeSet to CSR.")

    # print(edgeSet.shape)
    n,m,mat=edgeSet2Matrix(edgeSet)
    return matrix2CSR(mat)

def transfer(para, outType):
    """
    function: 
        transfer graph data from one format to another.
    
    parameters: 
        para: class, Parameters object. (more info please see the developer documentation) .
        outType: str, the type you want to transfer.
    
    return: 
        None, no return.
    """

    logger.info("entering transfer func.")

    if(para.graphType == 'CSR'):
        if(outType == 'matrix'):
            para.n, para.m, para.matrix = CSR2Matrix(para.CSR)
        elif(outType == 'edgeSet'):
            para.n, para.m, para.edgeSet = CSR2edgeSet(para.CSR)
        else:
            raise Exception("can not tranfer graph type to an undefined type")

    elif(para.graphType == 'matrix'):
        if(outType == 'CSR'):
            para.n, para.m, para.CSR = matrix2CSR(para.matrix)
        elif(outType == 'edgeSet'):
            para.n, para.m, para.edgeSet = matrix2edgeSet(para.matrix)
            para.directed = 1
        else:
            raise Exception("can not tranfer graph type to an undefined type")

    elif(para.graphType == 'edgeSet'):
        if(outType == 'matrix'):
            para.n, para.m, para.matrix = edgeSet2Matrix(para.edgeSet)
        elif(outType == 'CSR'):
            para.n, para.m, para.CSR = edgeSet2CSR(para.edgeSet)
        else:
            raise Exception("can not tranfer graph type to an undefined type")

    else:
        raise Exception("can not tranfer graph type from an undefined type")

    para.graphType = outType