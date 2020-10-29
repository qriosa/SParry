# created by wenake 2020.10.16
from utils.settings import INF
import numpy as np

def CSR2Matrix(CSR):
	"""
	function: transfer CSR graph to matrix graph.
	
	parameters: CSR graph.
	
	return: matrix graph.

	may I ask user-sama input a correct CSR format graph data please? thanks
	"""
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
	function: transfer CSR graph to edgeSet graph.
	
	parameters: CSR graph.
	
	return: edgeSet graph.
	"""
	V = CSR[0]
	E = CSR[1]
	W = CSR[2]
	n = len(V)-1
	edgeSet=[]
	for u in range(n):
		for ind in range(V[u],V[u+1]):
			edgeSet.append((u,E[ind],W[ind]))
	return np.int32(n), np.int32(len(E)), np.array(edgeSet,dtype=np.int32)

def matrix2CSR(mat):
	"""
	function: transfer matrix graph to CSR graph.
	
	parameters: matrix graph.
	
	return: CSR graph.
	"""
	n = len(mat)
	V = [0 for i in range(n)]
	E = []
	W = []
	for u in range(n):
		for v in range(n):
			w=mat[u][v]
			if(w < INF):
				V[u]=V[u]+1
				E.append(v)
				W.append(w)
	return np.int32(n),np.int32(len(E)),[np.array(V,dtype=np.int32), np.array(E,dtype=np.int32), npp.array(W,dtype=np.int32)]

def matrix2edgeSet(mat):
	"""
	function: transfer matrix graph to edgeSet graph.
	
	parameters: matrix graph.
	
	return: edgeSet graph.
	"""
	n = len(mat)
	edgeSet = []
	for u in range(n):
		for v in range(n):
			w=mat[u][v]
			if(w < INF):
				edgeSet.append([u,v,w])
				
	return np.int32(n),np.int32(len(edgeSet)),np.array(edgeSet,np.int32)

def edgeSet2Matrix(edgeSet):
	"""
	function: transfer edgeSet graph to matrix graph.
	
	parameters: edgeSet graph.
	
	return: matrix graph.	
	"""
	m = len(edgeSet)
	for item in edgeSet:
		u,v,w=item[0],item[1],item[2]
		n=max(n,u)
		n=max(n,v)
	mat = [ [INF for i in range(n)] for i in range(n)]
	for item in edgeSet:
		u,v,w=item[0],item[1],item[2]
		mat[u][v]=w
	return np.int32(n),np.int32(m),np.array(mat,dtype=np.int32)

def edgeSet2CSR(edgeSet):
	"""
	function: transfer edgeSet graph to CSR graph.
	
	parameters: edgeSet graph.
	
	return: CSR graph.
	"""
	n,m,mat=edgeSet2Matrix(edgeSet)
	return matrix2CSR(mat)

def transfer(para, outType):
	"""
	function: transfer graph data from one format to another.
	
	parameters: 
		para: a parameters class. (more info please see the developer documentation) .
		outType: the type you want to transfer.
	
	return: None.
	"""
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
			para.n, para.m, para.matrix = edgeSet2Matrix(para.edgeSet, para.directed)
		elif(outType == 'CSR'):
			para.n, para.m, para.CSR = edgeSet2CSR(para.edgeSet, para.directed)
		else:
			raise Exception("can not tranfer graph type to an undefined type")

	else:
		raise Exception("can not tranfer graph type from an undefined type")

	para.graphType = outType