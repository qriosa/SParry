from calc import calc
import numpy as np
from pretreat import read


def MemoryMatrix():

	print("="*10 + "This is function MemoryMatrix" + "="*10 + "\n")

	matrix = np.array([[0,1,2,3],[1,0,2,3],[2,2,0,4],[3,3,4,0]], dtype = np.int32) # the data of the adjacency matrix

	print("The matrix data is:")
	print(matrix)

	graph = read(matrix = matrix, # the data passed in is the adjacency matrix
	             method = "dij", # the calculation method uses Dijkstra
	             detail = True # record the details of the graph
	            ) # process the graph data

	res = calc(graph = graph, # class to pass in graph data
	           useCUDA = True, # use CUDA acceleration
	           srclist = 0, # set source to node 0
	           )

	print("The dist is:")
	print(res.dist) # output shortest path



def MemoryCSR():

	print("="*10 + "This is function MemoryCSR" + "="*10 + "\n")

	CSR = np.array([np.array([0, 2, 3, 4, 4]), 
	                np.array([1, 2, 3, 1]), 
	                np.array([1, 3, 4, 5])]) # simulation already has CSR format is graph data

	print("The CSR data is:")
	print(CSR)

	graph = read(CSR = CSR, # The data type passed in is CSR
	             method = "delta", # The algorithm used is delta-stepping
	             detail = True) # record the details of the graph

	res = calc(graph = graph, # the incoming graph data class
	           useCUDA = True, # use CUDA parallel acceleration
	           srclist = 0) # source point is node 0

	print("The dist is:")
	print(res.dist) # calculated shortest path

	print(res.display()) # print related parameters


def MemoryEdgeSet():
	
	print("="*10 + "This is function MemoryEdgeSet" + "="*10 + "\n")

	# simulation already has edgeSet format is graph data
	edgeSet = [[0, 0, 2, 1], # start point of each edge
	           [1, 3, 1, 3], # the end point of each edge
	           [1, 2, 5, 4]] # weights of each edge

	print("The edgeSet data is:")
	print(edgeSet) 

	graph = read(edgeSet = edgeSet, # the incoming graph data is edgeSet
	             detail = True) # need to record the data in the graph

	# calculated shortest path
	res = calc(graph = graph, # the incoming graph data class
	           useCUDA = False, # sse CPU serial computation
	           srclist = 0) # source point is node 0

	print("The dist is:")
	print(res.dist)

	print(res.display()) # print related parameters


if __name__ == '__main__':

	# In Memory-matrix
	MemoryMatrix()
	# In Memory-CSR
	MemoryCSR()
	# In Memory-edgeSet
	MemoryEdgeSet()
