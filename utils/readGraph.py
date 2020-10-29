from time import time

from classes.graph import Graph

def read(filename = 'data.txt'):
	"""
	function:
		read graph from file, and shape to a Graph object.
	
	parameters:
		filename: the graph data file name.
	
	return:
		a Graph object.
	"""

	start = time()
	
	g = Graph(filename)

	end = time()

	g.msg += ('用时 t = %.3f s \n' % (end - start))

	return g
