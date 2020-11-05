# [SPoon](https://github.com/LCX666/SPoon)

![image](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_logo.png)

**SPoon** is a shortest path calc tool using some algorithms with cuda to speedup.

It's **developing**.



------

`spoon` is a **shortest path calculation toolkit**, the main shortest path algorithms, including `Dijkstra`, `Bellman-Ford`, `Delta-Stepping`, and `Edge-Threads`, are encapsulated. It also provides **a parallel accelerated version based on CUDA is provided** to improve development efficiency.

At the same time, it can divide the graph data into parts, and solve it more quickly than using the CPU when the graph is too large to put it in the GPU directly.



## Installation

### Environment & Dependence

**window：**

> python 3.6/3.7
>
> pycuda 2019.1.2
>
> numpy 1.18.2
>
> CUDA Version  10.2.89
>
> cudnn 7.6.5
>
> networkx 2.4
>
> cyaron 0.4.2
>
> logging 0.5.1.2

**linux**

> python 3.6
>
> pycuda ?
>
> numpy 1.19.2
>
> CUDA Version 10.1.243
>
> cudnn ?

### Installation

Download the file package directly and run the `calc` interface function in the main directory.

**It's not not a release version currently, so it cannot be installed with pip, and the development structure is not yet perfect. **





## flow chart

![image](https://raw.githubusercontent.com/LCX666/SPoon/main/chart.svg)



## Quick start tutorial

This section is an introduction to help beginners of `SPoon` get started quickly.

### step1. cd to the current root directory

```powershell
cd XXX/spoon/
```



### step2. Import calculation interface

- When your graph data is **in memory** and **meets the data requirements**, you can import your data as follows to quickly calculate the results.

```python
>>> from calc import calc
>>> import numpy as np
>>>
>>> matrix = np.array([[0,1,2,3],[1,0,2,3],[2,2,0,4],[3,3,4,0]], dtype = np.int32) # 邻接矩阵是数据
array([[0, 1, 2, 3],
       [1, 0, 2, 3],
       [2, 2, 0, 4],
       [3, 3, 4, 0]])
>>>
>>> result = calc(graph = matrix, # Graph data
... graphType = 'matrix', # Graph format
... srclist = 0, # Calculate the shortest path of a single source. The source point is node 0
... useCUDA = True) # Use CUDA for acceleration 
>>>
>>> result.dist
array([0, 1, 2, 3])
>>>
>>> result.display()
'\n计算方法\tmethod = dij, \n使用CUDA\tuseCUDA = True, \n源点列表\tsrclist = 0, \n问题类型\tsourceType = SSSP, \n记录路 径\tpathRecord = False, \n\n结点数量\tn = 4, \n无向边数量\tm = 15, \n最大边权\tMAXW = 4, \n计算用时\ttimeCost = 0.009 sec'
>>>
>>> result.drawPath() # The red line indicates that this edge is on the shortest path; the orange point is the source point; the arrow indicates the direction of the edge; and the number on the edge indicates the edge weight.
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103113127.png" alt="image-20201104103113127"  />



- You can also pass **in a file** to calculate the shortest path when your graph data is stored in a file and **meets the data requirements**.

```python
>>> from calc import calc
>>> import numpy as np
>>> 
>>> filename = 'test.txt'
>>> 
>>> result = calc(graph = filename, # Graph
... graphType = 'edgeSet', # Graph format
... srclist = 0, # Calculate the shortest path of a single source. The source point is node 0
... useCUDA = True) # Use CUDA for acceleration
>>>
>>> result.dist
array([ 0,  9,  6, 13])
>>>
>>> result.display()
'\n结点数量\tn = 4, \n无向边数量\tm = 8, \n最大边权\tMAXW = 28, \n最大度\tdegree(0) = 7, \n最小度\tdegree(3) = 1, \n用时 t = 0.000 s \n\n计算方法\tmethod = dij, \n使用CUDA\tuseCUDA = True, \n源点列表\tsrclist = 0, \n问题类型\tsourceType = SSSP, \n记录路径\tpathRecord = False, \n\n结点数量\tn = 4, \n无向边数量\tm = 8, \n最大边权\tMAXW = 25, \n计算用时\ttimeCost = 0.0 sec'
>>>
>>> result.drawPath() # The red line indicates that this edge is on the shortest path; the orange point is the source point; the arrow indicates the direction of the edge; and the number on the edge indicates the edge weight.
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103213127.png" alt="image-20201104113340700" style="zoom: 67%;" />



------



## Interface

### Version Queries

This tool is still in the initial development iterative version, and the pip station has not been uploaded.



### Error Reporting

All error reports in the software will be reported with python error messages, and hint sentences are embedded in some places where errors are expected.



### Calculation Interface

#### func calc

This method is the only calculation interface of the software. By calling this method, the shortest path of the graph can be directly calculated. See ([func calc](https://github.com/LCX666/SPoon/blob/main/tutorials.md#func-calc-1)) for more information. 

------



## Data Type & Specification

1. The current version of this tool only supports data with 32-bit integers. Therefore, the edge weights, the number of points and the number of edges must be within the range of what a 32-bit integer can express. Also the calculated shortest path result value should not be outside the range of the 32-bit integer. One explanation for this is that most current data problems can be solved within the range of 32-bit integers. The second is that most non-professional NVIDIA GPUs are neutered for 64-bit and therefore do not support 64-bit data.

2. In this tool, all graphs are numbered from node 0 by default. Therefore graphs numbered from node 1 should ignore the data from the first node and start with the second data. (At this point the number of nodes perceived in the tool will be 1 more than the true number of nodes in the graph, because the tool always thinks there is another node 0.



### Graph Data  Specification

The graph data received in the tool is available in **both file and memory formats** (three types: adjacency matrix (matrix), compressed adjacency matrix (CSR), and edgeSet array (edgeSet)).

The following is an example of a common graph:

![image-20201023091301096.png](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201023091301096.png)

#### File Format

The above graph, when stored in a file, should be in the following format.

```
4 4
0 1 1
0 2 3
1 3 4
2 1 5

```

- First row: 2 parameters
  - parameter 1, n = 4, this means that there are 4 nodes in this graph.
  - parameter 2, m = 4, indicates that the graph has 4 directional/non-directional edges.
    - If you set the `directed = False` in `func clac` ，which means the graph is undirected. The tool automatically converts the above unidirectional edges into bidirectional edges and automatically doubles the number of edges in the tool, since it uses two directed edges to represent a directionless edge.
    - If you set the `directed = True` in `func clac` ，which means the graph is directed. This tool will read the map strictly in one direction, one way.
    - Default parameter directed = False. The default graph is an undirected graph.
- Second row to fifth row: 3 integers, representing an edge.
  - The first parameter represents the start of the current edge.
  - The second parameter represents the endpoint of the current side.
  - The third parameter represents the side weight of the current edge.
- Sixth row(the last row): A blank line.
  - **End of document with a blank line**



#### Adjacency Matrix

The adjacency matrix is an n × n two-dimensional array to represent a graph, any one element of the matrix can represent an edge. That is, `matrix[i][j] = w` means that there is at least one edge in the graph from node `i` to node `j`, and its edge weight is `w`.

Since this tool is a shortest-path calculator, more strictly speaking, `matrix[i][j] = w` should be the side weight of the shortest of all edges of the representation from node `i` to node `j` is `w`.

Since this tool calculates in the range of a 32-bit integer, positive infinity in this tool is a number that is very close to the maximum number that can be represented by a 32-bit integer. The positive infinity of this tool can be returned with `func: calc.INF`.

The above figure translates into an adjacency matrix as follows：

```python
In [1]: matrix
    
Out[1]:
array([[0, 1, 3, 2139045759],
       [2139045759, 0, 2139045759, 4],
       [2139045759, 5, 0, 2139045759],
       [2139045759, 2139045759, 2139045759, 0]])
```



#### CSR

CSR is the main storage and computing method of this tool, which saves memory space in most cases compared to the adjacency matrix storage method. It also saves memory space when using GPU to accelerate computation. It is represented by three one-dimensional arrays: `V`, `E` and `W`. In this tool, these three arrays are combined in the order of `V`, `E` and `W` to form CSR.

The `V` array is the starting subscript of the first edge of each node in the `E` array, and its dimension is the number of points in the diagram, but for the convenience of calculation, a virtual node is usually added at the end to determine whether the end is reached or not. So the dimension must be strictly virtual nodes in this tool.

The `E` array is a record of which node each edge ends at. Its dimension is therefore strictly the number of (directed) edges in the graph, and an undirected edge is represented by two directed edges in this tool.

The `W` array is a record of the edge weights of each edge corresponding to the `E` array, so its dimension is also strictly the number of edges in the graph.

The above figure translates into CSR as follows.

```python
In [1]: CSR
    
Out[1]:
array([array([0, 2, 3, 4, 4]), 
       array([1, 2, 3, 1]), 
       array([1, 3, 4, 5])]

In [2]: V
Out[2]: array([0, 2, 3, 4, 4])

In [3]: E
Out[3]: array([1, 2, 3, 1])

In [4]: W
Out[4]: array([1, 3, 4, 5])
```

**A virtual node at the end of V to define the boundary.**



#### edgeSet

edgeSet is a ternary list where the first list represents the starting point of all edges, the second list represents the end point of all edges, and the third list represents the edge weights of all edges.

The above figure translates into an edgeSet array as follows

```python
In [1]: edgeSet
Out[1]: [[0, 0, 2, 1], [1, 3, 1, 3], [1, 2, 5, 4]]
```

**If it is an undirected edge it needs to be represented as two directed edges in the above list and then passed in.**



### Constants

INF: positive infinity in this tool. it's 2139045759

------



## Func

### func calc()

#### Function 

This function is an interface function of `SPoon`. Through this function, users can pass in their own graph data and some necessary parameters to calculate the shortest path of the graph.

#### Structure

```python
def calc(graph = None, graphType = None, method = 'dij', useCUDA = True, directed = False, pathRecordBool = False, srclist = None, block = None, grid = None):
    """
    function: 
        a calculate interface.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: str, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: bool, use CUDA to speedup or not.
        directed: bool, directed or not. only valid in read graph from file.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.

    return:
        class, Result object. (more info please see the developer documentation).  
    """
    return result
```

#### Parameters

This method is an interface function, and the meaning of each parameter is as follows:

- graph, Graph data, required, the graph data or graph data storage file that needs to calculate the shortest path. See figure [data specification](https://github.com/LCX666/SPoon/blob/main/tutorials.md#data-type--specification).
  1. In the case of graph data in memory, data in three formats are supported: adjacency matrix (matrix), compressed adjacency matrix (CSR), and edge set array (edgeSet).
  2. If it is a file of graph data, it means the file name of the graph file that meets the **graph data specification**.
- graphType, the type of the incoming graph data, required. str. It can only be of the following three types:

  1. matrix: Indicates that the incoming data is an adjacency matrix.
  2. CSR: Indicates that the incoming data is a compressed adjacency matrix.
  3. edgeSet: Indicates that the incoming data is an edge set array.
- method, the shortest path calculation method used, the default is'dij'. str. It can only be of the following types:
  1. dij: Indicates to use the `dijkstra` algorithm to solve the shortest path.
  2. spfa: Means to use the `bellman-ford` algorithm to solve the shortest path.
  3. delta: Means to use the `delta-stepping` algorithm to solve the shortest path.
  4. edge: Indicates that fine-grained edges are used to calculate the shortest path.
- useCUDA, whether to use CUDA acceleration, the default is True. bool. It can only be of the following types:
  1. True: means to use CUDA for acceleration.
  2. False: Indicates that only the CPU is used for serial calculation.
- directed, whether the graph is directed, the default is False. bool. It can only be of the following types:
  1. True: indicates that the graph is a directed graph.
  2. False: indicates that the graph is undirected.
- pathRecordBool, whether to record the path of the shortest path, the default is False. bool. It can only be of the following types:
  1. True: Indicates the path of the shortest path needs to be recorded.
  2. False: indicates that the path does not need to be calculated, only the value of the path is required.
- srclist, the collection of source points, the default is None. int/list/None. It can be of the following three types:
  1. int: An integer, representing a node in the graph as the source point for the shortest path calculation.
  2. List: A list, each element in the list is int, which means that these points in the graph are source points, and the corresponding shortest paths need to be calculated separately.
  3. None: A null value, indicating the shortest path of all sources in the calculation graph.
- block, the structure of thread block, the default program will automatically set. It can only be a triple `(x, y, z)`.
- grid, the structure of thread block, the default program will automatically set. It can only be a two-tuple `(x, y)`.

#### Return

The return value is an instance of `class Result`. Please refer to [class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result) for details.



### func dispatcher()

#### Function

The task scheduling function judges the direction of the program according to the user's input data, adjusts the operation of the program and checks the legality of the parameters.

#### Structure

```python
def dispatch(graph, graphType, method, useCUDA, pathRecordBool, srclist, msg, block, grid):
    """
    function: 
        schedule the program by passing in parameters.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: str, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: bool, use CUDA to speedup or not.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        msg: the info of the graph.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return:
        class, Result object. (more info please see the developer documentation). 
    """

   
    return result
```

#### Parameters

This method is a transfer-in function inherited by the interface function. The meaning of each parameter is consistent with the interface function. The meaning of each parameter is as follows:

- graph, graph data, required, graph data or graph data storage file that needs to calculate the shortest path. See figure data specification.
  1. If it is the graph data in the memory, three formats of data are supported: adjacency matrix (matrix), compressed adjacency matrix (CSR), edge set array (edgeSet).
  2. If it is a graph data file, it means the file name of the graph file that meets the **Graph Data Specification**.
- graphType, the type of the incoming graph data, required. str. It can only be of the following three types:

  1. Matrix: Indicates that the incoming data is an adjacency matrix.
  2. CSR: Indicates that the incoming data is a compressed adjacency matrix.
  3. edgeSet: Indicates that the incoming data is an edge set array or a file.
- method, the shortest path calculation method used, the default is'dij'. str. It can only be of the following types:
  1. dij: means to use the `dijkstra` algorithm to solve the shortest path.
  2. spfa: Means to use the `bellman-ford` algorithm to solve the shortest path.
  3. delta: Means to use the `delta-stepping` algorithm to solve the shortest path.
  4. edge: Indicates that fine-grained edges are used to calculate the shortest path.
- useCUDA, whether to use CUDA acceleration, the default is True. bool. It can only be of the following types:
  1. True: means to use CUDA for acceleration.
  2. False: Indicates that only the CPU is used for serial calculation.
- pathRecordBool, whether to record the path of the shortest path, the default is False. bool. It can only be of the following types:
  1. True: Indicates the path of the shortest path needs to be recorded.
  2. False: indicates that the path does not need to be calculated, only the value of the path is required.
- srclist, the collection of source points, the default is None. int/list/None. It can be of the following three types:
  1. int: An integer, representing a node in the graph as the source point for the shortest path calculation.
  2. List: A list, each element in the list is int, which means that these points in the graph are source points, and the corresponding shortest paths need to be calculated separately.
  3. None: A null value, indicating the shortest path of all sources in the calculation graph.
- block, the structure of thread block, the default program will automatically set. It can only be a triple `(x, y, z)`.
- grid, the structure of thread block, the default program will automatically set. It can only be a two-tuple `(x, y)`.

#### Return

The return value is an instance of `class Result`. Please refer to [class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result) for details.



### func transfer()

#### Function

Standardize the graph data input by the user, transform the graph format, and calculate some necessary parameters required by the subsequent algorithm.

#### Structure

```python
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
	
```

#### Parameters

- para, the parameter class passed between functions. class parameter. The type is fixed.
- outType, the type of output you want to convert. str. Can be of the following types:
  1. matrix, represents the adjacency matrix.
  2. CSR, stands for compressed adjacency matrix.
  3. edgeSet, represents the edge set array.

#### Return

None



### func judge()

#### Function

Determine whether the current graph data can use GPU acceleration under the current version of the algorithm and the current hardware conditions, and judge whether it is necessary to start the graph segmentation algorithm.

#### Structure

```python
def judge(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object.
    
    return:
        bool, True/False. (more info please see the developer documentation).
```

#### Parameters

- para, the parameter class passed between functions. class parameter. The type is fixed.

#### Return

- bool, whether to enable graph segmentation.
  1. True: indicates that the graph segmentation algorithm needs to be enabled.
  2. False: The graph segmentation algorithm does not need to be enabled.



### func read()

#### Function

Read the graph from the file and convert it to CSR format or edgeSet format. And get some characteristic information of the graph.

#### Structure

```python
def read(filename = 'data.txt', directed = False):
    """
    function:
        read graph from file, and shape to a Graph object.
    
    parameters:
        filename: str, the graph data file name.
    
    return:
        class, Graph object.
    """
```

#### Parameters

- filename, the name of the file to be read, required. str.
- directed, whether the graph is directed, the default is False. bool. It can only be of the following types:
  1. True: indicates that the graph is a directed graph.
  2. False: indicates that the graph is undirected.

#### Return

The return value is an instance of `class Class`. Please refer to[class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result)for details.



### func draw()

#### Function

Draw the shortest path diagram of the first source based on the shortest path.

#### Structure

```python
def draw(path, n, s, graph, graphType):
    """
    function: 
        use path to draw a pic.

    parameters:
        path: list, must,  about precursors of each vertex in each problem.
        n: int, must, the number of vertices.
        s: int , must, the source vertex.
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
    
    return: 
        None, no return. 
```

#### Parameters

- path, the precursor array of the shortest path, required. list.
- n, the number of nodes in the figure, required. int.
- s, the number of the source point, required. int.
- graph, graph data, required, graph data for calculating the shortest path. See figure data specification.
- graphType, the type of the incoming graph data, required. str. It can only be of the following three types:
  1. Matrix: Indicates that the incoming data is an adjacency matrix.
  2. CSR: Indicates that the incoming data is a compressed adjacency matrix.
  3. edgeSet: Indicates that the incoming data is an edge set array.

#### Return

None



### func generate()

#### Function

Generate a connected graph containing the specified number of points, number of edges, and edge weights.

#### Structure

```python
def generate(filename = 'data0.txt', n = 1000, m = 30000, l = 1, r = 12):
    """
    function: 
        generate a random graph to file. (more info please see the developer documentation). 

    parameters:
        filename: str, the filename of file to save the graph.
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        l: int, the min value of a edge.
        r: int, the max value of a edge.
    
    return: 
        None, no return.
```

#### Parameters

- filename, the name of the file where the generated image is stored, the default value is'data0.txt'. str.
- n, the number of nodes in the generated graph, the default is 1000. int.
- m, the number of edges of the generated graph, the default is 30000. int.
- l, the lower bound of the edge weight in the figure, the default is 1. int.
- r, the upper bound of the edge weight in the figure, the default is 12. int.

#### Return

None



### func check()

#### Function

Check whether the two data are equal.

#### Structure

```python
def check(data1, data2, name1 = 'data1', name2 = 'data2'):
    """
    function: 
        to check the data is equal or not.

    parameters:
        data1: numpy.ndarray, must, first data.
        data2: numpy.ndarray, must, second data.
        name1: str, the name of data1.
        name2: str, the name of data2.
    
    return: 
        str, the EQUAL or NOT.
    """
```

#### Parameters

- data1, the first data to be detected, required. list.
- data2, the second data to be tested, required. list.
- name1, the first data name, the default is'data1'. str.
- name2, the second data name, the default is'data2'. str.

#### Return

str, equal or not equal



### func [apsp, sssp, mssp].'[dij, delta, edge, spfa]'_[cpu, gpu]

This tool encapsulates shortest path algorithms such as `Dijkstra`, `Bellman-Ford`, `Delta-Stepping`, ʻEdge-thread`. Support single source at the same time. Multi-source and full-source algorithms, CPU serial calculation versions and CUDA accelerated versions belong to multiple methods in multiple files, but have similar parameters and return values.

#### Function

Through the incoming parameters, the algorithm actually calculates the shortest path to solve the problem. The method name is different, the parameters are slightly different, and the calculation method enabled is also slightly different.

#### Structure

```python
def dijkstra(CSR, n, s, pathRecordingBool = False):
	"""
	function: use dijkstra algorithm in GPU to solve the SSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        s: the source list, can be number.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) . 
	"""
	...
```

#### Parameters

- CSR, CSR graph data.
- n, the number of nodes in the figure, required. int.
- s, the number of the source point, required. int.
- pathRecordBool, whether to record the path of the shortest path, the default is False. bool. It can only be of the following types:
  1. True: Indicates the path of the shortest path needs to be recorded.
  2. False: indicates that the path does not need to be calculated, only the value of the path is required.

#### Return

The return value is an instance of `class Result`. Please refer to [class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result) for details.

------



## Class

### class Result

#### Function

This class is a unified return class for tools, and the results of all calculation methods will be returned as instances of this class.

#### Structure

```python
class Result(object):
	"""
    function: 
        to store the result of different algorithm.

    parameters:
        dist: list, the shortest path distance answer for algorithm.
        timeCostNum: float, a float data of time cost of getting the answer, so it can use to calculate.
        timeCost: str, a str data of time cost of getting the answer.
        memoryCost: str, memory cost of getting the answer.
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        msg: str, the info of the graph.
    
    method:
        display: show the detail of this calculation.
        drawPath: draw the path from vertices to the sources.
        calcPath: calc the path through the graph and dist.
    
    return: Result object.
	"""
	def __init__(self, 
                dist = None, 
                timeCost = None, 
                memoryCost = None, 
                graph = None,
                graphType = None,
                msg = ""):
		"""
		"""
		
	def display(self):
        """
        function: 
            show the detail of the graph, parameters and calc time.

        parameters:
            None, but 'self'.
        
        return: 
            str, the msg info.   
        """
        
    def drawPath(self):
        """
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.     
        """
        
	def calcPath(self,CSR=None,matrix=None,edgeSet=None):
		"""
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.		
		"""
        

```

#### Parameters

- dist, the array of the shortest path calculation. np.array.
- timeCost, the time taken to calculate the shortest path. float.
- ~~memoryCost, the memory usage during calculation. Not currently.~~
- graph, graph data, required, graph data for calculating the shortest path. See figure data specification.
- graphType, the type of the incoming graph data, required. str. It can only be of the following three types:
  1. Matrix: Indicates that the incoming data is an adjacency matrix.
  2. CSR: Indicates that the incoming data is a compressed adjacency matrix.
  3. edgeSet: Indicates that the incoming data is an edge set array.
- msg, prompt message, built by default.

#### Attributes

- dist, the array of the shortest path calculation. np.array. The details are as follows:
  1. If calculating the single-source shortest path, it is a one-dimensional array, `dist[i]` means the shortest distance from the source point to the node ʻi` in the current single-source shortest path problem.
  2. If calculating the multi-source shortest path, it is a two-dimensional array composed of multiple one-dimensional arrays, `dist[i][j]` means that in the `i`th source single-source shortest path problem, the `th The shortest distance from i` sources to node `j`.
  3. If the shortest path of all sources is calculated, it is a two-dimensional array of `n × n`, `dist[i][j]` means that in the single source shortest path problem of source point ʻi`, source point ʻi `The shortest distance to node `j`.
- path, the path of the shortest path. np.array. The details are as follows:
  1. If calculating the single-source shortest path, it is a one-dimensional array, `path[i]` means that in the current single-source shortest path problem, the predecessor node of node ʻi`, that is, the source point needs to be from the node `path[i]` reaches the node ʻi`.
  2. If calculating the multi-source shortest path, it is a two-dimensional array composed of multiple one-dimensional arrays. `path[i][j]` means that in the `i`th source single-source shortest path problem, the `th i` source points need to reach the node `j` from the node `path[i][j]`.
  3. If the shortest path of all sources is calculated, it is a two-dimensional array of `n × n`, `path[i][j]` means that in the single source shortest path problem of source point ʻi`, source point ʻi `Need to go from node `path[i][j]` to reach node `j`.
- timeCost, time spent. float. Indicates the time it takes to calculate the shortest path problem, in seconds.
- memoryCost, space cost. int. Indicates the space spent to calculate the shortest path problem, in bytes.
- graph, graph data, required, graph data for calculating the shortest path. See figure data specification.
- graphType, the type of the incoming graph data, required. str. It can only be of the following three types:
  1. Matrix: Indicates that the incoming data is an adjacency matrix.
  2. CSR: Indicates that the incoming data is a compressed adjacency matrix.
  3. edgeSet: Indicates that the incoming data is an edge set array.
- msg, some parameter information in the calculation process, built by default. str.

#### Methods

- display, func, display the calculation details `msg` information to the user.
  1. parameters, none.
  2. Return value, msg calculation information. str.
- drawPath, func, visually draw the calculated shortest path.
  1. parameters, none.
  2. Return value, none.
- calcPath, func, reproduce the path from the dist array and graph. Assign directly to the `path` attribute.
  1. parameters, none.
  2. Return value, none.



### class parameter

#### Function

This class is used by the tool's internal parameter transfer. It records various parameters related to algorithms and has nothing to do with users.

#### Structure

```python
class parameter(object):
	"""
    function: 
        transfer the parameters in the functions.

    parameters: 
        None, but 'self'.

    attributes:
        BLOCK: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        GRID: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        useCUDA: bool, use CUDA to speedup or not.
        CSR: tuple, a 3-tuple of integers as (V, E, W) about the CSR of graph data. (more info please see the developer documentation).
        matrix: matrix, as (n,n), about adjacency matrix of graph data.
        edgeSet: tuple, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
        graphType: str, type of graph. [edgeSet, matrix, CSR].
        method: str, the algorithm. [dij, delta, spfa, fw, edge]
        srclist: list/int, the source of shortest path problem.
        sourceType: str, the type of the problem. [APSP, SSSP, MSSP]
        pathRecordingBool: bool, record the path or not.
        delta: int, the delta of delta-stepping algorithm.
        MAXN: int, the max value of the edges.
        MAXU: int, the vertex has the maxOutDegree.
        maxOutDegree: int, the max out degree of the graph.
        part: int, the number of the edges that will put to GPU at a time.(divide algorithm)
        streamNum: int, the number of streams used.
        msg: str, the info of the graph.
    
    method:
        None, but init.
            
    return 
        class, Parameter object.
	"""
	def __init__(self):
		
		...
```

#### Parameters

None

#### Attributes

- BLOCK, the triple `(x,y,z)` represents the structure of the thread in the calculation process. tuple. Optional.
- GRID, the two-tuple `(x,y)` represents the structure of the block in the calculation process. tuple. Optional.
- n, the number of nodes in the graph. int.
- m, the number of directed edges in the graph. int.
- useCUDA, whether to use CUDA acceleration. bool.
- CSR, CSR graph data.
- matrix, adjacent matrix graph data.
- edgeSet, edge set array graph data.
- graphType, the type of graph data passed in. str. It can only be of the following three types:
  1. Matrix: Indicates that the incoming data is an adjacency matrix.
  2. CSR: Indicates that the incoming data is a compressed adjacency matrix.
  3. edgeSet: Indicates that the incoming data is an edge set array or a file.
- method, the shortest path calculation method used. str. It can only be of the following types:
  1. dij: means to use the `dijkstra` algorithm to solve the shortest path.
  2. spfa: Means to use the `bellman-ford` algorithm to solve the shortest path.
  3. delta: Means to use the `delta-stepping` algorithm to solve the shortest path.
  4. edge: Indicates that fine-grained edges are used to calculate the shortest path.
- srclist, a collection of source points. int/list/None. It can be of the following three types:
  1. int: An integer, representing a node in the graph as the source point for the shortest path calculation.
  2. List: A list, each element in the list is int, which means that these points in the graph are source points, and the corresponding shortest paths need to be calculated separately.
  3. None: A null value, indicating the shortest path of all sources in the calculation graph.
- sourceType, the type of problem to be solved. str. It can be of the following three types:
  1. APSP: Shortest path for all sources
  2. MSSP: Multi-source shortest path
  3. SSSP: Single source shortest path
- pathRecordBool, whether it is necessary to record the path of the shortest path. bool. It can only be of the following types:
  1. True: Indicates the path of the shortest path needs to be recorded.
  2. False: indicates that the path does not need to be calculated, only the value of the path is required.
- delta, the delta value used in the delta-stepping algorithm. int.
- MAXW, the maximum edge weight in the graph. int
- MAXU, the point number of the largest degree in the figure. int.
- maxOutDegree, the largest out degree in the graph. int.
- part, the number of edges copied into the GPU at a time in the part-graph algorithm. int.
- streamNum, the number of streams enabled in the sub-picture multi-stream. int.
- msg, the prompt message of the calculation process. str.

#### Methods

None



### class device

#### Function

This class is used by this tool to obtain the graphics card information of the device. The functions are implemented through `pycuda` and `pynvml`.

#### Structure

```python
class Device(object):
	"""
    function: 
        get the GPU device infomation, get the type and attributes.

    parameters: 
        None, but 'self'.

    attributes:
        device: class, a pyCUDA device class object.
        CUDAVersion: str, the version of CUDA.
        driverVersion: int, the version of CUDA driver.
        deviceNum: int, the number of valid GPU device.
        deviceName: str, the name of the device.
        globalMem: int, the max number of the global memory.
        sharedMem: int, the max number of the shared memory.
        processNum: int, the number of processors.
        freeMem: int, the bytes of free memory.
        temperature: int, the temperature of the device.
        powerStstus: the power ststus of the device.

    method:
        get_device_type: get the type of the device.
        get_number_of_device: get the number of the device.
        get_version: obtain the version of CUDA against which PyCuda was compiled.
        get_driver_version: obtain the version of the CUDA driver on top of which PyCUDA is running. 
        getDeviceInfo: obtain the device infomation include: 'freeMemory, totalMemory, memoryUsed, temperature, powerStstus'.
        get_attributes: the pycuda's get_attributes.

    return 
        class, Device object.
	"""
	def __init__(self):
		...
```

#### Parameters

None

#### Attributes

- device, the Device class of pyCUDA. object.
- CUDAVersion, the version of CUDA. str.
- driverVersion, the version of the driver. str.
- deviceName, the model of the GPU device. str.
- deviceNum, the number of CUDA devices. int.
- total, the total memory capacity. int.
- free, the capacity of free memory. int.
- used, the memory that has been used. int.
- temperature, the temperature of the device. str.
- powerStstus, the state of the power supply. str.

#### Methods

- getDeviceInfo, func, get the remaining information of the video memory that needs to be calculated for the sub-picture.
  1. parameters, none.
  2. Return value, none.



### class graph

#### Function

Converts user-supplied graph data into a graph class that saves information about the graph.

#### Structure

```python
class Graph(object):
    """
    function: 
        a graph class.

    parameters:
        filename: str, must, the graph data file. (more info please see the developer documentation).
        directed: bool, the graph is directed ot not.
    
    attributes:
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        CSR: tuple, a 3-tuple of integers as (V, E, W) about the CSR of graph data. (more info please see the developer documentation).
        src, des, w: tuple, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
        MAXW: int, the max weight of the edges.
        MINW: int, the min weight of the edges.
        MAXD: int, the max degree(In degree and Out degree) of all the vertices.
        MAXU: int, one of the vertices with the max degree.
        MIND: int, the min degree(In degree and Out degree) of all the vertices.
        MINU: int, one of the vertices with the min degree.
        degree: list, save the degree of each vertex.
        msg: str, the message about the read func.
        filename: str, the file name of the graph.
    
    method:
        read: read the graph from file.
        reshape: convert data to numpy.int32.

    return: 
        class, Graph object.
    """
    
    def __init__(self, filename = None, directed = False):
        """
        """
    
    def read(self):
        """
        function: 
            read the graph from file.
            only accept graphic data in edgeSet format and store it in memory in CSR/edgeSet format
            by the way, we wanna to specify the edgeSet format as a variable with 3 array src/des/weight which are consistent with every edge in graph

        parameters: 
            None, but 'self'.

        return:
            None, no return.
        """
        
    def reshape(self):
        
        """
        function: 
            convert data to numpy.int32.

        parameters: 
            None, but 'self'.

        return:
            None, no return.
        """        
```

#### Parameters

- filename, the name of the file to be read, the default is not to read the map. str.
- directed, whether the identifier is directed or not, default is False. bool.

#### Attributes

- n, the number of nodes in the graph. int.
- m, the number of vector edges in the graph. int.
- csr, csr graph data.
- matrix, adjacency matrix graph data.
- src, des, w, edge set array graph data.
- maxw, maximum edge weight. int. 7.
- minw, minimum edge weight. int. 8.
- MAXD, maximum degree. int. 9.
  MAXU, node number of maximum degree. int. 10.
  MIND, minimum degree. int. 11.
- minu, node number of minimum degree. int. 12. degree, number of each node.
- degree, degree of each node. list. 13.
- msg, message for calculation. str. 14.
- filename, the name of the file to be read. str.

#### Methods

- read, func, read the graph from the file.
  1. parameters, None.
  2. return, None.
- reshape, func, converts the data to a numpy format.
  1. parameters, None.
  2. return, None.







------

所谓工欲善其事必先利其器。

`SPoon` 是一个**最短路径计算工具包**，封装了诸如：`Dijkstra`, `Bellman-Ford`,`Delta-Stepping`, `Edge-Threads` 等主流的最短路径算法。它也提供了**基于CUDA的并行加速版本**，以提高开发效率。

同时本工具还封装了自动分图计算方法的 `dijkstra` 算法，可有效解决大规模图在GPU显存不足无法直接并行计算的问题。



## 安装

### 环境依赖

**window：**

> python 3.6/3.7
>
> pycuda 2019.1.2
>
> numpy 1.18.2
>
> CUDA Version  10.2.89
>
> cudnn 7.6.5
>
> networkx 2.4
>
> cyaron 0.4.2
>
> logging 0.5.1.2

**linux**

> python 3.6
>
> pycuda ?
>
> numpy 1.19.2
>
> CUDA Version 10.1.243
>
> cudnn ?

### 安装

直接下载文件包，即可在主目录中运行 `calc` 接口函数。

**目前不是发行版本，故不可pip安装，开发结构尚不是很完善。**



## 流程图

![image](https://raw.githubusercontent.com/LCX666/SPoon/main/chart.svg)



## 快速入门教程

本节是帮助 `SPoon` 新手快速上手的简介。

### step1. cd 到当前根目录

```powershell
cd XXX/spoon/
```



### step2. 导入计算接口

#### 内存数据

- 当您的图数据在**内存**中并**符合数据要求** 时，可以像下面一样导入您的数据，快速计算结果。

```python
>>> from calc import calc
>>> import numpy as np
>>>
>>> matrix = np.array([[0,1,2,3],[1,0,2,3],[2,2,0,4],[3,3,4,0]], dtype = np.int32) # 邻接矩阵是数据
array([[0, 1, 2, 3],
       [1, 0, 2, 3],
       [2, 2, 0, 4],
       [3, 3, 4, 0]])
>>>
>>> result = calc(graph = matrix, # 图
... graphType = 'matrix', # 图的格式
... srclist = 0, # 计算单源最短路径 源点是结点0
... useCUDA = True) # 使用CUDA加速
>>>
>>> result.dist
array([0, 1, 2, 3])
>>>
>>> result.display()
'\n计算方法\tmethod = dij, \n使用CUDA\tuseCUDA = True, \n源点列表\tsrclist = 0, \n问题类型\tsourceType = SSSP, \n记录路 径\tpathRecord = False, \n\n结点数量\tn = 4, \n无向边数量\tm = 15, \n最大边权\tMAXW = 4, \n计算用时\ttimeCost = 0.009 sec'
>>>
>>> result.drawPath() # 红色的线表示此条边在最短路径上；橙色的点为源点；箭头表示边的方向；边上的数字表示边权。
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103113127.png" alt="image-20201104103113127"  />

#### 文件数据

- 当您的图数据存储在**文件**中并**符合数据要求时**，您也可以传入文件来计算最短路径。

```python
>>> from calc import calc
>>> import numpy as np
>>> 
>>> filename = 'test.txt'
>>> 
>>> result = calc(graph = filename, # 图
... graphType = 'edgeSet', # 图的格式
... srclist = 0, # 计算单源最短路径 源点是结点0
... useCUDA = True) # 使用CUDA加速
>>>
>>> result.dist
array([ 0,  9,  6, 13])
>>>
>>> result.display()
'\n结点数量\tn = 4, \n无向边数量\tm = 8, \n最大边权\tMAXW = 28, \n最大度\tdegree(0) = 7, \n最小度\tdegree(3) = 1, \n用时 t = 0.000 s \n\n计算方法\tmethod = dij, \n使用CUDA\tuseCUDA = True, \n源点列表\tsrclist = 0, \n问题类型\tsourceType = SSSP, \n记录路径\tpathRecord = False, \n\n结点数量\tn = 4, \n无向边数量\tm = 8, \n最大边权\tMAXW = 25, \n计算用时\ttimeCost = 0.0 sec'
>>>
>>> result.drawPath() # 红色的线表示此条边在最短路径上；橙色的点为源点；箭头表示边的方向；边上的数字表示边权。
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103213127.png" alt="image-20201104113340700" style="zoom: 67%;" />

------



## 接口

### 版本信息

此工具尚在初始开发迭代版本，未上传pip站。



### 错误报告

工具中的一切错误报告都将以 python 的错误信息进行报告，并在一些预想到错误的地方，嵌入了提示语句。



### 计算接口

#### func calc

该方法是本工具的唯一计算接口，通过调用此方法可以直接计算得到图的最短路径。更多信息请参阅 [func calc](https://github.com/LCX666/SPoon/blob/main/tutorials.md#func-calc-3)。

------



## 数据类型与规范

1. 本工具中目前的版本只支持32位整型的数据。因此，边权、点的数量和边的数量都必须是在32位整型能够表述的范围内。同时计算出的最短路径结果值也不应该超出32位整型的范围。关于此，一个解释是：目前大部分的数据问题都可以在32位整型范围内解决。其次是大部分非专业的英伟达GPU都对64位进行了阉割，因此不支持64位的数据。
2. 本工具中，所有的图默认都从结点0开始编号。因此从结点1开始编号的图应该无视第一个结点的数据，从第二个数据开始。（此时工具中认为的图中的结点数会比真正的图中的结点数多1，因为工具始终认为还有一个0号结点。



### 图数据规范

工具中接收的图数据有**文件格式**和**内存格式**(三种类型：邻接矩阵(matrix)、压缩邻接矩阵(CSR)和边集数组(edgeSet))。

下图是一个普通的图例子：

![image-20201023091301096.png](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201023091301096.png)

#### 文件格式

上述图存储在文件中时，格式应该如下：

```
4 4
0 1 1
0 2 3
1 3 4
2 1 5

```

- 第一行：  2个参数
  - 第一个参数 n = 4 表示此图一共有4个结点。 
  - 第二个参数 m = 4  表示此图一共有**4**条**有向/无向**边。
    - 若在 `func clac` 中指定 `directed = False` 时，即指定为无向图，本工具会自动将上述单向边转化为双向边，同时工具中的边数会自动翻倍，即用两条有向边表示一条无向边。
    - 若在 `func clac` 中指定 `directed = True` 时，即指定为有向图，本工具会严格按照单向边方向读图。
    - 默认参数 `directed = False` 默认图为无向图。
- 第二行~第五行： 3个整数，表示一条边
  - 第一个参数  表示当前边的起点
  - 第二个参数 表示当前边的终点
  - 第三个参数 表示当前边的边权
- 第六行（最后一行）：一个空行
  - **以一个空行表示文件的结束**



#### 邻接矩阵

邻接矩阵即用一个n×n二维数组来表示一个图，矩阵中的任意一个元素都可以代表一条边。即 `matrix[i][j] = w` 表示在图中存在至少一条从结点 `i` 到结点 `j` 的边，其边权为 `w` 。

由于本工具是最短路径计算工具，更严格地说， `matrix[i][j] = w` 应该是表示的结点 `i` 到结点 `j` 的所有边中最短的边的边权是 `w` 。

由于本工具是在32位整型的范围中进行计算，因此，本工具中的正无穷是一个很接近32位整型能表示的最大数的一个数。可以通过 `func: calc.INF` 返回本工具的正无穷。

上转化为邻接矩阵如下：

```python
In [1]: matrix
    
Out[1]:
array([[0, 1, 3, 2139045759],
       [2139045759, 0, 2139045759, 4],
       [2139045759, 5, 0, 2139045759],
       [2139045759, 2139045759, 2139045759, 0]])
```



#### 压缩邻接矩阵

压缩邻接矩阵是本工具的主要存储和运算的方式，相较于邻接矩阵的存储方式可以在绝大多数情况下节约内存空间。同时在利用GPU进行加速计算时亦可以节约显存空间。其表示方式是三个一维数组： `V` 、 `E` 和 `W` ，在本工具中将上述三个数组按照 `V` 、 `E` 和 `W` 的顺序组合成 CSR 。

其中 `V` 数组是记录图中各个结点的第一条边的在 `E` 数组中的起始下标的，其维数是图中点的个数，但是为了计算的方便，通常会在末尾增加一个虚拟结点来判断是否到达末尾。因此其维数在本工具中必须是严格的包含了虚拟结点的。

`E` 数组是记录每一条边的终点是哪个结点。因此其维度是严格的图中的（有向）边的数目，本工具中通过两条有向边来表示一条无向边。

`W` 数组是记录与 `E` 数组对应的每一条边的边权，故其维度也是严格的图中的边的数目。

上转化为压缩邻接矩阵如下：

```python
In [1]: CSR
    
Out[1]:
array([array([0, 2, 3, 4, 4]), 
       array([1, 2, 3, 1]), 
       array([1, 3, 4, 5])]

In [2]: V
Out[2]: array([0, 2, 3, 4, 4])

In [3]: E
Out[3]: array([1, 2, 3, 1])

In [4]: W
Out[4]: array([1, 3, 4, 5])
```

**V 中在最后虚拟了一个结点，以确定边界。**



#### 边集数组

边集数组是一个列表，表中的每个元素都是表示一条边的三元组 `(u, v, w)` 即一条边的起点、终点和边权。

上图转化为边集数组如下：

```python
In [1]: edgeSet
Out[1]: [[0, 0, 2, 1], [1, 3, 1, 3], [1, 2, 5, 4]]
```

**如果是无向边需要在上述列表中表示成两条有向边再传入。**



### 常量

INF：工具中的正无穷，此处是 2139045759

------



## 函数 

### func calc()

#### 功能 

该函数是 Spoon 的接口函数，通过该函数，用户可以传入自己的图数据和一些必要的参数从而计算得到图的最短路径。

#### 结构

```python
def calc(graph = None, graphType = None, method = 'dij', useCUDA = True, directed = False, pathRecordBool = False, srclist = None, block = None, grid = None):
    """
    function: 
        a calculate interface.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: str, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: bool, use CUDA to speedup or not.
        directed: bool, directed or not. only valid in read graph from file.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.

    return:
        class, Result object. (more info please see the developer documentation).  
    """
    return result
```

#### parameters

该方法是接口函数，各个参数意义如下：

- graph， 图数据，必填，需要计算最短路径的图数据或者图数据存储文件。参见[图数据规范](https://github.com/LCX666/SPoon/blob/main/tutorials.md#%E5%9B%BE%E6%95%B0%E6%8D%AE%E8%A7%84%E8%8C%83)。
  1. 若是内存中的图数据，则支持三种格式的数据：邻接矩阵(matrix)、压缩邻接矩阵(CSR)、边集数组(edgeSet)。
  2. 若是图数据的文件，则表示满足**图数据规范**的图文件的文件名。
- graphType ，传入的图数据的类型，必填。 str。仅可以是以下三种类型：

  1. matrix： 表示传入的数据是邻接矩阵。
  2. CSR： 表示传入的数据是压缩邻接矩阵。
  3. edgeSet： 表示传入的数据是边集数组。
- method， 使用的最短路径计算方法，缺省为 'dij'。str。仅可以是以下类型：
  1. dij： 表示使用 `dijkstra` 算法求解最短路径。
  2. spfa： 表示使用 `bellman-ford` 算法求解最短路径。
  3. delta： 表示使用 `delta-stepping` 算法求解最短路径。
  4. edge： 表示使用边细粒度来计算最短路径。
- useCUDA，是否使用 CUDA 加速，缺省为 True。bool。仅可以是以下类型：
  1. True： 表示使用 CUDA 进行加速。
  2. False： 表示只使用 CPU 进行串行计算。
- directed，图是否有向，缺省为 False。bool。仅可以是以下类型：
  1. True：表示图为有向图。
  2. False：表示图为无向图。
- pathRecordBool， 是否需要记录最短路径的路径，缺省为 False。bool。仅可以是以下类型：
  1. True： 表示需要记录最短路径的路径。
  2. False：表示不需要计算路径，只需要路径的值。
- srclist，源点的集合，缺省为 None。int/list/None。可以是以下三种类型：
  1. int： 一个整数，表示图中的一个结点作为最短路径计算的源点。
  2. list： 一个列表，列表中的各个元素都是 int 表示图中的这些点都是源点，需要分别计算对应的最短路径。
  3. None： 一个空值，表示计算图中的全源最短路径。
- block， 线程block的结构，缺省程序会自动设置。仅可以是一个三元组 `(x, y, z)`。
- grid，线程block的结构，缺省程序会自动设置。仅可以是一个二元组`(x, y)`。

#### 返回值

返回值是 `class Result` 的一个实例。详细请参阅  [class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result-1)。



### func dispatcher()

#### 功能

任务调度函数，依据用户的输入数据判断程序的走向，调整程序的运行以及参数合法的检验。

#### 结构

```python
def dispatch(graph, graphType, method, useCUDA, pathRecordBool, srclist, msg, block, grid):
    """
    function: 
        schedule the program by passing in parameters.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        method: str, the shortest path algorithm that you want to use, only can be [dij, spfa, delta, fw, edge].
        useCUDA: bool, use CUDA to speedup or not.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        msg: the info of the graph.
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return:
        class, Result object. (more info please see the developer documentation). 
    """

   
    return result
```

#### parameters

该方法是接口函数承接的转入函数，各个参数意义都与接口函数一致，各个参数意义如下：

- graph， 图数据，必填，需要计算最短路径的图数据或者图数据存储文件。参见图数据规范。
  1. 若是内存中的图数据，则支持三种格式的数据：邻接矩阵(matrix)、压缩邻接矩阵(CSR)、边集数组(edgeSet)。
  2. 若是图数据的文件，则表示满足**图数据规范**的图文件的文件名。
- graphType ，传入的图数据的类型，必填。 str。仅可以是以下三种类型：

  1. matrix： 表示传入的数据是邻接矩阵。
  2. CSR： 表示传入的数据是压缩邻接矩阵。
  3. edgeSet： 表示传入的数据是边集数组或者是文件。
- method， 使用的最短路径计算方法，缺省为 'dij'。str。仅可以是以下类型：
  1. dij： 表示使用 `dijkstra` 算法求解最短路径。
  2. spfa： 表示使用 `bellman-ford` 算法求解最短路径。
  3. delta： 表示使用 `delta-stepping` 算法求解最短路径。
  4. edge： 表示使用边细粒度来计算最短路径。
- useCUDA，是否使用 CUDA 加速，缺省为 True。bool。仅可以是以下类型：
  1. True： 表示使用 CUDA 进行加速。
  2. False： 表示只使用 CPU 进行串行计算。
- pathRecordBool， 是否需要记录最短路径的路径，缺省为 False。bool。仅可以是以下类型：
  1. True： 表示需要记录最短路径的路径。
  2. False：表示不需要计算路径，只需要路径的值。
- srclist，源点的集合，缺省为 None。int/list/None。可以是以下三种类型：
  1. int： 一个整数，表示图中的一个结点作为最短路径计算的源点。
  2. list： 一个列表，列表中的各个元素都是 int 表示图中的这些点都是源点，需要分别计算对应的最短路径。
  3. None： 一个空值，表示计算图中的全源最短路径。
- block， 线程block的结构，缺省程序会自动设置。仅可以是一个三元组 `(x, y, z)`。
- grid，线程block的结构，缺省程序会自动设置。仅可以是一个二元组`(x, y)`。

#### 返回值

返回值是 `class Result` 的一个实例。详细请参阅 [class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result-1)。



### func transfer()

#### 功能

将用户输入的图数据进行规范化，转化图的格式，计算一些后续算法需要的必要参数。

#### 结构

```python
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
	
```

#### parameters

- para，函数间传递的参数类。class parameter。类型固定。
- outType， 想要转化输出的类型。str。可以是以下类型：
  1. matrix， 表示邻接矩阵。
  2. CSR， 表示压缩邻接矩阵。
  3. edgeSet， 表示边集数组。

#### 返回值

无



### func judge()

#### 功能

判断当前图数据在当前版本的算法和当前的硬件条件下是否可以使用GPU加速，以及判断是否需要启动图分割算法。

#### 结构

```python
def judge(para):
    """
    function: 
        determine whether the current graph needs to use graph segmentation.
    
    parameters: 
        para: class, Parameter object.
    
    return:
        bool, True/False. (more info please see the developer documentation).
```

#### parameters

- para，函数间传递的参数类。class parameter。类型固定。

#### 返回值

- bool，是否需要启用图分割。
  1. True：表示需要启用图分割算法。
  2. False：不需要启用图分割算法。



### func read()

#### 功能

从文件中读取图，并转化为CSR格式或者edgeSet格式。并获取图的一些特征信息。

#### 结构

```python
def read(filename = 'data.txt', directed = False):
    """
    function:
        read graph from file, and shape to a Graph object.
    
    parameters:
        filename: str, the graph data file name.
    
    return:
        class, Graph object.
    """
```

#### parameters

- filename，待读取的文件名，必填。str。
- directed，图是否有向，缺省为 False。bool。仅可以是以下类型：
  1. True：表示图为有向图。
  2. False：表示图为无向图。

#### 返回值

返回值是 `class Class` 的一个实例。详细请参阅 [class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result-1)。



### func draw()

#### 功能

依据最短路径绘制第一个源的最短路径图。

#### 结构

```python
def draw(path, n, s, graph, graphType):
    """
    function: 
        use path to draw a pic.

    parameters:
        path: list, must,  about precursors of each vertex in each problem.
        n: int, must, the number of vertices.
        s: int , must, the source vertex.
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
    
    return: 
        None, no return. 
```

#### parameters

- path，最短路径的前驱数组，必填。list。
- n，图中结点的数量，必填。int。
- s，源点的编号，必填。int。
- graph， 图数据，必填，需要计算最短路径的图数据。参见图数据规范。
- graphType ，传入的图数据的类型，必填。 str。仅可以是以下三种类型：
  1. matrix： 表示传入的数据是邻接矩阵。
  2. CSR： 表示传入的数据是压缩邻接矩阵。
  3. edgeSet： 表示传入的数据是边集数组。

#### 返回值

无



### func generate()

#### 功能

生成一张包含指定点数、边数和边权范围的连通图。

#### 结构

```python
def generate(filename = 'data0.txt', n = 1000, m = 30000, l = 1, r = 12):
    """
    function: 
        generate a random graph to file. (more info please see the developer documentation). 

    parameters:
        filename: str, the filename of file to save the graph.
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        l: int, the min value of a edge.
        r: int, the max value of a edge.
    
    return: 
        None, no return.
```

#### parameters

- filename，生成图存放的文件名，缺省值为 'data0.txt'。str。
- n，生成图的结点数量，缺省为 1000。int。
- m，生成图的边数量，缺省为 30000。int。
- l，图中的边权下界，缺省为 1。int。
- r，图中的边权上界，缺省为 12。int。

#### 返回值

无



### func check()

#### 功能

检查两个数据是否相等。

#### 结构

```python
def check(data1, data2, name1 = 'data1', name2 = 'data2'):
    """
    function: 
        to check the data is equal or not.

    parameters:
        data1: numpy.ndarray, must, first data.
        data2: numpy.ndarray, must, second data.
        name1: str, the name of data1.
        name2: str, the name of data2.
    
    return: 
        str, the EQUAL or NOT.
    """
```

#### parameters

- data1，第一个待检测的数据，必填。list。
- data2，第二个待检测的数据，必填。list。
- name1，第一个数据名字，缺省为 'data1'。str。
- name2，第二个数据名字，缺省为 'data2'。str。

#### 返回值

str， 相等或者不相等



### func [apsp, sssp, mssp].'[dij, delta, edge, spfa]'_[cpu, gpu]

本工具中封装了`Dijkstra`、`Bellman-Ford`、`Delta-Stepping`、`Edge-thread` 最短路径算法。同时支持单源。多源和全源的算法和使用CPU串行计算的版本和CUDA加速的版本，属于多个文件中的多个方法，但是具有相似的参数和返回值。

#### 功能

通过传入的参数，利用算法实际计算出最短路径，解决问题。方法名不同，参数略有不用，启用的计算方法也略有差异。

#### 结构

```python
def dijkstra(CSR, n, s, pathRecordingBool = False):
	"""
	function: use dijkstra algorithm in GPU to solve the SSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        s: the source list, can be number.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) . 
	"""
	...
```

#### parameters

```python
CSR: CSR graph data. (more info please see the developer documentation) .
n: the number of the vertexs in the graph.
s: the source list, can be number.(more info please see the developer documentation).
pathRecordingBool: record the path or not.
```

#### 返回值

返回值是 `class Result` 的一个实例。详细请参阅 [class Result](https://github.com/LCX666/SPoon/blob/main/tutorials.md#class-result-1)。

------



## 类

### class Result

#### 功能

该类是工具的统一返回类，所有的计算方法的结果都将以该类的实例进行返回。

#### 结构

```python
class Result(object):
	"""
    function: 
        to store the result of different algorithm.

    parameters:
        dist: list, the shortest path distance answer for algorithm.
        timeCostNum: float, a float data of time cost of getting the answer, so it can use to calculate.
        timeCost: str, a str data of time cost of getting the answer.
        memoryCost: str, memory cost of getting the answer.
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.(more info please see the developer documentation).
        graphType: str, must, type of the graph data, only can be [matrix, CSR, edgeSet].(more info please see the developer documentation).
        msg: str, the info of the graph.
    
    method:
        display: show the detail of this calculation.
        drawPath: draw the path from vertices to the sources.
        calcPath: calc the path through the graph and dist.
    
    return: Result object.
	"""
	def __init__(self, 
                dist = None, 
                timeCost = None, 
                memoryCost = None, 
                graph = None,
                graphType = None,
                msg = ""):
		"""
		"""
		
	def display(self):
        """
        function: 
            show the detail of the graph, parameters and calc time.

        parameters:
            None, but 'self'.
        
        return: 
            str, the msg info.   
        """
        
    def drawPath(self):
        """
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.     
        """
        
	def calcPath(self,CSR=None,matrix=None,edgeSet=None):
		"""
        function: 
            to get the path.

        parameters:
            None, but 'self'.
        
        return: 
            None, no return.		
		"""
        

```

#### parameters

- dist, 最短路径计算的数组。np.array。
- timeCost, 计算最短路径用时。float。
- ~~memoryCost， 计算过程中的内存占用。暂时无。~~
- graph， 图数据，必填，需要计算最短路径的图数据。参见图数据规范。
- graphType ，传入的图数据的类型，必填。 str。仅可以是以下三种类型：
  1. matrix： 表示传入的数据是邻接矩阵。
  2. CSR： 表示传入的数据是压缩邻接矩阵。
  3. edgeSet： 表示传入的数据是边集数组。
- msg， 提示信息，默认构建。

#### 属性

- dist， 最短路径计算的数组。np.array。详细信息如下：
  1. 若是计算单源最短路径，则是一维数组，`dist[i]` 即表示在当前的单源最短路径问题中，源点到结点 `i` 的最短距离。
  2. 若是计算多源最短路径，则是多个一维数组组成的二维数组，`dist[i][j]` 即表示在第 `i` 个源的单源最短路径问题中，第 `i` 个源点到结点 `j` 的最短距离。
  3. 若是计算全源最短路径，则是 `n × n` 的二维数组，`dist[i][j]` 即表示在源点 `i` 的单源最短路径问题中，源点 `i` 到结点 `j` 的最短距离。
- path，最短路径的路径。np.array。详细信息如下：
  1. 若是计算单源最短路径，则是一维数组，`path[i]` 即表示在当前的单源最短路径问题中，结点 `i` 的前驱结点，即源点需要从结点 `path[i]` 到达结点 `i` 。
  2. 若是计算多源最短路径，则是多个一维数组组成的二维数组，`path[i][j]` 即表示在第 `i` 个源的单源最短路径问题中，第 `i` 个源点需要从结点 `path[i][j]` 到达结点 `j` 。
  3. 若是计算全源最短路径，则是 `n × n` 的二维数组，`path[i][j]` 即表示在源点 `i` 的单源最短路径问题中，源点 `i` 需要从结点 `path[i][j]` 到达到结点 `j` 。
- timeCost, 时间花费。 float。表示计算最短路径问题花费的时间，单位是秒。
- memoryCost， 空间花费。int。表示计算最短路径问题花费的空间，单位是字节。
- graph， 图数据，必填，需要计算最短路径的图数据。参见图数据规范。
- graphType ，传入的图数据的类型，必填。 str。仅可以是以下三种类型：
  1. matrix： 表示传入的数据是邻接矩阵。
  2. CSR： 表示传入的数据是压缩邻接矩阵。
  3. edgeSet： 表示传入的数据是边集数组。
- msg， 计算过程中的一些参数信息，默认构建。str。

#### 方法

- display，func，将计算详情 `msg` 信息展示给用户。
  1. parameters，无。
  2. 返回值，msg 计算信息。str。
- drawPath，func，将计算的最短路径可视化绘制。
  1. parameters， 无。
  2. 返回值，无。
- calcPath，func，从dist数组和图中复现出path。直接赋值给 `path` 属性。
  1. parameters， 无。
  2. 返回值， 无。



### class parameter

#### 功能

该类是此工具的内部传参使用的类，记录了各类与算法相关的参数，与用户无关。

#### 结构

```python
class parameter(object):
	"""
    function: 
        transfer the parameters in the functions.

    parameters: 
        None, but 'self'.

    attributes:
        BLOCK: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        GRID: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        useCUDA: bool, use CUDA to speedup or not.
        CSR: tuple, a 3-tuple of integers as (V, E, W) about the CSR of graph data. (more info please see the developer documentation).
        matrix: matrix, as (n,n), about adjacency matrix of graph data.
        edgeSet: tuple, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
        graphType: str, type of graph. [edgeSet, matrix, CSR].
        method: str, the algorithm. [dij, delta, spfa, fw, edge]
        srclist: list/int, the source of shortest path problem.
        sourceType: str, the type of the problem. [APSP, SSSP, MSSP]
        pathRecordingBool: bool, record the path or not.
        delta: int, the delta of delta-stepping algorithm.
        MAXN: int, the max value of the edges.
        MAXU: int, the vertex has the maxOutDegree.
        maxOutDegree: int, the max out degree of the graph.
        part: int, the number of the edges that will put to GPU at a time.(divide algorithm)
        streamNum: int, the number of streams used.
        msg: str, the info of the graph.
    
    method:
        None, but init.
            
    return 
        class, Parameter object.
	"""
	def __init__(self):
		
		...
```

#### parameters

无

#### 属性

- BLOCK,  三元组 `(x,y,z)` 表示计算过程中线程的结构。tuple。选填。
- GRID，二元组 `(x,y)`  表示计算过程中block的结构。tuple。选填。
- n，图中结点的数量。int。
- m，图中有向边的数量。int。
- useCUDA, 是否使用CUDA加速。bool。
- CSR， CSR图数据。
- matrix， 邻接矩阵图数据。
- edgeSet， 边集数组图数据。
- graphType ，传入的图数据的类型。 str。仅可以是以下三种类型：
  1. matrix： 表示传入的数据是邻接矩阵。
  2. CSR： 表示传入的数据是压缩邻接矩阵。
  3. edgeSet： 表示传入的数据是边集数组或者是文件。
- method， 使用的最短路径计算方法。str。仅可以是以下类型：
  1. dij： 表示使用 `dijkstra` 算法求解最短路径。
  2. spfa： 表示使用 `bellman-ford` 算法求解最短路径。
  3. delta： 表示使用 `delta-stepping` 算法求解最短路径。
  4. edge： 表示使用边细粒度来计算最短路径。
- srclist，源点的集合。int/list/None。可以是以下三种类型：
  1. int： 一个整数，表示图中的一个结点作为最短路径计算的源点。
  2. list： 一个列表，列表中的各个元素都是 int 表示图中的这些点都是源点，需要分别计算对应的最短路径。
  3. None： 一个空值，表示计算图中的全源最短路径。
- sourceType， 解决的问题类型。str。可以是以下三种类型：
  1. APSP： 全源最短路径
  2. MSSP：多源最短路径
  3. SSSP：单源最短路径
- pathRecordBool， 是否需要记录最短路径的路径。bool。仅可以是以下类型：
  1. True： 表示需要记录最短路径的路径。
  2. False：表示不需要计算路径，只需要路径的值。
- delta, delta-stepping算法中使用的delta值。 int。
- MAXW，图中最大边权值。int
- MAXU，图中最大度的点编号。int。
- maxOutDegree，图中最大的出度。int。
- part，分图算法中一次拷贝进GPU中的边的数量。int。
- streamNum， 分图多流中启用的流的数量。int。
- msg，计算过程的提示信息。str。

#### 方法

无



### class device

#### 功能

该类是此工具获取设备显卡信息的类，通过 `pycuda` 和 `pynvml` 实现功能。

#### 结构

```python
class Device(object):
	"""
    function: 
        get the GPU device infomation, get the type and attributes.

    parameters: 
        None, but 'self'.

    attributes:
        device: class, a pyCUDA device class object.
        CUDAVersion: str, the version of CUDA.
        driverVersion: int, the version of CUDA driver.
        deviceNum: int, the number of valid GPU device.
        deviceName: str, the name of the device.
        globalMem: int, the max number of the global memory.
        sharedMem: int, the max number of the shared memory.
        processNum: int, the number of processors.
        freeMem: int, the bytes of free memory.
        temperature: int, the temperature of the device.
        powerStstus: the power ststus of the device.

    method:
        get_device_type: get the type of the device.
        get_number_of_device: get the number of the device.
        get_version: obtain the version of CUDA against which PyCuda was compiled.
        get_driver_version: obtain the version of the CUDA driver on top of which PyCUDA is running. 
        getDeviceInfo: obtain the device infomation include: 'freeMemory, totalMemory, memoryUsed, temperature, powerStstus'.
        get_attributes: the pycuda's get_attributes.

    return 
        class, Device object.
	"""
	def __init__(self):
		...
```

#### parameters

无

#### 属性

- device，pyCUDA的Device类。object。
- CUDAVersion， CUDA的版本。str。
- driverVersion，驱动的版本。str。
- deviceName， GPU设备的型号。str。
- deviceNum，CUDA设备数量。int。
- total，总显存的容量。int。
- free，空闲显存的容量。int。
- used，已经使用的显存容量。int。
- temperature，设备的温度。str。
- powerStstus，电源的状态。str。

#### 方法

- getDeviceInfo，func，获取分图需要计算的显存剩余信息。
  1. parameters，无。
  2. 返回值，无。



### class graph

#### 功能

将用户提供的图数据转化为一个图类保存图的信息。

#### 结构

```python
class Graph(object):
    """
    function: 
        a graph class.

    parameters:
        filename: str, must, the graph data file. (more info please see the developer documentation).
        directed: bool, the graph is directed ot not.
    
    attributes:
        n: int, the number of the vertices in the graph.
        m: int, the number of the edges in the graph.
        CSR: tuple, a 3-tuple of integers as (V, E, W) about the CSR of graph data. (more info please see the developer documentation).
        src, des, w: tuple, a 3-tuple of integers as (src(list), des(list), val(list)) about the edge set.
        MAXW: int, the max weight of the edges.
        MINW: int, the min weight of the edges.
        MAXD: int, the max degree(In degree and Out degree) of all the vertices.
        MAXU: int, one of the vertices with the max degree.
        MIND: int, the min degree(In degree and Out degree) of all the vertices.
        MINU: int, one of the vertices with the min degree.
        degree: list, save the degree of each vertex.
        msg: str, the message about the read func.
        filename: str, the file name of the graph.
    
    method:
        read: read the graph from file.
        reshape: convert data to numpy.int32.

    return: 
        class, Graph object.
    """
    
    def __init__(self, filename = None, directed = False):
        """
        """
    
    def read(self):
        """
        function: 
            read the graph from file.
            only accept graphic data in edgeSet format and store it in memory in CSR/edgeSet format
            by the way, we wanna to specify the edgeSet format as a variable with 3 array src/des/weight which are consistent with every edge in graph

        parameters: 
            None, but 'self'.

        return:
            None, no return.
        """
        
    def reshape(self):
        
        """
        function: 
            convert data to numpy.int32.

        parameters: 
            None, but 'self'.

        return:
            None, no return.
        """        
```

#### parameters

- filename, 待读取的文件名，缺省即不读图。str。
- directed，标识图是否有向，缺省为 False。bool。

#### 属性

- n，图中结点的数量。int。
- m，图中有向边的数量。int。
- CSR， CSR图数据。
- matrix， 邻接矩阵图数据。
- src, des, w， 边集数组图数据。
- MAXW，最大边权。int。
- MINW，最小边权。int。
- MAXD，最大度。int。
- MAXU，最大度的结点编号。int。
-  MIND，最小度。int。
- MINU，最小度的结点编号。int。
- degree，各个结点的度。list。
- msg，计算过程中的提示信息。str。
- filename，待读图的文件名。str。

#### 方法

- read，func，从文件中读取图。
  1. parameters，无。
  2. 返回值，无。
- reshape，func，将数据转化为numpy的格式。
  1. parameters，无。
  2. 返回值，无。