# [SParry](https://github.com/LCX666/SParry)

![image](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/sparry.png)

**SParry** is a shortest path calculating tool using some algorithms with CUDA to speedup.
[English Version](https://github.com/LCX666/SParry/blob/main/tutorials.md#sparry)|[中文版](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E5%AE%89%E8%A3%85)

It's **developing**.



------

`SParry` is a **shortest path calculation toolkit**, the main shortest path algorithms, including `Dijkstra`, `Bellman-Ford`, `Delta-Stepping`, and `Edge-Based`, are encapsulated.  **A parallel accelerated version based on CUDA is provided** to improve development efficiency.

At the same time, it can divide the graph data into parts, and solve it more quickly than using the CPU when the graph is too large to put it in the GPU directly.



------

## Installation

### Environment & Dependence

The following is the environment that passed the test in the development experiment.

**Window：**

> python 3.6/3.7
>
> pycuda 2019.1.2
>
> numpy 1.18.2
>
> CUDA Version  10.2.89
>
> networkx 2.4
>
> logging 0.5.1.2
>
> pynvml 8.0.4

**Linux**

> python 3.6
>
> pycuda 2019.1.2
>
> numpy 1.19.2
>
> CUDA Version 10.1.243
>
> networkx 2.4
>
> logging 0.5.1.2
>
> pynvml 8.0.4

<br>

### Installation

Download the file package directly and run the `calc` interface function **in the main directory**.

**It's not not a release version currently, so it cannot be installed with pip, and the development structure is not yet perfect. **



------

## flow chart

![image](https://raw.githubusercontent.com/LCX666/SParry/main/chart.svg)



------

## Test&Result

We have conducted a lot of tests on this tool, and no errors occurred in the test results. We have counted the time consumption and serial-parallel speedup ratio of each algorithm on some graphs to solve the **single-source shortest path (SSSP) and all-pairs shortest path (APSP)**. The following shows the time variation of each algorithm with the number of nodes when the average degree is 4, please refer to [SParry/testResult](https://github.com/LCX666/SParry/tree/main/testResult) .

![SSSPTimeConsumptionByDegree4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/SSSPTimeConsumptionByDegree4.png)

<center>SSSP time consumption by degree=4</center>



![SSSPSpeedupRatioByDegree4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/SSSPSppedupRatioByDegree4.png)

<center>SSSP speedup ratio by degree=4</center>

![APSPTimeConsumptionByDegreeAPSP4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/APSPTimeConsumptionByDegreeAPSP4.png)

<center>APSP time consumption by degree=4</center>

![APSPSpeedupRatioByDegree4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/APSPSpeedupRatioByDegree4.png)

<center>APSP speedup ratio by degree=4</center>












------

## Quick start tutorial

This section is an introduction to help beginners of `SParry` get started quickly.

### step1. cd to the current root directory

```powershell
cd XXX/SParry/
```



### step2. Import calculation interface

#### In Memory-matrix

When your graph data is in **memory** and **meets the [adjacency-matrix data](https://github.com/LCX666/SParry/blob/main/tutorials.md#adjacency-matrix)** requirements, you can import your data as below to quickly calculate the results.

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> matrix = np.array([[0,1,2,3],[1,0,2,3],[2,2,0,4],[3,3,4,0]], dtype = np.int32) # the data of the adjacency matrix
>>> matrix # simulate graph data that already has an adjacency matrix
array([[0, 1, 2, 3],
       [1, 0, 2, 3],
       [2, 2, 0, 4],
       [3, 3, 4, 0]], dtype=int32)
>>>
>>> graph = read(matrix = matrix, # the data passed in is the adjacency matrix
...              method = "dij", # the calculation method uses Dijkstra
...              detail = True # record the details of the graph
...             ) # process the graph data
>>>
>>> res = calc(graph = graph, # class to pass in graph data
...            useCUDA = True, # use CUDA acceleration
...            srclist = 0, # set source to node 0
...            )
>>>
>>> res.dist # output shortest path
array([0, 1, 2, 3], dtype=int32)
>>>
>>> print(res.display()) # print related parameters

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 16,
[+] the max edge weight in the Graph:                   MAXW = 4,
[+] the min edge weight in the Graph:                   MINW = 0,
[+] the max out degree in the Graph:                    degree(0) = 4,
[+] the min out degree in the Graph:                    degree(0) = 4,
[+] the average out degree of the Graph:                avgDegree = 4.0,
[+] the directed of the Graph:                          directed = Unknown,
[+] the method of the Graph:                            method = dij.


[+] calc the shortest path timeCost = 0.017 sec
>>>
>>> res.drawPath() # The red line indicates that this edge is on the shortest path; the orange vertex is the source vertex; the arrow indicates the direction of the edge; and the number on the edge indicates the edge weight.
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103113127.png" alt="image-20201104103113127"  />



#### In Memory-CSR

When your graph data is in **memory** and **compliant with the [CSR data](https://github.com/LCX666/SParry/blob/main/tutorials.md#csr) requirements**, you can import your data as below to quickly calculate the results.

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> CSR = np.array([np.array([0, 2, 3, 4, 4]), 
...                 np.array([1, 2, 3, 1]), 
...                 np.array([1, 3, 4, 5])])
>>> CSR # simulation already has CSR format is graph data
array([array([0, 2, 3, 4, 4]), array([1, 2, 3, 1]), array([1, 3, 4, 5])],
      dtype=object)
>>>
>>> graph = read(CSR = CSR, # The data type passed in is CSR
...              method = "delta", # The algorithm used is delta-stepping
...              detail = True) # record the details of the graph
>>>
>>> res = calc(graph = graph, # the incoming graph data class
...            useCUDA = True, # use CUDA parallel acceleration
...            srclist = 0) # source point is node 0
>>>
>>> res.dist # calculated shortest path
array([0, 1, 3, 5], dtype=int32)
>>>
>>> print(res.display()) # print related parameters

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 4,
[+] the max edge weight in the Graph:                   MAXW = 5,
[+] the min edge weight in the Graph:                   MINW = 1,
[+] the max out degree in the Graph:                    degree(0) = 2,
[+] the min out degree in the Graph:                    degree(3) = 0,
[+] the average out degree of the Graph:                avgDegree = 1.0,
[+] the directed of the Graph:                          directed = Unknown,
[+] the method of the Graph:                            method = delta.


[+] calc the shortest path timeCost = 0.007 sec
```



#### In Memory-edgeSet

When your graph data is in **memory** and **compliant with the [edgeSet data](https://github.com/LCX666/SParry/blob/main/tutorials.md#edgeset)** requirements, you can import your data as below to quickly calculate the results.

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> edgeSet = [[0, 0, 2, 1], # start point of each edge
...            [1, 3, 1, 3], # the end point of each edge
...            [1, 2, 5, 4]] # weights of each edge
>>>
>>> edgeSet # simulation already has edgeSet format is graph data
[[0, 0, 2, 1], [1, 3, 1, 3], [1, 2, 5, 4]]
>>>
>>> graph = read(edgeSet = edgeSet, # the incoming graph data is edgeSet
...              detail = True) # need to record the data in the graph
>>>
>>> res = calc(graph = graph, # the incoming graph data class
...            useCUDA = False, # sse CPU serial computation
...            srclist = 0) # source point is node 0
>>> res.dist # calculated shortest path
array([         0,          1, 2139045759,          2], dtype=int32)
>>>
>>> print(res.display()) # print related parameters

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 4,
[+] the max edge weight in the Graph:                   MAXW = 5,
[+] the min edge weight in the Graph:                   MINW = 1,
[+] the max out degree in the Graph:                    degree(0) = 2,
[+] the min out degree in the Graph:                    degree(3) = 0,
[+] the average out degree of the Graph:                avgDegree = 1.0,
[+] the directed of the Graph:                          directed = Unknown,
[+] the method of the Graph:                            method = dij.


[+] calc the shortest path timeCost = 0.0 sec
```



#### In File

When your graph data is stored in a **file** and **meets the data requirements** ([file data](https://github.com/LCX666/SParry/blob/main/tutorials.md#file-format)) , you can also pass in a file to calculate the shortest path.

The file in this example is as follows, `test.txt` .

```
4 6
0 1 1
0 2 2
0 3 3
1 2 2
1 3 3
2 3 4

```

code is below:

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> filename = "./data/test.txt" # the path to the file where the graph data is stored
>>> graph = read(filename = filename, # the data passed in is the file
...              method = "spfa", # the algorithm used is spfa
...              detail = True, # record the details of the graph
...              directed = False) # the graph is an undirected graph
>>>
>>> res = calc(graph = graph, # the incoming graph data class
...            useCUDA = True, # use CUDA parallel acceleration
...            srclist = 0) # source point is node 0
>>>
>>> res.dist # calculated shortest path
array([0, 1, 2, 3], dtype=int32)
>>>
>>> print(res.display()) # print related parameters

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 12,
[+] the max edge weight in the Graph:                   MAXW = 4,
[+] the min edge weight in the Graph:                   MINW = 1,
[+] the max out degree in the Graph:                    degree(0) = 3,
[+] the min out degree in the Graph:                    degree(0) = 3,
[+] the average out degree of the Graph:                avgDegree = 3.0,
[+] the directed of the Graph:                          directed = False,
[+] the method of the Graph:                            method = spfa.


[+] calc the shortest path timeCost = 0.002 sec
>>> res.drawPath() # The red line indicates that this edge is on the shortest path; the orange vertex is the source vertex; the arrow indicates the direction of the edge; and the number on the edge indicates the edge weight.
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103213127.png" alt="image-20201104113340700" style="zoom: 67%;" />





------

## Interface

### Version Queries

This tool is still in its initial development iteration and has not been uploaded to the `pip` site.

<br>

### Error Reporting

All error reports in the software will be reported with `Python` error messages, and hint sentences are embedded in some places where errors are expected.

<br>

### Calculation Interface

#### func calc()

This function is the only calculation interface of this tool. By calling this method, you can directly calculate the shortest path of the graph. See [func calc()](https://github.com/LCX666/SParry/blob/main/tutorials.md#func-calc-1) for more information.



#### func read()

This function is used to pre-process graphs in this tool. By calling this method, you can normalize the user's various graph data into a uniform class. See [func read()](https://github.com/LCX666/SParry/blob/main/tutorials.md#func-read-1) for more information.



#### func INF()

This function returns the positive infinity in the tool.





------

## Data Type & Specification

1. The current version of this tool only supports data with 32-bit integers. Therefore, the edge weights, the number of vertices and the number of edges must be within the range of what a 32-bit integer can express. Also the calculated shortest path result value should not be outside the range of the 32-bit integer. One explanation for this is that most current data problems can be solved within the range of 32-bit integers. The second is that most non-professional NVIDIA GPUs are neutered for 64-bit and therefore do not support 64-bit data.

2. In this tool, all graphs are numbered from vertex0 by default. Therefore graphs numbered from vertex1 should ignore the data from the first vertexand start with the second data. (At this vertex the number of nodes perceived in the tool will be 1 more than the true number of nodes in the graph, because the tool always thinks there is another vertex0.



### Graph Data  Specification

The graph data received in the tool is available in **both [file formats](https://github.com/LCX666/SParry/blob/main/tutorials.md#file-format) and memory formats** (three types: [adjacency matrix (matrix)](https://github.com/LCX666/SParry/blob/main/tutorials.md#adjacency-matrix), [compressed adjacency matrix (CSR)](https://github.com/LCX666/SParry/blob/main/tutorials.md#csr), and [edgeSet array (edgeSet)](https://github.com/LCX666/SParry/blob/main/tutorials.md#edgeset)).

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
    - If you set the `directed = False` in `func calc` ，which means the graph is undirected. The tool automatically converts the above unidirectional edges into bidirectional edges and automatically doubles the number of edges in the tool, since it uses two directed edges to represent a directionless edge.
    - If you set the `directed = True` in `func calc` ，which means the graph is directed. This tool will read the map strictly in one direction, one way.
    - Default parameter directed = False. The default graph is an undirected graph.
- Second row to fifth row: 3 integers, representing an edge.
  - The first parameter represents the start of the current edge.
  - The second parameter represents the endvertex of the current side.
  - The third parameter represents the side weight of the current edge.
- Sixth row(the last row): A blank line.
  - **End of document with a blank line**



#### Adjacency Matrix

The adjacency matrix is an n × n two-dimensional array to represent a graph, any one element of the matrix can represent an edge. That is, `matrix[i][j] = w` means that there is at least one edge in the graph from vertex`i` to vertex`j`, and its edge weight is `w`.

Since this tool is a shortest-path calculator, more strictly speaking, `matrix[i][j] = w` should be the side weight of the shortest of all edges of the representation from vertex`i` to vertex`j` is `w`.

Since this tool calculates in the range of a 32-bit integer, positive infinity in this tool is a number that is very close to the maximum number that can be represented by a 32-bit integer. The positive infinity of this tool can be returned with `func: calc.INF()`.

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

`CSR` is the main storage and computing method of this tool, which saves memory space in most cases compared to the adjacency matrix storage method. It also saves memory space when using GPU to accelerate computation. It is represented by three one-dimensional arrays: `V`, `E` and `W`. In this tool, these three arrays are combined in the order of `V`, `E` and `W` to form CSR.

The `V` array is the starting subscript of the first edge of each vertexin the `E` array, and its dimension is the number of vertices in the diagram, but for the convenience of calculation, a virtual vertexis usually added at the end to determine whether the end is reached or not. So the dimension must be strictly virtual nodes in this tool.

The `E` array is a record of which vertexeach edge ends at. Its dimension is therefore strictly the number of (directed) edges in the graph, and an undirected edge is represented by two directed edges in this tool.

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

**A virtual vertex at the end of V to define the boundary.**



#### edgeSet

`edgeSet` is a ternary list where the first list represents the starting vertex of all edges, the second list represents the end vertex of all edges, and the third list represents the edge weights of all edges.

The above figure translates into an edgeSet array as follows

```python
In [1]: edgeSet
Out[1]: [[0, 0, 2, 1], [1, 3, 1, 3], [1, 2, 5, 4]]
```

**If it is an undirected edge it needs to be represented as two directed edges in the above list and then passed in.**

<br>

### Constants

INF: Positive infinity in this tool. it's 2139045759



------

## Func

### func read()

#### Function

This function is the function used to pre-process graphs in this tool. By calling this method, you can normalize various graph data of users into a unified class.

#### Structure

```python
def read(CSR = None, matrix = None, edgeSet = None, filename = "", method = "dij", detail = False, directed = "Unknown", delta = 3):
    """
    function: 
        convert a graph from [CSR/matrix/edgeSet/file] 
            to Graph object(see the 'SParry/classes/graph.py/Graph')
            as the paremeter 'graph' of the func calc(see the 'SParry/calc.py/calc')
    
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
        class, Graph object. (see the 'SParry/classes/graph.py/Graph')     
    """
```

#### Parameters

- CSR, graph data, optional, meets the CSR data format as detailed in [CSR Data Format Specification](https://github.com/LCX666/SParry/blob/main/tutorials.md#csr).
- matrix, graph data, optional, satisfies the data format of the adjacency matrix, see [matrix data format specification](https://github.com/LCX666/SParry/blob/main/tutorials.md#adjacency-matrix).
- edgeSet, graph data, optional, satisfies the data format of the edge set array, see [edgeSet data format specification](https://github.com/LCX666/SParry/blob/main/tutorials.md#edgeset).
- filename, graph file name, optional, satisfies the data format of the file format, see [file data format specification](https://github.com/LCX666/SParry/blob/main/tutorials.md#file-format).
- **Note! The above four parameters are optional, but all four cannot be None. priority is: CSR > matrix > edgeSet > filename.**
- method, the shortest path calculation method used, default is 'dij'. str. can only be of the following types: 
  1. dij: Indicates that the shortest path is solved using the `dijkstra` algorithm. 
  2. spfa: indicates that the shortest path is solved using the `bellman-ford` algorithm. 
  3. delta: denotes the shortest path using the `delta-stepping` algorithm.
  4. edge: denotes the use of edge fine-grained to compute the shortest path.
- detail, whether to count the details of the graph, default is False. bool. if set to True, it may take more time to read the graph data.
- directed, whether the graph is directed or not, default is False. bool. valid only if filename is entered.
- delta, the parameter delta of the delta-stepping algorithm, defaults to 3. int. valid only if the selected algorithm is delta-stepping.

#### Return

The return value is an instance of `class Graph`. See [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph) for details.

<br>

### func calc()

#### Function 

This function is an interface function of `SParry`. Through this function, users can pass in their own graph data and some necessary parameters to calculate the shortest path of the graph.

#### Structure

```python
def calc(graph = None, useCUDA = True, useMultiPro = False, pathRecordBool = False, srclist = None, block = None, grid = None, namename = None):
    
    """
    function: 
        a calculate interface.
    
    parameters: 
        graph: class Graph, must, the graph data that you want to get the shortest path.
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.

    return:
        class, Result object. (see the 'SParry/classes/result.py/Result') 
    """
```

#### Parameters

This method is an interface function with the following meaning of each parameter.

- graph, graph data, required, the graph data or graph data storage file where the shortest path needs to be calculated. This function accepts this parameter must be [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph).

- useCUDA, whether to use CUDA acceleration, defaults to True. bool. can only be of the following types.

  1. True: Indicates that CUDA is used for acceleration. 
  False: Indicates that only the CPU is used for serial computation.

- useMultiPro, whether to use CPU multiprocess acceleration, default is False. bool. can only be of the following types: 

  1. True: Indicates that the CPU multiprocess is used for accelerated computation. 

  2. False: Indicates that the CPU multiprocess is not used for accelerated computation.

     **Note that this parameter is only meaningful if `method` is specified as `dij`, `useCUDA` is `False`, and the problem being solved is `APSP` or `MSSP`. **

- pathRecordBool, whether to record the path of the shortest path, default is False. bool. can only be of the following types: 

  1. True: means the path of the shortest path needs to be recorded. 
  2. False: means the path does not need to be calculated, only the value of the path is needed.

- srclist, the set of source points, default is None. int/list/None. can be of the following three types.

  1. int: an integer that represents a node in the graph as the source point for shortest path computation.
  2. list: a list, the elements of which are int means that the points in the graph are all source points and need to be computed separately for the corresponding shortest paths. 
  3. none: a null value, which means that the shortest path of all sources in the graph is calculated.

- block: The structure of the thread block, which is set automatically by the default program. it can only be a triple `(x, y, z)`.

- grid, the structure of the thread block, set automatically by default. can only be a binary `(x, y)`.

#### Return

The return value is an instance of `class Result`. Please refer to [class Result](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-result) for details.

<br>

### func dispatcher()

#### Function

The task scheduling function judges the direction of the program according to the user's input data, adjusts the operation of the program and checks the legality of the parameters.

#### Structure

```python
def dispatch(graph, useCUDA, useMultiPro, pathRecordBool, srclist, block, grid):
    """
    function: 
        schedule the program by passing in parameters.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.
            (more info please see the developer documentation).
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].
            (more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return:
        class, Result object. (see the 'SParry/classes/result.py/Result').
    """
   
    return result
```

#### Parameters

This method is the transfer function undertaken by the interface function, the meaning of each parameter is consistent with the interface function, the meaning of each parameter is as follows.

- graph, graph data, required, need to calculate the shortest path of the graph data or graph data storage file. this function accepts this parameter must be [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph).

- useCUDA, whether to use CUDA acceleration, defaults to True. bool. can only be of the following types.

  1. True: Indicates that CUDA is used for acceleration. 
  2. False: Indicates that only the CPU is used for serial computation.

- useMultiPro, whether to use CPU multiprocess acceleration, default is False. bool. can only be of the following types: 

  1. True: Indicates that the CPU multiprocess is used for accelerated computation. 

  2. False: Indicates that the CPU multiprocess is not used for accelerated computation.

     **Note that this parameter is only meaningful if `method` is specified as `dij`, `useCUDA` is `False`, and the problem being solved is `APSP` or `MSSP`. **

- pathRecordBool, whether to record the path of the shortest path, default is False. bool. can only be of the following types: 

  1. True: means the path of the shortest path needs to be recorded. 
  2. False: means the path does not need to be calculated, only the value of the path is needed.

- srclist, the set of source points, default is None. int/list/None. can be of the following three types.

  1. int: an integer that represents a node in the graph as the source point for shortest path computation.
  2. list: a list, the elements of which are int means that the points in the graph are all source points and need to be computed separately for the corresponding shortest paths. 
  3. none: a null value, which means that the shortest path of all sources in the graph is calculated.

- block: The structure of the thread block, which is set automatically by the default program. it can only be a triple `(x, y, z)`.

- grid, the structure of the thread block, set automatically by default. can only be a binary `(x, y)`.

#### Return

The return value is an instance of `class Result`. Please refer to [class Result](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-result) for details.

<br>

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
        para: class, Parameter object. (see the 'SParry/classes/parameter.py/Parameter') 
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).  
    """
```

#### Parameters

- para, the parameter class passed between functions. `class Parameter`.  Please refer to [class Parameter](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-parameter) for details. The type is fixed.

#### Return

- bool, whether to enable graph segmentation.
  1. True: Indicates that the graph segmentation algorithm needs to be enabled.
  2. False: The graph segmentation algorithm does not need to be enabled.

<br>

### func draw()

#### Function

Draw the shortest path diagram of the first source based on the shortest path.

#### Structure

```python
def draw(path, s, graph):
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
    """
```

#### Parameters

- path, the precursor array of the shortest path, required. list.
- s, the number of the source vertex, required. int.
- graph, graph data, required. [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph). the graph data needed to calculate the shortest path.

#### Return

None

<br>

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
- name1, the first data name, the default is 'data1'. str.
- name2, the second data name, the default is 'data2'. str.

#### Return

str, equal or not equal

<br>

### func [apsp, sssp, mssp].'[dij, delta, edge, spfa]'_[cpu, gpu]

This tool encapsulates shortest path algorithms such as `Dijkstra`, `Bellman-Ford`, `Delta-Stepping`, `Edge-Based`. Support single source at the same time. Multi-source and full-source algorithms, CPU serial calculation versions and CUDA accelerated versions belong to multiple methods in multiple files, but have similar parameters and return values.

#### Function

Through the incoming parameters, the algorithm actually calculates the shortest path to solve the problem. The method name is different, the parameters are slightly different, and the calculation method enabled is also slightly different.

#### Structure

```python
def dijkstra(para):
	"""
    function: 
        use dijkstra algorithm in CPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SParry/classes/parameter.py/Parameter') 
    
    return: 
        class, Result object. (see the 'SParry/classes/result.py/Result') 
	"""
	...
```

#### Parameters

- para, the parameter class passed between functions. `class Parameter`.  Please refer to [class Parameter](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-parameter) for details. The type is fixed.

#### Return

The return value is an instance of `class Result`. Please refer to [class Result](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-result) for details.



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
        graph: class Graph, must, the graph data that you want to get the shortest path.
    
    method:
        display: 
            show the detail of this calculation.
        drawPath: 
            draw the path from vertices to the sources.
        calcPath:  
            calc the path through the graph and dist.
    
    return: 
        class, Result object. (see the 'SParry/classes/result.py/Result') 
	"""
	def __init__(self, 
                dist = None, 
                timeCost = None, 
                memoryCost = None, 
                graph = None): 
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

- dist, array of shortest paths. numpy.ndarray.
- timeCost, the time taken to calculate the shortest path. float.
- ~~memoryCost, the memory consumption during the calculation. None for now.~~
- graph, graph data, required, the graph data or graph data storage file to be used for calculating the shortest path. this function accepts this parameter must be [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph).



#### Attributes

- dist, the array for shortest path computation. numpy.ndarray. details are as follows.
  1. If computing a single-source shortest path, it is a one-dimensional array, `dist[i]` that is, the shortest distance from the source point to node `i` in the current single-source shortest path problem.
  2. If computing multiple source shortest paths, it is a two-dimensional array composed of multiple one-dimensional arrays, and `dist[i][j]` is the shortest distance from the `i`th source to the node `j` in the single-source shortest path problem for the `i`th source.
  3. If we calculate the full source shortest path, it is a two-dimensional array of `n × n`, and `dist[i][j]` means the shortest distance from source `i` to node `j` in the single source shortest path problem with source `i`.
- path, the path of the shortest path. numpy.ndarray. details are as follows.
  1. If computing a single-source shortest path, it is a one-dimensional array. `path[i]` means the predecessor node of node `i` in the current single-source shortest path problem, i.e., the source needs to reach node `i` from node `path[i]`.
  2. If computing multiple source shortest paths, it is a two-dimensional array consisting of multiple one-dimensional arrays, `path[i][j]` which means that in the single source shortest path problem for the `i` source, the `i` source needs to reach the node `j` from the node `path[i][j]`.
  3. If we compute the full source shortest path, then it is a two-dimensional array of `n × n`, `path[i][j]` which means that in the single source shortest path problem for source `i`, source `i` needs to go from node `path[i][j]` to reach node `j`.
- timeCost, time spent. float. indicates the time in seconds spent on the shortest path problem.
- memoryCost, space cost. int. denotes the space in bytes used to compute the shortest path problem.
- graph, graph data, required, the graph data or graph data storage file to be used to compute the shortest path. This function accepts this parameter must be [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph).

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

<br>

### class Parameter

#### Function

This class is used by the tool's internal parameter transfer. It records various parameters related to algorithms and has nothing to do with users.

#### Structure

```python
class Prameter(object):
	"""
    function: 
        transfer the parameters in the functions.

    parameters: 
        None, but 'self'.

    attributes:
    	graph: class Graph, must, the graph data that you want to get the shortest path.
        BLOCK: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        GRID: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        device: class, Device object. (see the 'SParry/classes/device.py/Device') 
        srclist: list/int, the source of shortest path problem.
        pathRecordBool: bool, record the path or not.
        part: int, the number of the edges that will put to GPU at a time.(divide algorithm)
    
    method:
        None, but init.
            
    return 
        class, Parameter object. (see the 'SParry/classes/parameter.py/Parameter') 
	"""
	def __init__(self):
		
		...
```

#### Parameters

None

#### Attributes

- graph, graph data, see [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph) for details.

- BLOCK, triple `(x,y,z)` denotes the structure of the threads during computation. tuple. optional.

- GRID, binary `(x,y)` denotes the structure of the block during computation. tuple. optional.

- useCUDA, whether to use CUDA acceleration, default is True. bool. can only be of the following types.

  1. True: Indicates that CUDA is used for acceleration. 
  2. False: Indicates that only the CPU is used for serial computation.

- useMultiPro, whether to use CPU multiprocess acceleration, default is False. bool. can only be of the following types: 1.

  1. True: Indicates that the CPU multiprocess is used for accelerated computation. 2.

  2. False: Indicates that the CPU multiprocess is not used for accelerated computation.

     **Note that this parameter is only meaningful if `method` is specified as `dij`, `useCUDA` is `False`, and the problem being solved is `APSP` or `MSSP`. **

- device, an instance of class Device. See [class Device](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-device) for details.

- srclist, a collection of source points. int/list/None. can be of the following three types.

  1. int: An integer that represents a node in the graph as a source point for shortest path computation.
  2. list: A list, each element of which is an int indicating that the points in the graph are all source points, and the corresponding shortest paths need to be calculated separately. 
  3. none: a null value indicating that the shortest path of the whole source is calculated.

- sourceType, the type of the problem to be solved. str. can be of the following three types.

  1. APSP: all-source shortest path
  2. MSSP: Multi-source shortest path
  3. SSSP: single source shortest path

- pathRecordBool, whether to record the path of the shortest path. bool. can only be of the following types: 

  1. True: means the path of the shortest path needs to be recorded. 
  2. False: means the path does not need to be calculated, only the value of the path is needed.

- part, the number of edges copied into the GPU at one time in the split graph algorithm. int.



#### Methods

None

<br>

### class Device

#### Function

This class is used by this tool to obtain the graphics card information of the device. The functions are implemented through `pyCUDA` and `pynvml`.

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

<br>

### class Graph

#### Function

Converts user-supplied graph data into a graph class that saves information about the graph.

#### Structure

```python
class Graph(object):
    """
    function: 
        a graph class.

    parameters:
        filename: str, must, the graph data file. 
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
        class, Graph object. (see 'SParry/classes/graph.py/Graph')
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
- CSR, CSRgraph data.
- matrix, adjacency matrix graph data.
- src, des, w, edge set array graph data.
- maxw, maximum edge weight. int. 
- minw, minimum edge weight. int. 
- MAXD, maximum degree. int. 
  MAXU, vertexnumber of maximum degree. int. 1
  MIND, minimum degree. int. 1
- minu, vertexnumber of minimum degree. int. 
- degree, degree of each node. list. 
- msg, message for calculation. str. 
- filename, the name of the file to be read. str.

#### Methods

- read, func, read the graph from the file.
  1. parameters, None.
  2. return, None.
- reshape, func, converts the data to a numpy.ndarray format.
  1. parameters, None.
  2. return, None.



------

所谓工欲善其事必先利其器。

`SParry` 是一个**最短路径计算工具包**，封装了诸如：`Dijkstra` , `Bellman-Ford` , `Delta-Stepping` ,  `Edge-Based` 等主流的最短路径算法。它也提供了**基于CUDA的并行加速版本**，以提高开发效率。

同时本工具还封装了自动分图计算方法的 `Dijkstra` 算法，可有效解决大规模图在 `GPU` 显存不足无法直接并行计算的问题。



------

## 安装

### 环境依赖

下面是开发实验中通过测试的环境。

**Window：**

> python 3.6/3.7
>
> pycuda 2019.1.2
>
> numpy 1.18.2
>
> CUDA Version  10.2.89
>
> networkx 2.4
>
> logging 0.5.1.2
>
> pynvml 8.0.4

**Linux**

> python 3.6
>
> pycuda 2019.1.2
>
> numpy 1.19.2
>
> CUDA Version 10.1.243
>
> networkx 2.4
>
> logging 0.5.1.2
>
> pynvml 8.0.4

<br>

### 安装

直接下载文件包，即可在**主目录**中运行 `pretreat.py` 中的 `read()` 函数预处理数据，然后就可以放入 `calc()` 接口函数。

**目前不是发行版本，故不可pip安装，开发结构尚不是很完善。**



------

## 流程图

![image](https://raw.githubusercontent.com/LCX666/SParry/main/chart.svg)





------

## 测试和效果

我们对此工具进行了大量的测试，在测试结果中无错误情况发生，统计了各个算法在部分图上计算**单源最短路径问题(SSSP)和全源最短路径问题(APSP)**的用时情况和串并行加速比。下面展示平均度为 4 时，各算法随结点数量变化的用时变化情况，更详细的数据请查阅 [SParry/testResult](https://github.com/LCX666/SParry/tree/main/testResult)。



![SSSPTimeConsumptionByDegree4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/SSSPTimeConsumptionByDegree4.png)

<center>SSSP time consumption by degree=4</center>



![SSSPSpeedupRatioByDegree4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/SSSPSppedupRatioByDegree4.png)

<center>SSSP speedup ratio by degree=4</center>

![APSPTimeConsumptionByDegreeAPSP4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/APSPTimeConsumptionByDegreeAPSP4.png)

<center>APSP time consumption by degree=4</center>

![APSPSpeedupRatioByDegree4](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/APSPSpeedupRatioByDegree4.png)

<center>APSP speedup ratio by degree=4</center>







------

## 快速入门教程

本节是帮助 `SParry` 新手快速上手的简介。

### step1. cd 到当前根目录

```powershell
cd XXX/SParry/
```



### step2. 导入计算接口

#### 内存数据-matrix

当您的图数据在**内存**中并**符合[邻接矩阵数据](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-adjacency-matrix)要求** 时，可以像下面一样导入您的数据，快速计算结果。

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> matrix = np.array([[0,1,2,3],[1,0,2,3],[2,2,0,4],[3,3,4,0]], dtype = np.int32) # 邻接矩阵的数据
>>> matrix # 模拟已经拥有了一个邻接矩阵的图数据
array([[0, 1, 2, 3],
       [1, 0, 2, 3],
       [2, 2, 0, 4],
       [3, 3, 4, 0]], dtype=int32)
>>>
>>> graph = read(matrix = matrix, # 传入的数据是邻接矩阵
...              method = "dij", # 计算方法使用 Dijkstra
...              detail = True # 记录图中的详细信息
...             ) # 处理图数据
>>>
>>> res = calc(graph = graph, # 传入图数据的类
...            useCUDA = True, # 使用 CUDA 加速
...            srclist = 0, # 设置源点为 0 号结点
...            )
>>>
>>> res.dist # 输出最短路径
array([0, 1, 2, 3], dtype=int32)
>>>
>>> print(res.display()) # 打印相关参数

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 16,
[+] the max edge weight in the Graph:                   MAXW = 4,
[+] the min edge weight in the Graph:                   MINW = 0,
[+] the max out degree in the Graph:                    degree(0) = 4,
[+] the min out degree in the Graph:                    degree(0) = 4,
[+] the average out degree of the Graph:                avgDegree = 4.0,
[+] the directed of the Graph:                          directed = Unknown,
[+] the method of the Graph:                            method = dij.


[+] calc the shortest path timeCost = 0.017 sec
>>>
>>> res.drawPath() # 红色的线表示此条边在最短路径上；橙色的点为源点；箭头表示边的方向；边上的数字表示边权。
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103113127.png" alt="image-20201104103113127"  />

#### 内存数据-CSR

当您的图数据在**内存**中并**符合[CSR数据](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E5%8E%8B%E7%BC%A9%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-csr)要求** 时，可以像下面一样导入您的数据，快速计算结果。

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> CSR = np.array([np.array([0, 2, 3, 4, 4]), 
...                 np.array([1, 2, 3, 1]), 
...                 np.array([1, 3, 4, 5])])
>>> CSR # 模拟已经拥有了CSR格式是图数据
array([array([0, 2, 3, 4, 4]), array([1, 2, 3, 1]), array([1, 3, 4, 5])],
      dtype=object)
>>>
>>> graph = read(CSR = CSR, # 传入的数据类型是 CSR
...              method = "delta", # 使用的算法为 delta-stepping
...              detail = True) # 记录图中的详细信息
>>>
>>> res = calc(graph = graph, # 传入的图数据类
...            useCUDA = True, # 使用 CUDA 并行加速
...            srclist = 0) # 源点为 0 号结点
>>>
>>> res.dist # 计算的最短路径
array([0, 1, 3, 5], dtype=int32)
>>>
>>> print(res.display()) # 打印相关参数

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 4,
[+] the max edge weight in the Graph:                   MAXW = 5,
[+] the min edge weight in the Graph:                   MINW = 1,
[+] the max out degree in the Graph:                    degree(0) = 2,
[+] the min out degree in the Graph:                    degree(3) = 0,
[+] the average out degree of the Graph:                avgDegree = 1.0,
[+] the directed of the Graph:                          directed = Unknown,
[+] the method of the Graph:                            method = delta.


[+] calc the shortest path timeCost = 0.007 sec
```



#### 内存数据-edgeSet

当您的图数据在**内存**中并**符合[edgeSet数据](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E8%BE%B9%E9%9B%86%E6%95%B0%E7%BB%84-edgeset)要求** 时，可以像下面一样导入您的数据，快速计算结果。

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> edgeSet = [[0, 0, 2, 1], # 每条边的起始点
...            [1, 3, 1, 3], # 每条边的结束点
...            [1, 2, 5, 4]] # 每条边的权值
>>>
>>> edgeSet # 模拟已经拥有了edgeSet格式是图数据
[[0, 0, 2, 1], [1, 3, 1, 3], [1, 2, 5, 4]]
>>>
>>> graph = read(edgeSet = edgeSet, # 传入的图数据是 edgeSet
...              detail = True) # 需要记录图中的数据
>>>
>>> res = calc(graph = graph, # 传入的图数据类
...            useCUDA = False, # 使用 CPU 串行计算
...            srclist = 0) # 源点为 0 号结点
>>> res.dist # 计算的最短路径
array([         0,          1, 2139045759,          2], dtype=int32)
>>>
>>> print(res.display()) # 打印相关参数

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 4,
[+] the max edge weight in the Graph:                   MAXW = 5,
[+] the min edge weight in the Graph:                   MINW = 1,
[+] the max out degree in the Graph:                    degree(0) = 2,
[+] the min out degree in the Graph:                    degree(3) = 0,
[+] the average out degree of the Graph:                avgDegree = 1.0,
[+] the directed of the Graph:                          directed = Unknown,
[+] the method of the Graph:                            method = dij.


[+] calc the shortest path timeCost = 0.0 sec
```



#### 文件数据

当您的图数据存储在**文件**中并**符合数据要求时** ([文件数据](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F))，您也可以传入文件来计算最短路径。

本例子中的文件如下， `test.txt` 。

```
4 6
0 1 1
0 2 2
0 3 3
1 2 2
1 3 3
2 3 4

```

代码如下：

```python
>>> from calc import calc
>>> import numpy as np
>>> from pretreat import read
>>>
>>> filename = "./data/test.txt" # 存储图数据的文件路径
>>> graph = read(filename = filename, # 传入的数据是文件
...              method = "spfa", # 使用的算法是 spfa
...              detail = True, # 记录图中的细节
...              directed = False) # 图为无向图
>>>
>>> res = calc(graph = graph, # 传入的图数据类
...            useCUDA = True, # 使用 CUDA 并行加速
...            srclist = 0) # 源点为 0 号结点
>>>
>>> res.dist # calculated shortest path
array([0, 1, 2, 3], dtype=int32)
>>>
>>> print(res.display()) # 打印相关参数

[+] the number of vertices in the Graph:                n = 4,
[+] the number of edges(directed) in the Graph:         m = 12,
[+] the max edge weight in the Graph:                   MAXW = 4,
[+] the min edge weight in the Graph:                   MINW = 1,
[+] the max out degree in the Graph:                    degree(0) = 3,
[+] the min out degree in the Graph:                    degree(0) = 3,
[+] the average out degree of the Graph:                avgDegree = 3.0,
[+] the directed of the Graph:                          directed = False,
[+] the method of the Graph:                            method = spfa.


[+] calc the shortest path timeCost = 0.002 sec
>>> res.drawPath() # 红色的线表示此条边在最短路径上；橙色的点为源点；箭头表示边的方向；边上的数字表示边权。
```

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201104103213127.png" alt="image-20201104113340700" style="zoom: 67%;" />

------





------

## 接口

### 版本信息

此工具尚在初始开发迭代版本，未上传 `pip` 站。

<br>

### 错误报告

工具中的一切错误报告都将以 `Python` 的错误信息进行报告，并在一些预想到错误的地方，嵌入了提示语句。

<br>

### 计算接口

#### func calc()

该函数是本工具的唯一计算接口，通过调用此方法可以直接计算得到图的最短路径。更多信息请参阅 [func calc()](https://github.com/LCX666/SParry/blob/main/tutorials.md#func-calc-3) 。



#### func read()

该函数是本工具中用于预处理图的函数，通过调用此方法可以将用户的各种图数据归一化为统一的类。更多信息请参阅 [func read()](https://github.com/LCX666/SParry/blob/main/tutorials.md#func-read-3) 。



#### func INF()

该函数会返回工具中的正无穷.



------

## 数据类型与规范

1. 本工具中目前的版本只支持32位整型的数据。因此，边权、点的数量和边的数量都必须是在32位整型能够表述的范围内。同时计算出的最短路径结果值也不应该超出32位整型的范围。关于此，一个解释是：目前大部分的数据问题都可以在32位整型范围内解决。其次是大部分非专业的英伟达GPU都对64位进行了阉割，因此不支持64位的数据。
2. 本工具中，所有的图默认都从结点0开始编号。因此从结点1开始编号的图应该无视第一个结点的数据，从第二个数据开始。因为此时工具中认为的图中的结点数会比真正的图中的结点数多1，工具始终认为还有一个0号结点。

<br>

### 图数据规范

工具中接收的图数据有[**文件格式**](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F-file-format)和**内存格式**(三种类型：[邻接矩阵(matrix)](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-adjacency-matrix)、[压缩邻接矩阵(CSR)](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E5%8E%8B%E7%BC%A9%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-csr)和[边集数组(edgeSet)](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E8%BE%B9%E9%9B%86%E6%95%B0%E7%BB%84-edgeset))。

下图是一个普通的图例子：

![image-20201023091301096.png](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201023091301096.png)

#### 文件格式 File Format

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
    - 若在 `func calc` 中指定 `directed = False` 时，即指定为无向图，本工具会自动将上述单向边转化为双向边，同时工具中的边数会自动翻倍，即用两条有向边表示一条无向边。
    - 若在 `func calc` 中指定 `directed = True` 时，即指定为有向图，本工具会严格按照单向边方向读图。
    - 默认参数 `directed = False` 默认图为无向图。
- 第二行~第五行： 3个整数，表示一条边
  - 第一个参数  表示当前边的起点
  - 第二个参数 表示当前边的终点
  - 第三个参数 表示当前边的边权
- 第六行（最后一行）：一个空行
  - **以一个空行表示文件的结束**



#### 邻接矩阵 Adjacency Matrix

`matrix` 邻接矩阵即用一个 `n×n` 二维数组来表示一个图，矩阵中的任意一个元素都可以代表一条边。即 `matrix[i][j] = w` 表示在图中存在至少一条从结点 `i` 到结点 `j` 的边，其边权为 `w` 。

由于本工具是最短路径计算工具，更严格地说， `matrix[i][j] = w` 应该是表示的结点 `i` 到结点 `j` 的所有边中最短的边的边权是 `w` 。

由于本工具是在32位整型的范围中进行计算，因此，本工具中的正无穷是一个很接近32位整型能表示的最大数的一个数。可以通过 `func: calc.INF()` 返回本工具的正无穷。

上转化为邻接矩阵如下：

```python
In [1]: matrix
    
Out[1]:
array([[0, 1, 3, 2139045759],
       [2139045759, 0, 2139045759, 4],
       [2139045759, 5, 0, 2139045759],
       [2139045759, 2139045759, 2139045759, 0]])
```



#### 压缩邻接矩阵 CSR

`CSR` 压缩邻接矩阵是本工具的主要存储和运算的方式，相较于邻接矩阵的存储方式可以在绝大多数情况下节约内存空间。同时在利用GPU进行加速计算时亦可以节约显存空间。其表示方式是三个一维数组： `V` 、 `E` 和 `W` ，在本工具中将上述三个数组按照 `V` 、 `E` 和 `W` 的顺序组合成 CSR 。

其中 `V` 数组是记录图中各个结点的第一条边的在 `E` 数组中的起始下标的，其维数是图中点的个数，但是为了计算的方便，通常会在末尾增加一个虚拟结点来判断是否到达末尾。因此其维数在本工具中必须是严格的包含了虚拟结点的。

`E` 数组是记录每一条边的终点是哪个结点。因此其维度是严格的图中的（有向）边的数目，本工具中通过两条有向边来表示一条无向边。

`W` 数组是记录与 `E` 数组对应的每一条边的边权，故其维度也是严格的图中的边的数目。

上转化为压缩邻接矩阵如下：

```python
In [1]: CSR = np.array(V, E, W)
    
Out[1]:array([array([0, 2, 3, 4, 4]), array([1, 2, 3, 1]), array([1, 3, 4, 5])],
      dtype=object)

In [2]: V
Out[2]: array([0, 2, 3, 4, 4])

In [3]: E
Out[3]: array([1, 2, 3, 1])

In [4]: W
Out[4]: array([1, 3, 4, 5])
```

**V 中在最后虚拟了一个结点，以确定边界。**



#### 边集数组 edgeSet

`edgeSet` 边集数组是一个列表，表中的每个元素都是表示一条边的三元组 `(u, v, w)` 即一条边的起点、终点和边权。

上图转化为边集数组如下：

```python
In [1]: edgeSet
Out[1]: [[0, 0, 2, 1], [1, 3, 1, 3], [1, 2, 5, 4]]
```

**如果是无向边需要在上述列表中表示成两条有向边再传入。**

<br>

### 常量

INF：工具中的正无穷，此处是 2139045759



------

## 函数 

### func read()

#### 功能

该函数是本工具中用于预处理图的函数，通过调用此方法可以将用户的各种图数据归一化为统一的类。

#### 结构

```python
def read(CSR = None, matrix = None, edgeSet = None, filename = "", method = "dij", detail = False, directed = "Unknown", delta = 3):
    """
    function: 
        convert a graph from [CSR/matrix/edgeSet/file] 
            to Graph object(see the 'SParry/classes/graph.py/Graph')
            as the paremeter 'graph' of the func calc(see the 'SParry/calc.py/calc')
    
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
        class, Graph object. (see the 'SParry/classes/graph.py/Graph')     
    """
```

#### parameters

- CSR，图数据，可选，满足 CSR 的数据格式，详见 [CSR 数据格式规范](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E5%8E%8B%E7%BC%A9%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-csr)。
- matrix，图数据，可选，满足邻接矩阵的数据格式，详见 [matrix 数据格式规范](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-adjacency-matrix)。
- edgeSet，图数据，可选，满足边集数组的数据格式，详见 [edgeSet 数据格式规范](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E8%BE%B9%E9%9B%86%E6%95%B0%E7%BB%84-edgeset)。
- filename，图文件名，可选，满足文件格式的数据格式，详见 [file 数据格式规范](https://github.com/LCX666/SParry/blob/main/tutorials.md#%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F-file-format)。
- **注意！上述四个参数虽然都是可选，但是4者不能都为 None。同时优先级为：CSR > matrix > edgeSet > filename。**
- method， 使用的最短路径计算方法，缺省为 'dij'。str。仅可以是以下类型：
  1. dij： 表示使用 `Dijkstra` 算法求解最短路径。
  2. spfa： 表示使用 `Bellman-Ford` 算法求解最短路径。
  3. delta： 表示使用 `Delta-Stepping` 算法求解最短路径。
  4. edge： 表示使用边细粒度来计算最短路径。
- detail，是否需要统计图中的详细信息，缺省为 False。bool。如果设置为 True，可能需要花费更多的时间进行图的数据读取。
- directed，图是否有向，缺省为 False。bool。只有在输入参数 filename 时才有效。
- delta，`Delta-Stepping` 算法的参数 delta，缺省为 3。int。只有在选择算法为 `Delta-Stepping` 时才有效。

#### 返回值

返回值是 `class Graph` 的一个实例。详细请参阅 [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph-1)。

<br>

### func calc()

#### 功能 

该函数是 [SParry](https://github.com/LCX666/SParry) 的接口函数，通过该函数，用户可以传入一些必要的参数从而计算得到图的最短路径。

#### 结构

```python
def calc(graph = None, useCUDA = True, useMultiPro = False, pathRecordBool = False, srclist = None, block = None, grid = None, namename = None):
    
    """
    function: 
        a calculate interface.
    
    parameters: 
        graph: class Graph, must, the graph data that you want to get the shortest path.
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].(more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.

    return:
        class, Result object. (see the 'SParry/classes/result.py/Result') 
    """
```

#### parameters

该方法是接口函数，各个参数意义如下：

- graph， 图数据，必填，需要计算最短路径的图数据或者图数据存储文件。本函数接受的此参数必须是 [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph-1)。

- useCUDA，是否使用 CUDA 加速，缺省为 True。bool。仅可以是以下类型：
  1. True： 表示使用 CUDA 进行加速。
  2. False： 表示只使用 CPU 进行串行计算。
  
- useMultiPro，是否使用CPU多进程加速计算，缺省为False。bool。仅可以是以下类型：

  1. True： 表示使用CPU多进程进行加速计算。

  2. False：表示不使用CPU多进程进行加速计算。

     **需要注意的是，只有当 method 指定为 dij ， useCUDA 为 False ，解决的问题是 APSP 或者 MSSP 时，此参数才有意义。**

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

返回值是 `class Result` 的一个实例。详细请参阅  [class Result](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-result-1)。

<br>

### func dispatcher()

#### 功能

任务调度函数，依据用户的输入数据判断程序的走向，调整程序的运行以及参数合法的检验。

#### 结构

```python
def dispatch(graph, useCUDA, useMultiPro, pathRecordBool, srclist, block, grid):
    """
    function: 
        schedule the program by passing in parameters.
    
    parameters: 
        graph: str/list/tuple, must, the graph data that you want to get the shortest path.
            (more info please see the developer documentation).
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        pathRecordBool: bool, record the path or not.
        srclist: int/lsit/None, the source list, can be [None, list, number].
            (more info please see the developer documentation).
        block: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        grid: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
    
    return:
        class, Result object. (see the 'SParry/classes/result.py/Result').
    """

   
    return result
```

#### parameters

该方法是接口函数承接的转入函数，各个参数意义都与接口函数一致，各个参数意义如下：

- graph，图数据，必填，需要计算最短路径的图数据或者图数据存储文件。本函数接受的此参数必须是 [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph-1)。
- useCUDA，是否使用 CUDA 加速，缺省为 True。bool。仅可以是以下类型：
  1. True： 表示使用 CUDA 进行加速。
  2. False： 表示只使用 CPU 进行串行计算。

- useMultiPro，是否使用CPU多进程加速计算，缺省为False。bool。仅可以是以下类型：

  1. True： 表示使用CPU多进程进行加速计算。

  2. False：表示不使用CPU多进程进行加速计算。

     **需要注意的是，只有当 `method` 指定为 `dij` ， `useCUDA` 为 `False` ，解决的问题是 `APSP` 或者 `MSSP` 时，此参数才有意义。**
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

返回值是 `class Result` 的一个实例。详细请参阅 [class Result](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-result-1)。

<br>

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
        para: class, Parameter object. (see the 'SParry/classes/parameter.py/Parameter') 
    
    return:
        bool, [0/1/2]. (more info please see the developer documentation).  
    """
```

#### parameters

- para，para，函数间传递的参数类。`class Parameter` 的一个实例。详细请参阅 [class Parameter](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-parameter-1)。类型固定。

#### 返回值

- int，是否需要启用图分割。
  1. 0： 表示并不需要使用图分割算法，可以直接放入 GPU 中。
  2. 1： 表示需要使用图分割，将图分解成更小的部分。
  3. 2： 表示多源问题或者全源问题可以直接解决。



<br>

### func draw()

#### 功能

依据最短路径绘制第一个源的最短路径图。

#### 结构

```python
def draw(path, s, graph):
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
    """
```

#### parameters

- path，最短路径的前驱数组，必填。list。
- s，源点的编号，必填。int。
- graph， 图数据，必填。[class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph-1)。 需要计算最短路径的图数据。

#### 返回值

无

<br>

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

<br>

### func [apsp, sssp, mssp].'[dij, delta, edge, spfa]'_[cpu, gpu]

本工具中封装了`Dijkstra`、`Bellman-Ford`、`Delta-Stepping`、`Edge-Based` 最短路径算法。同时支持单源。多源和全源的算法和使用 CPU 串行计算的版本和 CUDA 加速的版本，属于多个文件中的多个方法，但是具有相似的参数和返回值。

#### 功能

通过传入的参数，利用算法实际计算出最短路径，解决问题。方法名不同，参数略有不用，启用的计算方法也略有差异。

#### 结构

```python
def dijkstra(para):
	"""
    function: 
        use dijkstra algorithm in CPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SParry/classes/parameter.py/Parameter') 
    
    return: 
        class, Result object. (see the 'SParry/classes/result.py/Result') 
	"""
	...
```

#### parameters

para，函数间传递的参数类。`class Parameter` 的一个实例。详细请参阅 [class Parameter](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-parameter-1)。类型固定。

#### 返回值

返回值是 `class Result` 的一个实例。详细请参阅 [class Result](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-result-1)。



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
        graph: class Graph, must, the graph data that you want to get the shortest path.
    
    method:
        display: 
            show the detail of this calculation.
        drawPath: 
            draw the path from vertices to the sources.
        calcPath:  
            calc the path through the graph and dist.
    
    return: 
        class, Result object. (see the 'SParry/classes/result.py/Result') 
	"""
	def __init__(self, 
                dist = None, 
                timeCost = None, 
                memoryCost = None, 
                graph = None): 
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

- dist, 最短路径计算的数组。numpy.ndarray。
- timeCost, 计算最短路径用时。float。
- ~~memoryCost， 计算过程中的内存占用。暂时无。~~
- graph，图数据，必填，需要计算最短路径的图数据或者图数据存储文件。本函数接受的此参数必须是 [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph-1)。

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
- graph，图数据，必填，需要计算最短路径的图数据或者图数据存储文件。本函数接受的此参数必须是 [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph-1)。

#### 方法

- display，func，将计算详情 `msg` 信息展示给用户。
  1. parameters，无。
  2. 返回值，msg 计算信息。str。
- drawPath，func，将计算的最短路径可视化绘制。
  1. parameters， 无。
  2. 返回值，无。
- calcPath，func，从dist数组和图中复现出 path。直接赋值给 `path` 属性。
  1. parameters， 无。
  2. 返回值， 无。



<br>

### class Parameter

#### 功能

该类是此工具的内部传参使用的类，记录了各类与算法相关的参数，与用户无关。

#### 结构

```python
class Prameter(object):
	"""
    function: 
        transfer the parameters in the functions.

    parameters: 
        None, but 'self'.

    attributes:
    	graph: class Graph, must, the graph data that you want to get the shortest path.
        BLOCK: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        GRID: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        device: class, Device object. (see the 'SParry/classes/device.py/Device') 
        srclist: list/int, the source of shortest path problem.
        pathRecordBool: bool, record the path or not.
        part: int, the number of the edges that will put to GPU at a time.(divide algorithm)
    
    method:
        None, but init.
            
    return 
        class, Parameter object. (see the 'SParry/classes/parameter.py/Parameter') 
	"""
	def __init__(self):
		
		...
```

#### parameters

无

#### 属性

- graph，图数据，详见 [class Graph](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-graph-1)。

- BLOCK,  三元组 `(x,y,z)` 表示计算过程中线程的结构。tuple。选填。

- GRID，二元组 `(x,y)`  表示计算过程中block的结构。tuple。选填。

- useCUDA，是否使用 CUDA 加速，缺省为 True。bool。仅可以是以下类型：
  1. True： 表示使用 CUDA 进行加速。
  2. False： 表示只使用 CPU 进行串行计算。
  
- useMultiPro，是否使用 CPU 多进程加速计算，缺省为False。bool。仅可以是以下类型：

  1. True： 表示使用 CPU 多进程进行加速计算。

  2. False：表示不使用 CPU 多进程进行加速计算。

     **需要注意的是，只有当 `method` 指定为 `dij` ， `useCUDA` 为 `False` ，解决的问题是 `APSP` 或者 `MSSP` 时，此参数才有意义。**
  
- device，class Device 的一个实例。详情请见 [class Device](https://github.com/LCX666/SParry/blob/main/tutorials.md#class-device-1)。

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
  
- part，分图算法中一次拷贝进 GPU 中的边的数量。int。

#### 方法

无

<br>

### class Device

#### 功能

该类是此工具获取设备显卡信息的类，通过 `pyCUDA` 和 `pynvml` 实现功能。

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

- device，pyCUDA 的 Device 类。object。
- CUDAVersion， CUDA 的版本。str。
- driverVersion，驱动的版本。str。
- deviceName， GPU 设备的型号。str。
- deviceNum，CUDA 设备数量。int。
- total，总显存的容量。int。
- free，空闲显存的容量。int。
- used，已经使用的显存容量。int。
- temperature，设备的温度。str。
- powerStstus，电源的状态。str。

#### 方法

- getDeviceInfo，func，获取分图需要计算的显存剩余信息。
  1. parameters，无。
  2. 返回值，无。

<br>

### class Graph

#### 功能

将用户提供的图数据转化为一个图类保存图的信息。

#### 结构

```python
class Graph(object):
    """
    function: 
        a graph class.

    parameters:
        filename: str, must, the graph data file. 
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
        class, Graph object. (see 'SParry/classes/graph.py/Graph')
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
- reshape，func，将数据转化为 numpy.ndarray 的格式。
  1. parameters，无。
  2. 返回值，无。