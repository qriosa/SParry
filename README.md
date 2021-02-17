# [SParry](https://github.com/LCX666/SParry)

![image](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/sparry.png)

**SParry** is a shortest path calculating tool using some algorithms with CUDA to speedup.

[English Version](https://github.com/LCX666/SParry/blob/main/README.md)|[中文版](https://github.com/LCX666/SParry/blob/main/README_zh.md)

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



### More.

Please see the [developer tutorials](https://github.com/LCX666/SParry/blob/main/tutorials.md#interface) for more information. 
