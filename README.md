# [SPoon](https://github.com/LCX666/SPoon)

![image](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_logo.jpg)

**SPoon** is a shortest path calculating tool using some algorithms with CUDA to speedup.

It's **developing**.



------

`SPoon` is a **shortest path calculation toolkit**, the main shortest path algorithms, including `Dijkstra`, `Bellman-Ford`, `Delta-Stepping`, and `Edge-Based`, are encapsulated. It also provides **a parallel accelerated version based on CUDA is provided** to improve development efficiency.

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

![image](https://raw.githubusercontent.com/LCX666/SPoon/main/chart.svg)



------

## Test&Result

We have conducted a lot of tests on this tool, and no errors occurred in the test results. We have counted the time consumption and serial-parallel speedup ratio of each algorithm on some graphs to solve the **single-source shortest path (SSSP) and all-pairs shortest path (APSP)**. The following shows the running effect diagram of each algorithm in the figure of `M=10*N`. For more detailed data, please refer to [SPoon/testResult](https://github.com/LCX666/SPoon/tree/main/testResult) .

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_cpu_timecost_by_N_M=10N.png" alt="spoon_cpu_timecost_by_N_M=10N" style="zoom: 50%;" />

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_gpu_timecost_by_N_M=10N.png" alt="spoon_gpu_timecost_by_N_M=10N" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_speedUp_GPUoverCPU_by_N_M=10N.png" alt="spoon_speedUp_GPUoverCPU_by_N_M=10N" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_speedUp_CPU_Dij_over_8methods_by_N_M=10N.png" alt="spoon_speedUp_CPU_Dij_over_8methods_by_N_M=10N" style="zoom:50%;" />



------

## Quick start tutorial

This section is an introduction to help beginners of `SPoon` get started quickly.

### step1. cd to the current root directory

```powershell
cd XXX/SPoon/
```



### step2. Import calculation interface

#### In Memory-matrix

When your graph data is in **memory** and **meets the [adjacency-matrix data](https://github.com/LCX666/SPoon/blob/main/tutorials.md#adjacency-matrix)** requirements, you can import your data as below to quickly calculate the results.

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

When your graph data is in **memory** and **compliant with the [CSR data](https://github.com/LCX666/SPoon/blob/main/tutorials.md#csr) requirements**, you can import your data as below to quickly calculate the results.

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

When your graph data is in **memory** and **compliant with the [edgeSet data](https://github.com/LCX666/SPoon/blob/main/tutorials.md#edgeset)** requirements, you can import your data as below to quickly calculate the results.

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

When your graph data is stored in a **file** and **meets the data requirements** ([file data](https://github.com/LCX666/SPoon/blob/main/tutorials.md#file-format)) , you can also pass in a file to calculate the shortest path.

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

Please see the [developer tutorials](https://github.com/LCX666/SPoon/blob/main/tutorials.md#interface) for more information. 



------

所谓工欲善其事必先利其器。

`SPoon` 是一个**最短路径计算工具包**，封装了诸如：`Dijkstra` , `Bellman-Ford` , `Delta-Stepping` ,  `Edge-Based` 等主流的最短路径算法。它也提供了**基于CUDA的并行加速版本**，以提高开发效率。

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

![image](https://raw.githubusercontent.com/LCX666/SPoon/main/chart.svg)





------

## 测试和效果

我们对此工具进行了大量的测试，在测试结果中无错误情况发生，统计了各个算法在部分图上计算**单源最短路径问题(SSSP)和全源最短路径问题(APSP)**的用时情况和串并行加速比。下面展示了 `M=10*N` 的图中各个算法的运行效果图，更详细的数据请查阅 [SPoon/testResult](https://github.com/LCX666/SPoon/tree/main/testResult)。

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_cpu_timecost_by_N_M=10N.png" alt="spoon_cpu_timecost_by_N_M=10N" style="zoom: 50%;" />

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_gpu_timecost_by_N_M=10N.png" alt="spoon_gpu_timecost_by_N_M=10N" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_speedUp_GPUoverCPU_by_N_M=10N.png" alt="spoon_speedUp_GPUoverCPU_by_N_M=10N" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_speedUp_CPU_Dij_over_8methods_by_N_M=10N.png" alt="spoon_speedUp_CPU_Dij_over_8methods_by_N_M=10N" style="zoom:50%;" />



------

## 快速入门教程

本节是帮助 `SPoon` 新手快速上手的简介。

### step1. cd 到当前根目录

```powershell
cd XXX/SPoon/
```



### step2. 导入计算接口

#### 内存数据-matrix

当您的图数据在**内存**中并**符合[邻接矩阵数据](https://github.com/LCX666/SPoon/blob/main/tutorials.md#%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-adjacency-matrix)要求** 时，可以像下面一样导入您的数据，快速计算结果。

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

当您的图数据在**内存**中并**符合[CSR数据](https://github.com/LCX666/SPoon/blob/main/tutorials.md#%E5%8E%8B%E7%BC%A9%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5-csr)要求** 时，可以像下面一样导入您的数据，快速计算结果。

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

当您的图数据在**内存**中并**符合[edgeSet数据](https://github.com/LCX666/SPoon/blob/main/tutorials.md#%E8%BE%B9%E9%9B%86%E6%95%B0%E7%BB%84-edgeset)要求** 时，可以像下面一样导入您的数据，快速计算结果。

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

当您的图数据存储在**文件**中并**符合数据要求时** ([文件数据](https://github.com/LCX666/SPoon/blob/main/tutorials.md#%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F))，您也可以传入文件来计算最短路径。

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



### 更多

更多信息请参阅[开发者文档](https://github.com/LCX666/SPoon/blob/main/tutorials.md#%E6%8E%A5%E5%8F%A3)。







## 开发日志

### 10/29

- 完成了前期工作的总结和整合。
- 并新建了仓库。
- 测试了异步协同编程



### 10/30

- 绘制了新流程图
- 依据新流程修改程序流程与接口



### 10/31

- 对修改后的包进行简单测试
- 重新修改parameter中的属性
- 对接了读入函数
- debug delta-stepping & edge



### 11/1

- 集成化测试了各个用例
- 编写了使用算法的伪代码
- 增加了block和grid的属性参数



### 11/2

- 完成了自动化测试用例

- 修复了函数接口错误

- 完善分图路由，并测试和修复了分图中bug

  

### 11/3

- 完成了新结构代码的注释编写
- 在每个函数中添加了logger功能
- 去除了调整结构后的一些bug
- 添加了requirement
- 完成了path recording功能
- 重构和设计了Result类
- 去除了无用的另一些文件和方法
- 实现了文本display函数，展示本次计算的相关信息
- 实现了绘路径图功能 但是图还不理想

- [x] 但SPFA多源出现了奇怪的bug，每次只计算了一个源
- [x] 绘制的图还得漂亮



### 11/4

- 修复了绘图中的源点问题
- 修复了绘图中的点分布过于稠密的问题，分布点于同心圆上
- 在读图函数中添加了对单向边的支持
- 完善了readme
- 完成了开发者文档编写
- 修复了SPFA多源只算一个源的feature
- 修复了SPFA对有向图的支持feature



### 11/5

- 修复开发者文档中漏下的 func userTest 
- ~~测试加速情况~~
- 用法截图
- drawPath 添加了去重
- 修复了单源edge sssp 未知的错误。（个人PC正确，但集群中会一直错误



### 11/11

- 上传了流程图
- 代码注释中添加了类的位置



### 11/13

- 添加了部分全源的测试运行数据



### 1/24

- 尽力改注释为英文
- 修复全源 dij cu 的 bug
- 修改了接口和数据预处理
- 修改了 readme 和 开发者文档
- 添加了测试全源和单源的数据



## ToDo

- [ ] ~~fw 或者 matrix 给跑起来(新文章)~~
- [x] 开发者文档
- [x] 用法截图
- [x] 大规模集成测试
- [x] 简单usage
- [x] 在哪个文件夹下各个类
- [ ] ~~在开发者文档中添加各个类~~
- [x] 利用集群上跑的数据对分图实际效果进行测试
- [x] ~~添加多线程支持，待修改开发者文档到一致，但是在开倒车~~
- [x] 对集群 dijkstra 并行的不一样进行测试
- [x] 转入多进程
- [x] 获取的part多源和全源还需要修改计算公式
- [x] 更新 judge 的文档 分图的文档 多进程接口的文档
- [x] 更新了 logo
- [x] 修复文档中的一些纰漏
- [x] 添加换行和分割线 
- [x] 英文大小写 和 逗号冒号