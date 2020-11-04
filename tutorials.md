# [SPoon](https://github.com/LCX666/SPoon)

![image](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/spoon_logo.png)

**SPoon** is a shortest path calc tool using some algorithms with cuda to speedup.

It's **developing**.



------

## English version

`spoon` is a **shortest path calculation toolkit**, the main shortest path algorithms, including `Dijkstra`, `Bellman-Ford`, `Delta-Stepping`, and `Edge-Threads`, are encapsulated. It also provides **a parallel accelerated version based on CUDA is provided** to improve development efficiency.

At the same time, it can divide the graph data into parts, and solve it more quickly than using the CPU when the graph is too large to put it in the GPU directly.



### Installation

#### Environment & Dependence

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

#### Installation

Download the file package directly and run the `calc` interface function in the main directory.

**It's not not a release version currently, so it cannot be installed with pip, and the development structure is not yet perfect. **



### Quick start tutorial

This section is an introduction to help beginners of `SPoon` get started quickly.

#### step1. cd to the current root directory

```powershell
cd XXX/spoon/
```



#### step2. Import calculation interface

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



### More.

**See the developer documentation for more information. **



------

## 中文版本

所谓工欲善其事必先利其器。

`SPoon` 是一个**最短路径计算工具包**，封装了诸如：`Dijkstra`, `Bellman-Ford`,`Delta-Stepping`, `Edge-Threads` 等主流的最短路径算法。它也提供了**基于CUDA的并行加速版本**，以提高开发效率。

同时本工具还封装了自动分图计算方法的 `dijkstra` 算法，可有效解决大规模图在GPU显存不足无法直接并行计算的问题。



### 安装

#### 环境依赖

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

#### 安装

直接下载文件包，即可在主目录中运行 `calc` 接口函数。

**目前不是发行版本，故不可pip安装，开发结构尚不是很完善。**



### 快速入门教程

本节是帮助 `SPoon` 新手快速上手的简介。

#### step1. cd 到当前根目录

```powershell
cd XXX/spoon/
```



#### step2. 导入计算接口

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



### 接口

#### 版本信息 Version Queries

此软件尚在初始开发迭代版本，未上传pip站。



#### 错误报告 Error Reporting

软件中的一切错误报告都将以 python 的错误信息进行报告，并在一些预想到错误的地方，嵌入了提示语句。



#### 计算接口 calculate interface

##### func calc

该方法是本软件的唯一计算接口，通过调用此方法可以直接计算得到图的最短路径。更多信息请参阅 `func calc`。超链接



### 数据类型与规范 data type

1. 本工具中目前的版本只支持32位整型的数据。因此边权，点的数量和边的数量都必须是在32位整型能够表述的范围内，计算出的最短路径结果值也不应该超出32位整型的范围。关于此，一个解释是：目前大部分的数据问题都可以在32位整型范围内解决。其次是大部分非专业的英伟达GPU都对64位进行了阉割，因此不支持64位的数据。
2. 本工具所有的图默认从结点0开始编号，则从结点1开始编号的图应该无视第一个结点的数据，从第二个数据开始查看。（此时软件中认为的图中的结点数会比真正的图中的结点数多1，因为软件始终认为还有一个0号结点。

#### 图数据规范

软件中接收的图数据有**文件格式**和**内存格式**(三种类型：邻接矩阵(matrix)、压缩邻接矩阵(CSR)和边集数组(edgeSet))。

下图是一个普通的图例子：

![image-20201023091301096.png](https://cdn.jsdelivr.net/gh/LCX666/picgo-blog/img/image-20201023091301096.png)

##### 文件格式

上述图存储在文件中时，格式应该如下：

```
4 4
0 1 1
0 2 3
1 3 4
2 1 5

```

- 第一行 
  - 第一个参数 n = 4 表示此图一共有4个结点。 
  - 第二个参数 m = 4  表示此图一共有**4**条**有向/无向**边。
    - 若在 `func clac` 中指定 `directed = False` 时，即指定为无向图，本软件会自动将上述单向边转化为双向边，同时软件中的边数会自动翻倍，即用两条有向边表示一条无向边。
    - 若在 `func clac` 中指定 `directed = True时，即指定为有向图，本软件会严格按照单向边方向读图。
    - 默认参数 `directed = False` 默认图为无向图。



##### 邻接矩阵

邻接矩阵即用一个n×n二维数组来表示一个图，矩阵中的任意一个元素都可以代表一条边。即 `matrix[i][j] = w` 表示在图中存在至少一条从结点 `i` 到结点 `j` 的边，其边权为 `w` 。

由于本工具是最短路径计算工具，更严格地说， `matrix[i][j] = w` 应该是表示的结点 `i` 到结点 `j` 的所有边中最短的边的边权是 `w` 。

由于本软件是在32位整型的范围中进行计算，因此，本软件中的正无穷是一个很接近32位整型能表示的最大数的一个数。可以通过 `func: calc.INF` 返回本软件的正无穷。

图一转化为邻接矩阵如下：

```python
In [1]: matrix
    
Out[1]:
array([[0, 1, 3, 2139045759],
       [2139045759, 0, 2139045759, 4],
       [2139045759, 5, 0, 2139045759],
       [2139045759, 2139045759, 2139045759, 0]])
```



##### 压缩邻接矩阵

压缩邻接矩阵是本软件的主要存储和运算的方式，相较于邻接矩阵的存储方式可以在绝大多数情况下节约内存空间。同时在利用GPU进行加速计算时亦可以节约显存空间。其表示方式是三个一维数组： `V` 、 `E` 和 `W` ，在本软件中将上述三个数组按照 `V` 、 `E` 和 `W` 的顺序组合成 CSR 。

其中 `V` 数组是记录图中各个结点的第一条边的在 `E` 数组中的起始下标的，其维数是图中点的个数，但是为了计算的方便，通常会在末尾增加一个虚拟结点来判断是否到达末尾。因此其维数在本软件中必须是严格的包含了虚拟结点的。

`E` 数组是记录每一条边的终点是哪个结点。因此其维度是严格的图中的（有向）边的数目，本软件中通过两条有向边来表示一条无向边。

`W` 数组是记录与 `E` 数组对应的每一条边的边权，故其维度也是严格的图中的边的数目。

图一转化为邻接矩阵如下：

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



##### 边集数组

边集数组是一个列表，表中的每个元素都是表示一条边的三元组 `(u, v, w)` 即一条边的起点、终点和边权。

上述图一转化为边集数组如下：

```python
In [1]: edgeSet
Out[1]: [(0, 1, 1), (0, 3, 2), (2, 1, 5), (1, 3, 4)]
```

**如果是无向边需要在上述列表中表示成两条有向边再传入。**