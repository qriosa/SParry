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



### 更多

**更多信息请参阅开发者文档。**







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

- [ ] 但SPFA多源出现了奇怪的bug，每次只计算了一个源
- [ ] 绘制的图还得漂亮



### 11/4

- 修复了绘图中的源点问题
- 修复了绘图中的点分布过于稠密的问题
- 在读图函数中添加了对单向边的支持
- 完善了readme
- 完成了开发者文档编写



## ToDo

- [ ] fw 或者 matrix 给跑起来
- [x] 开发者文档
- [ ] 用法截图
- [ ] 大规模集成测试
- [ ] 简单usage