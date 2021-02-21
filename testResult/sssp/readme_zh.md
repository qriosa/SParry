本文件夹是单源最短路径计算的测试结果。

测试的基本方法是：对于同一个图，分别调用工具中封装的四种方法的 CUDA 版本和 单核 CPU 版本进行计算，并将计算的结果与 dijkstra 算法的单核 CPU 版本进行对比。同时记录各个算法和各个版本的用时。

------

`GPU&singleCPU.csv` 包含了本工具中各个算法的 CUDA 版本与单核 CPU 版本的计算对比情况。

| 列编号 | 列名                                    | 含义                                                       |
| ------ | --------------------------------------- | ---------------------------------------------------------- |
| A      | id                                      | 各行行号                                                   |
| B      | n                                       | 此测试图中的结点数量                                       |
| C      | m                                       | 此测试图中的边的数量                                       |
| D      | avgD                                    | 此测试图的平均度 (m/n)                                     |
| E      | method                                  | 此结果所使用的方法                                         |
| F      | GPU_time_cost(s)                        | CUDA 加速版本的用时                                        |
| G      | dist:GPU(method)==singleCPU(dij)?       | 判断此 CUDA 加速版本结果是否与 dijkstra CPU 版本的结果相等 |
| H      | singleCPU_time_cost(s)                  | 单核 CPU 版本的用时                                        |
| I      | dist:singleCPU(method)==singleCPU(dij)? | 判断此 CUDA 加速结果是否与 dijkstra CPU 版本的结果相等     |
| J      | singleCPU_time_cost/GPU_time_cost       | 加速比                                                     |
