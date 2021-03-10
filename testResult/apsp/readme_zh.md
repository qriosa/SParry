本文件夹是全源最短路径计算的测试结果。

测试的基本方法是：对于同一个图，分别调用工具中封装的四种方法的 CUDA 版本和 单核 CPU 版本进行计算，并将计算的结果与 dijkstra 算法的单核 CPU 版本进行对比。同时记录各个算法和各个版本的用时。

------

`GPU&singleCPU.csv` 包含了本工具中各个算法的 CUDA 版本与单核 CPU 版本的计算对比情况。

| 列编号 | 列名                                    | 含义                                                      |
| ------ | --------------------------------------- | --------------------------------------------------------- |
| A      | id                                      | 各行行号                                                  |
| B      | n                                       | 此测试图中的结点数量                                      |
| C      | m                                       | 此测试图中的边的数量                                      |
| D      | avgD                                    | 此测试图的平均度 (m/n)                                    |
| E      | method                                  | 此结果所使用的方法                                        |
| F      | GPU_time_cost                           | CUDA 加速版本的用时                                       |
| G      | dist:GPU(method)==singleCPU(dij)?       | 判断此 CUDA 加速版本结果是否与 dijkstra CPU版本的结果相等 |
| H      | singleCPU_time_cost                     | 单核 CPU 版本的用时                                       |
| I      | dist:singleCPU(method)==singleCPU(dij)? | 判断此 CUDA 加速结果是否与 dijkstra CPU版本的结果相等     |
| J      | singleCPU_time_cost/GPU_time_cost       | 加速比                                                    |

------

`singCPU&multiCPU.csv` 包含了本工具中，dijkstra 算法单核 CPU 与 8核CPU启用多进程+共享内存的计算对比情况。

| 列编号 | 列名                                              | 含义                                                       |
| ------ | ------------------------------------------------- | ---------------------------------------------------------- |
| A      | id                                                | 各行行号                                                   |
| B      | n                                                 | 此测试图中的结点数量                                       |
| C      | m                                                 | 此测试图中的边的数量                                       |
| D      | avgD                                              | 此测试图的平均度 (m/n)                                     |
| E      | method                                            | 此结果所使用的方法                                         |
| F      | TimeSingleCPU(s)                                  | 单核 CPU 版本的用时                                        |
| G      | TimeMultiCPUs(s)                                  | 8核 CPU 版本的用时                                         |
| H      | dist:multiCPU(dij)==singleCPU(dij)?               | 判断此单核 CPU 版本结果是否与 dijkstra 8核 CPU版本结果相等 |
| I      | singleCPU_time_cost/multiCPUs(8  cores)_time_cost | 加速比                                                     |