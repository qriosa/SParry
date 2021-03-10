This folder shows the test results of the single-source shortest path computation.

The basic approach of the test is to call the CUDA version and the single-core CPU version of each of the four methods wrapped in the tool for the same graph, and compare the results with the single-core CPU version of the dijkstra algorithm. The time taken for each algorithm and each version is also recorded.

------

`GPU&singleCPU.csv` contains the computational comparison between the CUDA version and the single-core CPU version of each algorithm in this tool.

| Column Number | Column Name                             | Meaning                                                      |
| ------------- | --------------------------------------- | ------------------------------------------------------------ |
| A             | id                                      | Line number                                                  |
| B             | n                                       | Number of vertices in this test graph                        |
| C             | m                                       | Number of edges in this test graph                           |
| D             | avgD                                    | Average degree of this test graph(m/n)                       |
| E             | method                                  | The method used for this result                              |
| F             | GPU_time_cost                           | Time spent on the accelerated version of CUDA                |
| G             | dist:GPU(method)==singleCPU(dij)?       | Determine if the result of this CUDA acceleration version is equal to the result of the dijkstra CPU version |
| H             | singleCPU_time_cost                     | Time taken for single-core CPU version                       |
| I             | dist:singleCPU(method)==singleCPU(dij)? | Determine if the result of this CUDA acceleration is equal to the result of the dijkstra CPU version |
| J             | singleCPU_time_cost/GPU_time_cost       | Acceleration ratio                                           |

