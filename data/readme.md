This folder is for some simple graph data samples.

Take the file `data_10_20.txt` as an example, it means that there are 10 nodes and 20 edges in this graph. The data in it are as follows.

```
10 20
0 1 41
0 2 78
0 9 79
0 7 43
0 7 16
1 3 23
1 4 51
1 6 36
1 1 9
1 6 90
2 5 38
2 3 27
3 8 34
3 6 87
5 7 18
6 8 34
6 7 5
7 9 50
7 9 22
7 9 67

```

The first row of the data indicates that the graph has 10 nodes and 20 edges. The next 2 to 21 lines represent the three parameters of a directed edge (or undirected edge, as defined by the user), which are the starting node, the ending node and the edge weight, in that order. The end of the file is indicated by a blank line. More information please see [file-format](https://github.com/LCX666/SParry/blob/main/tutorials.md#file-format)。