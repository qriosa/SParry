input data(`data_5_10.txt`) is showing below:

```
5
10
0 1 30
0 2 32
0 2 25
0 3 38
0 4 32
1 3 47
1 2 20
2 4 1
2 4 30
3 4 43

```

consider it's a directed graph, and the shortest path of each vertex is  like this below:

```json
[[0,30,25,38,26],
[-1,0,20,47,21],
[-1,-1,0,-1,1],
[-1,-1,-1,0,43],
[-1,-1,-1,-1,0]]
```

the output data(`out_5_10.json`) is showing below:

```json
[[0,30,25,38,32],
[-1,0,20,47,50],
[-1,-1,0,-1,30],
[-1,-1,-1,0,43],
[-1,-1,-1,-1,0]],
```

