// 普通的 dijkstra 算法的并行 拥有未更新退出 一个 block 代表着第 i 个源点的计算结果
__global__ void dijkstra(int* V, int* E, int* W, int* n, int* srcNum, int* vis, int* dist, int* predist){
	
	const int u0 = (const int)threadIdx.x;
	const int offset = (const int)(blockDim.x);
	const int blockNum = (const int)(gridDim.x); // block总数 也即是 一次最多解决多少和单源问题

	int u = -1;
	int sn = -1;
	int sIndex = blockIdx.x; // s是源点的问题

	__shared__ int quickBreak[1];

	while(sIndex < (*srcNum)){ // 源点也必须有效才行

		sn = (sIndex * (*n));

		for(int i = 0; i < (*n); i++){

			quickBreak[0] = 0; //应该是不需要原子操作，原子操作应该是更慢些。
			
			u = u0;
			while(u < *n){ 
				if(vis[u + sn] == 0){ 
					vis[u + sn] = 1;
					for(int j = V[u]; j < V[u + 1]; j++){ // 枚举u的终点，j是E和W数组的下标 E[j]是这条边的终点 W[j]是这条边的边权。	
						atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]); // s 为源点
					}
				}
				u += offset;
			}
			__syncthreads(); 

			u = u0;
			while(u < (*n)){
				if(predist[u + sn] < dist[u + sn]){ 
					dist[u + sn] = predist[u + sn];
					vis[u + sn] = 0; //后面再考虑把这个vis独立为自己的局部block变量 dist呢？

					quickBreak[0] = 1;
				}
				u += offset;
			}

			__syncthreads(); 
			if(quickBreak[0] == 0){
				break;
			}
		}
		sIndex += blockNum; // 调向下一个源点 
	}	
}
