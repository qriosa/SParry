// 简单并行 的 dijkstra
__global__ void dijkstra(int* V, int* E, int* W, int* n, int* vis, int* dist, int* predist){
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 每个thread有自己的编号 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
    __shared__ int quickBreak[1];
    
    int u = -1;

	for(int i = 0; i < (*n); i++){

		quickBreak[0] = 0;
		
		u = u0;
		while(u < *n){ 
			if(vis[u] == 0){ 
				vis[u] = 1;
				for(int j = V[u]; j < V[u + 1]; j++){ // 枚举u的终点，j是E和W数组的下标 E[j]是这条边的终点 W[j]是这条边的边权。	
					atomicMin(&predist[E[j]], dist[u] + W[j]);
				}
			}
			u += offset;
		}
		__syncthreads(); 

		u = u0;
		while(u < *n){
			if(predist[u] < dist[u]){ 
				dist[u] = predist[u];
				vis[u] = 0; 
				quickBreak[0] = 1;
			}
			u += offset;
		}
		__syncthreads(); 
		if(quickBreak[0] == 0)
			break;
	}	
}


/*下面这个是不使用多流的 使用默认流进行分块*/
// noStream 和下面的函数是一样的 就不再写了

/* 下面这个是 divide 也就是不使用多流的 使用默认流的*/
// 现在这个base是E中的起点 
__global__ void divide(int* V, int* E, int* W, int* n, int* flag, int* base, int* part, int* vis, int* dist, int* predist){
	
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 每个thread有自己的编号 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
	
	int u = -1;
	int l = -1;
	int r = -1;
	int localBase = base[0];
	int localPart = part[0];
	
	u = u0; // 此时的 u 不再是真的结点编号 来看是不是会超出 part 的范围, u + l 是 真的结点编号
	while(u < (*n)){ // 当前线程所代表的点在显存中

		if(V[u + 1] <= localBase){ // 自己右边
			u += offset;
			continue; // 这个结点的边不在合法范围内
		}
		else if(V[u] >= localBase + localPart){ // 自己的左边
			u += offset;
			continue; // 这个结点的边不在合法范围内
		}

		// 在上一轮更新过
		if(vis[u]){ 
			// 这个地方就不好判断了 因为分成的多块和多流之间的无先后顺序 故 vis 无法再使用
			//vis[u] -= 1; // 标记其松驰能力减一
			atomicSub(&vis[u], 1);

			// 对区间进行缩减
			l = localBase>V[u]?localBase:V[u];
			r = (localBase + localPart)<V[u + 1]?(localBase + localPart):V[u + 1];
			
			for(int j = l; j < r; j++){ // 枚举u的终点，j是E和W数组的下标 E[j]是这条边的终点 W[j]是这条边的边权。	
				atomicMin(&predist[E[j - localBase]], dist[u] + W[j - localBase]); // 注意原始的下标在现在的部分数组中是不对的 因此得映射一下
			}
		}
		
		u += offset;
	}
	__syncthreads(); 

	u = u0;
	while(u < (*n)){
		if(predist[u] < dist[u]){ 
			dist[u] = predist[u];
			vis[u] = (V[u + 1] + localPart - 1) / localPart - V[u] / localPart; // 重新计算其更新能力 
			flag[0] = 1;
		}
		u += offset;
	}
}
