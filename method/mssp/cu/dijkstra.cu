// 普通的 dijkstra 算法的并行 拥有未更新退出 一个 block 代表着第 i 个源点的计算结果
__global__ void dijkstra(int* V, int* E, int* W, int* n, int* srcNum, int* vis, int* dist, int* predist){
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 每个thread有自己的编号 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
	const int blockNum = (const int) gridDim.x * gridDim.y; // block 的数量
		
	// const int u0 = (const int)threadIdx.x;
	// const int offset = (const int)(blockDim.x);
	// const int blockNum = (const int)(gridDim.x); // block总数 也即是 一次最多解决多少和单源问题

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


__global__ void divide(int* V, int* E, int* W, int* n, int* sNum, int* flag, int* base, int* part, int* vis, int* dist, int* predist){
	
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 每个thread有自己的编号 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
	// const int blockNum = (const int) gridDim.x * gridDim.y; // block 的数量
	const int s0 = gridDim.x * blockIdx.y + blockIdx.x;
	
	// const int u0 = threadIdx.x; // 每个thread有自己的编号 
	// const int offset = blockDim.x; // 一个grid里面有多少的thread
	// const int s0 = blockIdx.x; 
	
	int u = -1;
	int sn = -1;
	int l = -1;
	int r = -1;
	int localBase = base[0];
	int localPart = part[0];

	if(s0 >= (*sNum)){
		return; // 超出了当前的源点数量
	}

	sn = s0 * (*n); // dist 中的偏移
	
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
			atomicSub(&vis[u + sn], 1);

			// 对区间进行缩减
			l = localBase>V[u]?localBase:V[u];
			r = (localBase + localPart)<V[u + 1]?(localBase + localPart):V[u + 1];
			
			for(int j = l; j < r; j++){ // 枚举u的终点，j是E和W数组的下标 E[j]是这条边的终点 W[j]是这条边的边权。	
				atomicMin(&predist[E[j - localBase] + sn], dist[u + sn] + W[j - localBase]); // 注意原始的下标在现在的部分数组中是不对的 因此得映射一下
			}
		}
		
		u += offset;
	}
	__syncthreads(); 

	u = u0;
	while(u < (*n)){
		if(predist[u + sn] < dist[u + sn]){ 
			dist[u + sn] = predist[u + sn];
			vis[u + sn] = (V[u + 1] + localPart - 1) / localPart - V[u] / localPart; // 重新计算其更新能力 
			flag[0] = 1; // 可以不同的block中的线程进行冲突赋值 反正有一个源发生了改变就应该是 1  
		}
		u += offset;
	}
}
