// 普通 edge 不记录路径
__global__ void edge(int* src, int* des, int* w, int* m, int* dist, int* flag){ // 每个线程作为一条边 判断两个端点是否发生了改变 w可以不用每次都传吧
	
	const int e0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 每个thread有自己的编号 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
	int e = -1;
	int old = -1;
		
	while(1){
		e = e0;

		flag[0] = 0;
		
		__syncthreads();
		
		while(e < (*m)){ // 有向边敏感型
			old = atomicMin(&dist[des[e]], dist[src[e]] + w[e]);
			if(dist[des[e]] < old){
				flag[0] = 1;
			}
			e += offset;
		}
		
		__syncthreads();

		if(flag[0] == 0){
			break;
		}
	}
}