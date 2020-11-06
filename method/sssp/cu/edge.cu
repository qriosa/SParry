// 普通 edge 不记录路径
__global__ void edge(int* src, int* des, int* w, int* m, int* dist){ // 每个线程作为一条边 判断两个端点是否发生了改变 w可以不用每次都传吧
	
	const int e0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; // 每个thread有自己的编号 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
	int e = -1;

	__shared__ int quickBreak[1]; // block 内部的退出标识
		
	while(1){
		e = e0;

		if(e == 0){
			quickBreak[0] = 0;
		} 
		__syncthreads();
		
		while(e < (*m)){
			
			// old = atomicMin(&dist[src[e]], dist[des[e]] + w[e]);

			// if(dist[src[e]] < old)
			// 	flag[0] = 1;
			
			if(dist[des[e]] > dist[src[e]] + w[e]){
				atomicMin(&dist[des[e]], dist[src[e]] + w[e]);

				if(dist[des[e]] = dist[src[e]] + w[e]){
					quickBreak[0] = 1;
				}
			}
			
			e += offset;
		}
		
		__syncthreads();

		if(quickBreak[0] == 0){
			break;
		}
	}
}