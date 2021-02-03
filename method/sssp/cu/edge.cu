__global__ void edge(int* src, int* des, int* w, int* m, int* dist){ 
	
	const int e0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;  
	const int offset = blockDim.x * blockDim.y * blockDim.z; 
	int e = -1;

	__shared__ int quickBreak[1];
		
	while(1){
		e = e0;

		if(e == 0){
			quickBreak[0] = 0;
		} 
		__syncthreads();
		
		while(e < (*m)){
			
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