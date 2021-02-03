// dijkstra 算法的并行自全源加未更新快速退出 
__global__ void dijkstra(int* V, int* E, int* W, int* n, int* vis, int* dist, int* predist){
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; 
	const int offset = blockDim.x * blockDim.y * blockDim.z; // the number of threads in a block
	const int blockNum = (const int) gridDim.x * gridDim.y; // the number of block

	int u = -1;
	int sn = -1;
	int s = blockIdx.z * (gridDim.x *  gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;

	__shared__ int quickBreak[1];

	while(s < (*n)){ // source must valid

		sn = (s * (*n));

		for(int i = 0; i < (*n); i++){

			quickBreak[0] = 0; 
			
			u = u0;
			while(u < *n){ 
				if(vis[u + sn] == 0){ 
					vis[u + sn] = 1;
					for(int j = V[u]; j < V[u + 1]; j++){ // the end of j, j is the index of E and W, E[j] is the end of this edge, W[j] is the weight of this edge.	
						atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]); // s is source.
					}
				}
				u += offset;
			}
			
			__syncthreads(); 

			u = u0;
			while(u < (*n)){
				if(predist[u + sn] < dist[u + sn]){ 
					dist[u + sn] = predist[u + sn];
					vis[u + sn] = 0; 

					quickBreak[0] = 1;
				}
				u += offset;
			}

			__syncthreads();

			if(quickBreak[0] == 0){
				break;
			}
			__syncthreads(); 
		}
		s += blockNum; // next vertex
	}	
}


// base is the start index of E  
__global__ void divide(int* V, int* E, int* W, int* n, int* flag, int* base, int* part, int* vis, int* dist, int* predist){
	
	const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;  
	const int offset = blockDim.x * blockDim.y * blockDim.z; // the number of threads in a  block
	
	int u = -1;
	int l = -1;
	int r = -1;
	int localBase = base[0];
	int localPart = part[0];
	
	u = u0; // this turn, u is not the number of vertex, it's just a offset to tell out of range(part), and u+l is the true number of vertex.
	while(u < (*n)){ // the vertex represented by the current thread  is in video memory.

		if(V[u + 1] <= localBase){ // self right
			u += offset;
			continue; // the range of vertex's edge is illegal. 
		}
		else if(V[u] >= localBase + localPart){ // self left
			u += offset;
			continue; // the range of vertex's edge is illegal. 
		}

		// dist is updated
		if(vis[u]){ 
			// different part is not ordered, so can not use vis to tell.
			//vis[u] -= 1; // 
			atomicSub(&vis[u], 1); // set the ability of running sub one.

			// Shrink the range
			l = localBase>V[u]?localBase:V[u];
			r = (localBase + localPart)<V[u + 1]?(localBase + localPart):V[u + 1];
			
			for(int j = l; j < r; j++){ // get the end of u, j is the index of E and W, E[j] is the end of this edge, W[j] is the weight of this edge.	
				atomicMin(&predist[E[j - localBase]], dist[u] + W[j - localBase]); // the index is not true, need to add offset.
			}
		}
		
		u += offset;
	}
	__syncthreads(); 

	u = u0;
	while(u < (*n)){
		if(predist[u] < dist[u]){ 
			dist[u] = predist[u];
			vis[u] = (V[u + 1] + localPart - 1) / localPart - V[u] / localPart; // recalc the ability of running.
			flag[0] = 1;
		}
		u += offset;
	}
}
