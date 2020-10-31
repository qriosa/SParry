__global__ void delta_stepping(int* V, int* E, int* W, int* n, int* delta, int* dist, int* predist, int* B, int* hadin){
    const int u0 = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x ; // 每个thread有自己的编号 
    const int s0 = gridDim.x * blockIdx.y + blockIdx.x;
    const int offset = blockDim.x * blockDim.y * blockDim.z; // 一个 block 里面有多少的thread
    const int blockNum = (const int) gridDim.x * gridDim.y; // block 的数量

	int u = -1;
	int s = -1;
	int sn = -1;
	int s0n = -1;
	int ln = (*n);

	__shared__ int id[1]; // 整个桶内共享的桶的编号
	__shared__ int maxBucketId[1]; // 维护的所有桶标签中最大的哪个
	__shared__ int bucketNowIsNull[1]; // 判断当前桶是否空了

	// 可以写在while/if等条件里的必须是私有变量 否则（global/shared）就应该通过同步来确保没有错误
	s = s0;
	while(s < ln){ // 源点必须处于有效范围中	

		sn = s * ln; // 在 dist 中的偏移
		s0n = s0 * ln; // 在桶中的偏移

		// 初始化一些值
		u = u0;
		while(u < ln){
			B[s0n+ u] = -1; 
			hadin[s0n + u] = 0; // 记录某个点在当前桶的更新中曾在过 用于搞定重边
			u += offset;
		}

		__syncthreads(); // 避免下面的被复写 即后的初始化将源点初始化了 

		id[0] = 0; // 初始化在 0 号桶
		B[s0n + s] = 0; // 将源点放入 0 号桶中

		maxBucketId[0] = 0; // 初始化认为就是空了
		bucketNowIsNull[0] = 1; // 初始化认为有点 

		__syncthreads();
		

		// 不可以通过放入的时候来判断 因为有可能某个桶是很久很久前放入了点的 不是直接的上一轮放的
		// 但是可以通过所有放的桶中最大的哪个是谁来判断
		while(1){ // 当前想要计算的桶必须小于等于最大桶 

			// 当前桶中有哪些点呢  遍历一次来确定
			while(1){ // 有可能有点会二次进入此桶
				// while(id[0] <= maxid) 这样是不可以的 会导致跑得快的线程就给设置为0了，然后其他的就进不来了

				bucketNowIsNull[0] = 0; // 标记当前桶空的 

				u = u0;
				while(u < ln){
					if(B[s0n + u] == id[0]){ // 当前u代表的这个点在当前这个 id 的桶中
						hadin[s0n + u] = 1; // 标记当前这个点曾经在过
						B[s0n + u] = -1; // 标记当前点已经出桶了 
							
						for(int j = V[u]; j < V[u + 1]; j++){ // 枚举邻接边
							if(W[j] <= (*delta)){ // 轻边
								atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]);
							}
						}
						
					}
					u += offset;
				}
				
				__syncthreads(); // 此同步也是必要的，必须更新完了才检测是否更新dist

				// 检测 predist 判断是否有点在本轮被更新了 同时检测是否有新点落入当前桶中
				u = u0;
				while(u < ln){
					if(predist[u + sn] < dist[u + sn]){ // 如果当前结点 predist 更小了，说明其可以被更新了

						dist[u + sn] = predist[u + sn]; // 更新其 dist
						B[s0n + u] = dist[u + sn] / (*delta); // 更新当前点的桶编号

						atomicMax(maxBucketId, B[s0n + u]); // 更新最大桶的编号
	
						if(B[s0n + u] == id[0]){ // 如果更新后还是落入当前桶中 那么就说明当前桶中是还有有点的 // 当前桶中还有点 标记当前桶不是空的
							bucketNowIsNull[0] = 1;
						}
					}
					u += offset;
				}
				 
				__syncthreads(); // 同步以确保这个bucketNowIsNull的更新能够到达每个线程
				
				if(bucketNowIsNull[0] == 0){
					break;
				}
			}
			
			__syncthreads(); // 这里是有必要的 轻边完事了就应该重边了 避免先到的错过了后面的更新的

			u = u0;
			while(u < ln){
				
				if(hadin[s0n + u]){ // 当前结点在本轮中进入过当前桶中 进入过当前桶其距离是一定发生过改变的  所以不要vis来判断
					hadin[s0n + u] = 0; // 标记为没有进入过了 下一个桶可以继续用
					
					for(int j = V[u]; j < V[u + 1]; j++){
						if(W[j] > (*delta)){ // 重边
							atomicMin(&predist[E[j] + sn], dist[u + sn] + W[j]);
						}
					}
				}
				u += offset;
			}

			__syncthreads(); //全部搞完了才应该进行更新的检测

			// 检测 predist 判断是否有点在本轮被更新了 同时检测是否有新点落入当前桶中
			u = u0;
			while(u < ln){
				if(predist[u + sn] < dist[u + sn]){ // 如果当前结点 predist 更小了，说明其可以被更新了

					dist[u + sn] = predist[u + sn]; // 更新其 dist
					B[s0n + u] = dist[u + sn] / (*delta); // 更新当前点的桶编号
	
					atomicMax(maxBucketId, B[s0n + u]); // 更新最大桶的编号
				}
				u += offset;
			}

			// __syncthreads();  // 这里同步不需要 比较后面有同步 即使出现了更新maxBucketId,问题也没有
			if(u0 == 0){
				atomicAdd_block(id, 1); // 下一个桶
				bucketNowIsNull[0] = 1; // 假设下一个桶中有点
			}

			__syncthreads(); // 确保每个线程都能得到正确的 id, maxBucketId 避免0号先到就break了但是其他的才又更新了 maxBucketId

			if(id[0] > maxBucketId[0])
				break;
					
		}
		__syncthreads();
		s += blockNum; // 下一个源点
	}
}
