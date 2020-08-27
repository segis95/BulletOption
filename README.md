# CUDA parallel implementation of the Bullet Option Pricing partial differential equation method

For more details on the Bullet option please read the option_explained.pdf. This description explains only the main idea and the parallelism organization.

Let there be a REALITY with vectors v[n] --> v[n-1] --> v[n-2] --> ... --> v[1] that can be calculated backwards from the next one to the previous one. And let there be some special
moments indexed T[s[1]], T[s[2]], ... , T[s[k]] where there appear other realities. Within the same reality vectors can still be calculated backwards but for the transition moments e.g. T[s[3]]
one staying in a reality needs to know the results of v[T[s[3]] + 1] **IN ALL OTHER EXISTING REALITIES** to calculate  v[T[s[3]]] **IN HIS OWN REALITY**. So, in those moments **realities interact with other realities and exchange information with each other** and **in other moments realities evolve backwards independently**. 

![Scheme](/images/scheme_realities.jpg)

## Code organization
In terms of the above introduction, the main goal of the code was to provide a way for realities to evolve independently(**in a parallel way**) between moments where the transition happens and in those transition moments to synchronize realities and transmit results to their neighbours so that they were able to use them for the further backpropagation. In the code realities are identified by **blockIdx.x** and processes inside them  - by **threadIdx.x**. Processes are used for the parallel computation of vectors v[i]. And blocks of those processes with **shared memory** where they can write form realities. The **KEY FEATURE** of the code is that there is **EXACTLY ONE** CUDA kernel call **PDE_Solver<<M+1, N>>**. This means that  CUDA device **DOES NOT EXCHANGE DATA WITH CPU WHEN THE REALITIES ARE SYNCHRONIZED**. All intra-blocks(**REALITIES**) synchronization is done **INSIDE** CUDA device using **__syncthreads()** and shared variables with atomic increment **atomicAdd(&synchro_count[i], 1)**.


