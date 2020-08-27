# CUDA parallel implementation of the Bullet Option Pricing partial differential equation method

For more details on the Bullet option please read the option_explained.pdf. This description explains only the main idea and the parallelism organization.

Let there be a REALITY with vectors v[n] --> v[n-1] --> v[n-2] --> ... --> v[1] that can be calculated backwards from the next one to the previous one. And let there be some special
moments indexed T[s[1]], T[s[2]], ... , T[s[k]] where there appear other realities. Within the same reality vectors can still be calculated backwards but for the transition moments e.g. T[s[3]]
one staying in a reality needs to know the results of v[T[s[3]] + 1] **IN ALL OTHER EXISTING REALITIES** to calculate v v[T[s[3]] + 1]
