import numpy as np
a=np.zeros((5,3))
a[0,0]=1
a[0,1]=2
a[1,0]=3
a[1,1]=4
b=np.mean(a,1)
print(b)