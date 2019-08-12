import numpy as np
from sklearn.decomposition import TruncatedSVD as TruncSVD
import time
dim1 = 10000
dim2 = 2000
dim3 = dim2 - 1
X = np.random.randn(dim1, dim2)

B = np.random.randn(dim2)
# B_trun = np.random.randn(dim2)[:-1]
# svd = TruncSVD(dim3)
# svd.fit(X)
# X_svd = svd.transform(X)

a, b, c = np.linalg.svd(X)
x_trun = a * b @ c

print(X_svd.shape, X.shape)
#assert np.allclose(X_svd, X)

exit()
t1 = time.time()

X @ B
t2 = time.time()

X_svd @ B_trun

t3 = time.time()

print("TIME1", t2-t1)
print("TIME2", t3-t2)
