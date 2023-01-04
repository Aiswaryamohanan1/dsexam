

import numpy as np
mat1=np.array([[1,2,3],[4,5,6],[7,8,9]])
mat2=np.array([[7,8,9],[4,5,6],[1,2,3]])
print('Matrix Addition')
print(np.add(mat1,mat2))
print('Matrix Subtraction')
print(np.subtract(mat1,mat2))
print("Matrix Multiplication")
print(np.multiply(mat1,mat2))
print("Matrix Division")
print(np.divide(mat1,mat2))

from scipy.linalg import svd

a=[[1,2,3],[4,5,6],[7,8,9]]
print(a)
u, s, VT = svd(a)
print("Decomposed matrix \n")
print(u)
print("Inverse matrix \n")
print(s)
print("Transpose matrix ")
print(VT)
