import numpy as np
arr = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[1,2,3],[4,5,6],[7,8,9],[10,11,12]
       ,[1,2,3],[4,5,6],[7,8,9],[10,11,12],[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
narr = np.array(arr)
#print(narr)

maximums, minimums, avgs = \
    narr.max(axis=0), \
    narr.min(axis=0), \
    narr.sum(axis=0) / narr.shape[0]

#print(maximums, minimums, avgs)

x = narr[:, :-1]
y = narr[:,-1:]
# print(x)
# print(y)

w = [0.1, 0.2]
w = np.array(w).reshape([2, 1])
# print(w)
# print(x[0])
t = np.dot(x[0], w)
# print(t)

w5 = np.arange(-10.0, 10.0, 1.0)
w9 = np.arange(-10.0, 10.0, 1.0)
# print(w5, w9)
losses  = np.zeros([len(w5),len(w9)])
# print(losses)
# for i in range(len(w5)):
#     for j in range(len(w9)):
#         net.w[5] = w5[i]
#         net.w[9] = w9[i]

mini_batches = [arr[k:k+3] for k in range(0, len(arr), 3)]
print(type(mini_batches))
print(mini_batches)
print(np.shape(mini_batches))
print(np.array(arr).shape)
