import numpy as np
import matplotlib.pyplot as plt


def show(x_sample, y, a=0, b=0, x_2=None, y_2=None):
    if b != 0:
        plt.plot(x_sample, a+b*x_sample, c='red')

    if x_2 is not None:
        plt.plot(x_2, y_2, c='red')
    plt.scatter(x_sample, y)
    plt.show()



x = np.linspace(2, 30, 30)
#y = 2*(x+15) + np.random.random_sample(x.shape)*5 + 2
x = x[:, np.newaxis]


#show(x, y)



def kernel(x_i,x_j):
    #return np.dot(x_j,x_i) # Linear Kernel, same as not using Kernels at all
    #return np.dot(x_j,x_i)**2 # Polynomial Kernel
    return np.exp(-0.5*np.linalg.norm(x_i-x_j))     # Gaussian Kernel


# let's set up some non linear data
y = np.sin(x[:, 0])+x[:, 0]
#show(x, y)
#y = x[:, 0] + 4*np.sin(x[:, 0]) + 4*np.random.rand(x.shape[0])

# We could just call the kernel function every time
# Instead we store the solutions in this matrix
# to save some computations

K = np.zeros((x.shape[0], x.shape[0]))
for i, row in enumerate(K):
    for j, col in enumerate(K.T):
        K[i, j] = kernel(x[i, :], x[j, :])


a = np.linalg.inv(K)
m = np.dot(y, a)

#print(m)


x_pred = np.arange(0, 35, 0.2)

y_pred = np.zeros(x_pred.shape[0])


for outer_i, x_p in enumerate(x_pred):
    k = np.zeros(x.shape[0])
    for i, row in enumerate(k):
        k[i] = kernel(x_p, x[i, :])
    y_pred[outer_i] = np.dot(m, k)

show(x[:, 0], y, x_2=x_pred, y_2=y_pred)

print("We are done")


print("I think this should be the ending, yours is lame")
