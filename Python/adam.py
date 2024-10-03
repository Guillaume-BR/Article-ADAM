#%%
# 3d plot of the test function
import numpy as np
import math
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
from numpy import random
from numpy import asarray
from numpy import arange, meshgrid, sin, cos, exp, sqrt

#%%
#fonction objectif
def objective(x, y):
    return -sin(x**2/2 - y**2.0/4 + 3) * cos(2*x + 1 - exp(y))

#%%
# define range for input
r_min, r_max = -2.0, 2.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.add_subplot(111, projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
# show the plot
pyplot.show()

#%%
# define range for input
bounds = np.asarray([[-2.0, 2.0], [-2.0, 2.0]])
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# show the plot
pyplot.show()


# %%
def derivative(x, y):
    a1 = x**2/2 - y**2/4 + 3
    a2 = 2*x + 1 - exp(y)
    b1 = cos(a1) * cos(a2)
    b2 = sin(a1) * sin(a2)
    return asarray([-x*b1 + 2*b2, -y/2*b1 + exp(y)*b2])

# gradient descent algorithm with adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    solutions = []
    # generate an initial point
    x = np.array([-1.5,0])
    score = objective(x[0], x[1])
    # initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
    #run iterations of gradient descent
    for t in range(n_iter):
    # calculate gradient g(t)
       g = derivative(x[0], x[1])
       # build a solution one variable at a time
       for i in range(x.shape[0]):
           # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
           m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
           # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
           v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
           # mhat(t) = m(t) / (1 - beta1(t))
           mhat = m[i] / (1.0 - beta1**(t+1))
           # vhat(t) = v(t) / (1 - beta2(t))
           vhat = v[i] / (1.0 - beta2**(t+1))
           # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
           x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
       # evaluate candidate point
       score = objective(x[0], x[1])
       # keep track of solutions
       solutions.append(x.copy())
       # report progress
       print('>%d f(%s) = %.5f' % (t, x, score))
    return solutions

#%%
# seed the pseudo random number generator
np.random.seed(1)
# define range for input
bounds = asarray([[-2.0, 2.0], [-2.0, 2.0]])
# define the total iterations
n_iter = 100
# steps size
alpha = 0.02
# factor for average gradient
beta1 = 0.8
# factor for average squared gradient
beta2 = 0.999
# perform the gradient descent search with adam
solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
# sample input range uniformly at 0.1 increments
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a filled contour plot with 50 levels and jet color scheme
pyplot.contourf(x, y, results, levels=50, cmap='jet')
# plot the sample as black circles
solutions = np.asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
## show the plot
#pyplot.show()

