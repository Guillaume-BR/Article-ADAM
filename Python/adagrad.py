
#%%
import numpy as np
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
from numpy.random import rand
from numpy import asarray
from numpy import arange, meshgrid, sin, cos, exp, sqrt

#fonction objectif
def objective(x, y):
    return -sin(x**2/2 - y**2.0/4 ) * cos(2*x )

#%%
#3D-plot
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
#ligne de niveau
## define range for input
#bounds = np.asarray([[-2.0, 2.0], [-2.0, 2.0]])
## sample input range uniformly at 0.1 increments
#xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
#yaxis = arange(bounds[1,0], bounds[1,1], 0.1)
## create a mesh from the axis
#x, y = meshgrid(xaxis, yaxis)
## compute targets
#results = objective(x, y)
## create a filled contour plot with 50 levels and jet color scheme
#pyplot.contourf(x, y, results, levels=50, cmap='jet')
## show the plot
#pyplot.show()


# %%
def derivative(x, y):
    a1 = x**2/2 - y**2/4
    a2 = 2*x 
    b1 = cos(a1) * cos(a2)
    b2 = sin(a1) * sin(a2)
    return asarray([-x*b1 + 2*b2, -y/2*b1])

#%%
# gradient descent algorithm with adagrad
def adagrad(objective, derivative, bounds, n_iter, step_size):
	# track all solutions
	solutions = list()
	# generate an initial point
	solution = np.array([-1.5,1])
	# list of the sum square gradients for each variable
	sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for it in range(n_iter):
		# calculate gradient
		gradient = derivative(solution[0], solution[1])
		# update the sum of the squared partial derivatives
		for i in range(gradient.shape[0]):
			sq_grad_sums[i] += gradient[i]**2.0
		# build solution
		new_solution = list()
		for i in range(solution.shape[0]):
			# calculate the learning rate for this variable
			alpha = step_size / (1e-8 + sqrt(sq_grad_sums[i]))
			# calculate the new position in this variable
			value = solution[i] - alpha * gradient[i]
			new_solution.append(value)
		# store the new solution
		solution = asarray(new_solution)
		solutions.append(solution)
		# evaluate candidate point
		solution_eval = objective(solution[0], solution[1])
		# report progress
		print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
	return solutions

# seed the pseudo random number generator
np.random.seed(1)
# define range for input
bounds = asarray([[-2.0, 2.0], [-2.0, 2.0]])
# define the total iterations
n_iter = 50
# define the step size
step_size = 0.1
# perform the gradient descent search
solutions = adagrad(objective, derivative, bounds, n_iter, step_size)
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
solutions = asarray(solutions)
pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
# show the plot
pyplot.show()
# %%
