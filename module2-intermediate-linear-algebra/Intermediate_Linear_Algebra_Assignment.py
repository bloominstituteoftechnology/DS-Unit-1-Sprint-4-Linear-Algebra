#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'module2-intermediate-linear-algebra'))
	# print(os.getcwd())
except:
	pass

#%%
import matplotlib.pyplot as pyplot
pyplot.rcParams['figure.facecolor'] = '#002B36'

#%% [markdown]
# # Statistics
#%% [markdown]
# ## 1.1 Sales for the past week was the following amounts: [3505, 2400, 3027, 2798, 3700, 3250, 2689]. Without using library functions, what is the mean, variance, and standard deviation of of sales from last week? (for extra bonus points, write your own function that can calculate these two values for any sized list)

#%%
sales = [3505, 2400, 3027, 2798, 3700, 3250, 2689]

mean = sum(sales)/len(sales)
variance = sum([(u - mean)**2 for u in sales])/(len(sales)-1)
stddev = variance**0.5

print(f'mean: {mean}')
print(f'variance: {variance}')
print(f'standard deviation: {stddev}')


#%% [markdown]
# ## 1.2 Find the covariance between last week's sales numbers and the number of customers that entered the store last week: [127, 80, 105, 92, 120, 115, 93] (you may use librray functions for calculating the covariance since we didn't specifically talk about its formula)

#%%
import numpy

customers = [127, 80, 105, 92, 120, 115, 93]

custmean = sum(customers)/len(customers)
covariance = sum([(u-mean)*(v-custmean) for u, v in zip(sales, customers)])/(len(sales)-1)

assert covariance == numpy.cov([sales, customers])[1][0]
print(f'covariance: {covariance}')

#%% [markdown]
# ## 1.3 Find the standard deviation of customers who entered the store last week. Then, use the standard deviations of both sales and customers to standardize the covariance to find the correlation coefficient that summarizes the relationship between sales and customers. (You may use library functions to check your work.)

#%%

customers_stddev = numpy.std(customers, ddof=1)
correlation = covariance/(customers_stddev*stddev)

assert correlation == numpy.corrcoef(sales, customers)[1][0]
print(f'coefficient of correlation: {correlation}')


#%% [markdown]
# ## 1.4 Use pandas to import a cleaned version of the titanic dataset from the following link: [Titanic Dataset](https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_clean.csv)
# 
# ## Calculate the variance-covariance matrix and correlation matrix for the titanic dataset's numeric columns. (you can encode some of the categorical variables and include them as a stretch goal if you finish early)

#%%
import pandas

df = pandas.read_csv('https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_clean.csv')

df.head()

#%%
df.cov()

#%%
df.corr()

#%% [markdown]
# # Orthogonality
#%% [markdown]
# ## 2.1 Plot two vectors that are orthogonal to each other. What is a synonym for orthogonal?

#%%
import matplotlib.pyplot as pyplot

vA = [-1,1]
vB = [0.5,.5]

pyplot.arrow(0,0,*vA)
pyplot.arrow(0,0,*vB)
pyplot.xlim(-2,2)
pyplot.ylim(-2,2)
pyplot.gcf().axes[0].set_aspect('equal')
pyplot.show()


print(f'vA dot vB: {numpy.dot(vA, vB)}')



#%% [markdown]
# ## 2.2 Are the following vectors orthogonal? Why or why not?
# 
# \begin{align}
# a = \begin{bmatrix} -5 \\ 3 \\ 7 \end{bmatrix}
# \qquad
# b = \begin{bmatrix} 6 \\ -8 \\ 2 \end{bmatrix}
# \end{align}

#%%
vA = [-5,3,7]
vB = [6,-8,2]

print(f'a and b orthogonal: {numpy.dot(vA, vB)==0} (a dot b = {numpy.dot(vA, vB)})')


#%% [markdown]
# ## 2.3 Compute the following values: What do these quantities have in common?
# 
# ## What is $||c||^2$? 
# 
# ## What is $c \cdot c$? 
# 
# ## What is $c^{T}c$?
# 
# \begin{align}
# c = \begin{bmatrix} 2 & -15 & 6 & 20 \end{bmatrix}
# \end{align}

#%%
mC = numpy.array([2, -15, 6, 20])
print(f'||c||^2 = {numpy.linalg.norm(mC)**2}')
print(f'c dot c = {numpy.dot(mC,mC)}')
print(f'c^T c = {numpy.matmul(mC.T, mC)}')
# Note that all of these are just the sum of the squares of the elements

#%% [markdown]
# # Unit Vectors
#%% [markdown]
# ## 3.1 Using Latex, write the following vectors as a linear combination of scalars and unit vectors:
# 
# \begin{align}
# d = \begin{bmatrix} 7 \\ 12 \end{bmatrix}
# \qquad
# e = \begin{bmatrix} 2 \\ 11 \\ -8  \end{bmatrix}
# \end{align}
#%% [markdown]
# \begin{align}
# d = 7 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 12 \begin{bmatrix} 0 \\ 1 \end{bmatrix}
# \end{align}
#%% [markdown]
# \begin{align}
# e = 2 \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} + 11 \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} - 8 \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
# \end{align}

#%% [markdown]
# ## 3.2 Turn vector $f$ into a unit vector:
# 
# \begin{align}
# f = \begin{bmatrix} 4 & 12 & 11 & 9 & 2 \end{bmatrix}
# \end{align}

#%%
vF = numpy.array([4, 12, 11, 9, 2])
mag = numpy.linalg.norm(vF)
unit_vF = vF/mag

print(f'unit vector of f: {unit_vF}')

#%% [markdown]
# # Linear Independence / Dependence 
#%% [markdown]
# ## 4.1 Plot two vectors that are linearly dependent and two vectors that are linearly independent (bonus points if done in $\mathbb{R}^3$).

#%%

dep_1 = [.9, .6]
dep_2 = [-.3, -.2]
ind_1 = [.1, .7]
pyplot.arrow(0,0,*dep_1,color='g')
pyplot.arrow(0,0,*dep_2,color='b')
pyplot.arrow(0,0,*ind_1,color='r')
pyplot.ylim(-1,1)
pyplot.xlim(-1,1)
pyplot.show()

#%% [markdown]
# # Span
#%% [markdown]
# ## 5.1 What is the span of the following vectors?
# 
# \begin{align}
# g = \begin{bmatrix} 1 & 2 \end{bmatrix}
# \qquad
# h = \begin{bmatrix} 4 & 8 \end{bmatrix}
# \end{align}

#%%
vG = [1,2]
vH = [4,8]
# Should be 1, since they're colinear
print(f'span: R{numpy.linalg.matrix_rank([vG, vH])}')

#%% [markdown]
# ## 5.2 What is the span of $\{l, m, n\}$?
# 
# \begin{align}
# l = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}
# \qquad
# m = \begin{bmatrix} -1 & 0 & 7 \end{bmatrix}
# \qquad
# n = \begin{bmatrix} 4 & 8  & 2\end{bmatrix}
# \end{align}

#%%
vL = [1, 2, 3]
vM = [-1, 0, 7]
vN = [4, 8, 2]

print(f'span: R{numpy.linalg.matrix_rank([vL, vM, vN])}')


#%% [markdown]
# # Basis
#%% [markdown]
# ## 6.1 Graph two vectors that form a basis for $\mathbb{R}^2$
# 
# 

#%%
import math

vA = [math.pi, -1.1**11]
vB = [-math.sqrt(7), math.e]

pyplot.arrow(0,0,*vA)
pyplot.arrow(0,0,*vB)
pyplot.xlim(-5,5)
pyplot.ylim(-5,5)
pyplot.show()
print(f'span: R{numpy.linalg.matrix_rank([vA, vB])}')

#%% [markdown]
# ## 6.2 What does it mean to form a basis?
#%% [markdown]
# The entire vector space can be represented as a linear combination of scalars and the basis vectors
#%% [markdown]
# # Rank
#%% [markdown]
# ## 7.1 What is the Rank of P?
# 
# \begin{align}
# P = \begin{bmatrix} 
# 1 & 2 & 3 \\
#  -1 & 0 & 7 \\
# 4 & 8  & 2
# \end{bmatrix}
# \end{align}

#%%

mP = numpy.array([	[ 1, 2, 3],
					[-1, 0, 7],
					[ 4, 8, 2]])

print(f'span: R{numpy.linalg.matrix_rank(mP)}')

#%% [markdown]
# ## 7.2 What does the rank of a matrix tell us?
#%% [markdown]
# Dimensionality of the vector space
#%% [markdown]
# # Linear Projections
# 
# ## 8.1 Line $L$ is formed by all of the vectors that can be created by scaling vector $v$ 
# \begin{align}
# v = \begin{bmatrix} 1 & 3 \end{bmatrix}
# \end{align}
# 
# \begin{align}
# w = \begin{bmatrix} -1 & 2 \end{bmatrix}
# \end{align}
# 
# ## find $proj_{L}(w)$
# 
# ## graph your projected vector to check your work (make sure your axis are square/even)

#%%
vV = numpy.array([1,3])
vW = numpy.array([-1,2])

v_projected = numpy.dot(vV,vW)/numpy.dot(vV,vV)*vV
v_orth = v_projected-vW

pyplot.arrow(0,0,*vV,color='g')
pyplot.arrow(0,0,*vW,color='y')
pyplot.arrow(0,0,*v_projected)
pyplot.arrow(*vW,*v_orth,color='r',linestyle=':')
pyplot.xlim(-1.5,2.5)
pyplot.ylim(-0.5,3.5)
pyplot.gcf().axes[0].set_aspect('equal')
pyplot.show()


#%% [markdown]
# # Stretch Goal
# 
# ## For vectors that begin at the origin, the coordinates of where the vector ends can be interpreted as regular data points. (See 3Blue1Brown videos about Spans, Basis, etc.)
# 
# ## Write a function that can calculate the linear projection of each point (x,y) (vector) onto the line y=x. run the function and plot the original points in blue and the new projected points on the line y=x in red. 
# 
# ## For extra points plot the orthogonal vectors as a dashed line from the original blue points to the projected red points.

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Creating a dataframe for you to work with -Feel free to not use the dataframe if you don't want to.
x_values = [1, 4, 7, 3, 9, 4, 5 ]
y_values = [4, 2, 5, 0, 8, 2, 8]

data = {"x": x_values, "y": y_values}

df = pd.DataFrame(data)

df.head()

plt.scatter(df.x, df.y)
plt.show()


#%%



