#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'module1-linear-algebra-review'))
	# print(os.getcwd())
except:
	pass
#%% [markdown]
# # Part 1 - Scalars and Vectors
# 
# For the questions below it is not sufficient to simply provide answer to the questions, but you must solve the problems and show your work using python (the NumPy library will help a lot!) Translate the vectors and matrices into their appropriate python  representations and use numpy or functions that you write yourself to demonstrate the result or property. 
#%% [markdown]
# ## 1.1 Create a two-dimensional vector and plot it on a graph

#%%
import numpy
import matplotlib.pyplot as pyplot

a = numpy.array([2,2])

pyplot.arrow(0,0, *a, head_width=0.2, head_length=0.2, overhang=0.5, length_includes_head=True)
pyplot.grid()
pyplot.ylim(0,4)
pyplot.xlim(0,4)

pyplot.show()


#%% [markdown]
# ## 1.2 Create a three-dimensional vecor and plot it on a graph

#%%
from mpl_toolkits.mplot3d import Axes3D

vectors = numpy.array([[0, 0, 0, .2, .4, .8]])

X, Y, Z, U, V, W = zip(*vectors)
figure = pyplot.figure()
axis = figure.add_subplot(111, projection='3d')
axis.quiver(X, Y, Z, U, V, W, length=1)
axis.set_xlim([0, 1])
axis.set_ylim([0, 1])
axis.set_zlim([0, 1])
axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')

pyplot.show()

#%% [markdown]
# ## 1.3 Scale the vectors you created in 1.1 by $5$, $\pi$, and $-e$ and plot all four vectors (original + 3 scaled vectors) on a graph. What do you notice about these vectors? 

#%%
from math import e, pi
print(e)
print(pi)

for v in [[2,2],[3,7],[7,1],[3,1]]:
	for s in [1,5,pi,e]:
		v_scaled = s * numpy.array(v)
		pyplot.arrow(0,0, *v_scaled, head_width=1, head_length=1, overhang=0.5, length_includes_head=True)
pyplot.grid()
pyplot.ylim(0,40)
pyplot.xlim(0,40)

pyplot.show()

#%%


#%% [markdown]
# ## 1.4 Graph vectors $\vec{a}$ and $\vec{b}$ and plot them on a graph
# 
# \begin{align}
# \vec{a} = \begin{bmatrix} 5 \\ 7 \end{bmatrix}
# \qquad
# \vec{b} = \begin{bmatrix} 3 \\4 \end{bmatrix}
# \end{align}

#%%
a = numpy.array([5,7])
b = numpy.array([3,4])
sub = a-b

a_arrow = pyplot.arrow(0,0, *a, head_width=0.4, head_length=0.4, overhang=0.5, length_includes_head=True, color='g', label='a')
b_arrow = pyplot.arrow(0,0, *b, head_width=0.4, head_length=0.4, overhang=0.5, length_includes_head=True, color='b', label='b')
sub_arrow = pyplot.arrow(*b, *sub, head_width=0.4, head_length=0.4, overhang=0.5, length_includes_head=True, color='r', label='a - b')
pyplot.grid()
pyplot.ylim(0,8)
pyplot.xlim(0,8)

pyplot.legend([a_arrow, b_arrow, sub_arrow], ['a', 'b', 'a - b'])

pyplot.show()


#%% [markdown]
# ## 1.5 find $\vec{a} - \vec{b}$ and plot the result on the same graph as $\vec{a}$ and $\vec{b}$. Is there a relationship between vectors $\vec{a} \thinspace, \vec{b} \thinspace \text{and} \thinspace \vec{a-b}$

#%%


#%% [markdown]
# ## 1.6 Find $c \cdot d$
# 
# \begin{align}
# \vec{c} = \begin{bmatrix}7 & 22 & 4 & 16\end{bmatrix}
# \qquad
# \vec{d} = \begin{bmatrix}12 & 6 & 2 & 9\end{bmatrix}
# \end{align}
# 

#%%
c = numpy.array([7,22,4,16])
d = numpy.array([12,6,2,9])

print(sum(c*d))


#%% [markdown]
# ##  1.7 Find $e \times f$
# 
# \begin{align}
# \vec{e} = \begin{bmatrix} 5 \\ 7 \\ 2 \end{bmatrix}
# \qquad
# \vec{f} = \begin{bmatrix} 3 \\4 \\ 6 \end{bmatrix}
# \end{align}

#%%

e = [5,7,2]
f = [3,4,6]
print(numpy.cross(e,f))


#%% [markdown]
# ## 1.8 Find $||g||$ and then find $||h||$. Which is longer?
# 
# \begin{align}
# \vec{g} = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 8 \end{bmatrix}
# \qquad
# \vec{h} = \begin{bmatrix} 3 \\3 \\ 3 \\ 3 \end{bmatrix}
# \end{align}

#%%
g = numpy.array([1,1,1,8])
h = numpy.array([3,3,3,3])

mag_g = sum(g*g)**0.5
mag_h = sum(h*h)**0.5

print(f'||g|| = {mag_g}')
print(f'||h|| = {mag_h}')

#%% [markdown]
# # Part 2 - Matrices
#%% [markdown]
# ## 2.1 What are the dimensions of the following matrices? Which of the following can be multiplied together? See if you can find all of the different legal combinations.
# \begin{align}
# A = \begin{bmatrix}
# 1 & 2 \\
# 3 & 4 \\
# 5 & 6
# \end{bmatrix}
# \qquad
# B = \begin{bmatrix}
# 2 & 4 & 6 \\
# \end{bmatrix}
# \qquad
# C = \begin{bmatrix}
# 9 & 6 & 3 \\
# 4 & 7 & 11
# \end{bmatrix}
# \qquad
# D = \begin{bmatrix}
# 1 & 0 & 0 \\
# 0 & 1 & 0 \\
# 0 & 0 & 1
# \end{bmatrix}
# \qquad
# E = \begin{bmatrix}
# 1 & 3 \\
# 5 & 7
# \end{bmatrix}
# \end{align}

#%%
mA = numpy.array([	[1,2],
					[3,4],
					[5,6]])
mB = numpy.array([	[2,4,6]])
mC = numpy.array([	[9,6,3],
					[4,7,11]])
mD = numpy.array([	[1,0,0],
					[0,1,0],
					[0,0,1]])
mE = numpy.array([	[1,3],
					[5,7]])

matrices = {'A': mA, 'B': mB, 'C': mC, 'D': mD, 'E': mE}
for name_1 in matrices:
	m1 = matrices[name_1]
	for name_2 in matrices:
		m2 = matrices[name_2]
		print(f'{name_1} shape: {m1.shape}')
		print(f'{name_2} shape: {m2.shape}')

		# Two different ways of checking if we can multiply
		# Because why not
		if m1.shape[1] == m2.shape[0]:
			print(f'Result shape: {(m1.shape[0],m2.shape[1])}')

		try:
			print(f'{name_1}{name_2}:\n{numpy.matmul(m1,m2)}')
		except ValueError as e:
			print(f'Can\'t multiply {name_1} by {name_2}:')
			print(e)

#%% [markdown]
# ## 2.2 Find the following products: CD, AE, and BA. What are the dimensions of the resulting matrices? How does that relate to the dimensions of their factor matrices?

#%%


#%% [markdown]
# ## 2.3  Find $F^{T}$. How are the numbers along the main diagonal (top left to bottom right) of the original matrix and its transpose related? What are the dimensions of $F$? What are the dimensions of $F^{T}$?
# 
# \begin{align}
# F = 
# \begin{bmatrix}
# 20 & 19 & 18 & 17 \\
# 16 & 15 & 14 & 13 \\
# 12 & 11 & 10 & 9 \\
# 8 & 7 & 6 & 5 \\
# 4 & 3 & 2 & 1
# \end{bmatrix}
# \end{align}

#%%
mF = numpy.array([	[20,19,18,17],
					[16,15,14,13],
					[12,11,10, 9],
					[ 8, 7, 6, 5],
					[ 4, 3, 2, 1]])

print(f'F:\n{mF}')
print(f'F^T:\n{mF.T}')
print(f'Shape of F: {mF.shape}')
print(f'Shape of F^T: {mF.T.shape}')

#%% [markdown]
# # Part 3 - Square Matrices
#%% [markdown]
# ## 3.1 Find $IG$ (be sure to show your work) 😃
# 
# \begin{align}
# G= 
# \begin{bmatrix}
# 12 & 11 \\
# 7 & 10 
# \end{bmatrix}
# \end{align}

#%% [markdown]
# \begin{align}
# IG= 
# \begin{bmatrix}
# 12 & 11 \\
# 7 & 10 
# \end{bmatrix}
# \end{align}


#%% [markdown]
# ## 3.2 Find $|H|$ and then find $|J|$.
# 
# \begin{align}
# H= 
# \begin{bmatrix}
# 12 & 11 \\
# 7 & 10 
# \end{bmatrix}
# \qquad
# J= 
# \begin{bmatrix}
# 0 & 1 & 2 \\
# 7 & 10 & 4 \\
# 3 & 2 & 0
# \end{bmatrix}
# \end{align}
# 

#%%
import math

mH = numpy.array([	[12,11],
					[ 7,10]])
mJ = numpy.array([	[ 0, 1, 2],
					[ 7,10, 4],
					[ 3, 2, 0]])

print(f'|H| = {math.floor(numpy.linalg.det(mH)+0.5)}')
print(f'|J| = {math.floor(numpy.linalg.det(mJ)+0.5)}')


#%% [markdown]
# ## 3.3 Find $H^{-1}$ and then find $J^{-1}$

#%%

print(f'H^-1:\n{numpy.linalg.inv(mH)}')
print(f'J^-1:\n{numpy.linalg.inv(mJ)}')


#%% [markdown]
# ## 3.4 Find $HH^{-1}$ and then find $J^{-1}J$. Is $HH^{-1} == J^{-1}J$? Why or Why not?

#%%

print(f'H(H^-1):\n{numpy.matmul(numpy.linalg.inv(mH),mH)}')
print(f'(J^-1)J:\n{numpy.matmul(mJ,numpy.linalg.inv(mJ))}')
print(f'{numpy.matmul(numpy.linalg.inv(mH),mH).shape} != {numpy.matmul(mJ,numpy.linalg.inv(mJ)).shape}')


#%% [markdown]
# # Stretch Goals: 
# 
# A reminder that these challenges are optional. If you finish your work quickly we welcome you to work on them. If there are other activities that you feel like will help your understanding of the above topics more, feel free to work on that. Topics from the Stretch Goals sections will never end up on Sprint Challenges. You don't have to do these in order, you don't have to do all of them. 
# 
# - Write a function that can calculate the dot product of any two vectors of equal length that are passed to it.
# - Write a function that can calculate the norm of any vector
# - Prove to yourself again that the vectors in 1.9 are orthogonal by graphing them. 
# - Research how to plot a 3d graph with animations so that you can make the graph rotate (this will be easier in a local notebook than in google colab)
# - Create and plot a matrix on a 2d graph.
# - Create and plot a matrix on a 3d graph.
# - Plot two vectors that are not collinear on a 2d graph. Calculate the determinant of the 2x2 matrix that these vectors form. How does this determinant relate to the graphical interpretation of the vectors?
# 
# 

