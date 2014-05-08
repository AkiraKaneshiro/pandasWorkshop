
# coding: utf-8

## Basic NumPy Refresher

# May, 2014
# 
# Chang Y. Chung

### Introduction

# * NumPy, Numerical Python, is the package for scientific computing and data analysis, implementing the multidimensional array object, `ndarray`.

# * Elements of an `ndarray` are homogeneous (all of the same `dtype`) and are indexed by a tuple of positive integers.

# * Dimensions are called `axes`. The number of axes is `rank`.

# In[ ]:

import numpy as np

a = np.array([[0, 1, 2], [3, 4, 5]])   # a 2D, 2 x 3, array
a


# In[ ]:

a.dtype


# In[ ]:

a.shape


### Get and set an element

# Get a (element) value via integer index.

# In[ ]:

a = np.array([[0, 1, 2], [3, 4, 5]])
a[0, 2]


# Set an element.

# In[ ]:

a[0, 2] = 99
a                  


### Why NumPy ndarray?

#### Because lists are not good for numerical calculations.

# Multiplication just repeats.

# In[ ]:

L1 = [2, 3]
2 * L1


# Addition concatenates.

# In[ ]:

L1 = [2, 3]
L2 = [5, 6]
L1 + L2


### NumPy `ndarray` provides element-wise operations.

# In[ ]:

x = np.array([2, 3])   # error if (2, 3) 
2 * x


# In[ ]:

y = np.array([5, 6])
x + y


### Familiar mathematical functions (`ufunc`'s) operate element-wise, as well.

# In[ ]:

x = np.array([1.0, 2.0, 3.0])
np.exp(x), np.sqrt(x), np.log(x)


# See the list of available `ufunc`'s [here](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).

### NumPy array creation

# In[ ]:

a = np.array([2, 3, 4], dtype=float)  # from list
a, a.dtype, a.shape            


# In[ ]:

b = np.arange(10)   # short-cut for np.array(range(...))
b


### Some convenient functions.

# In[ ]:

Z = np.zeros((2, 3))
Z


# In[ ]:

Ones = np.ones((10,)) 
Ones


# In[ ]:

I = np.eye(4)
I


# 5 evenly spaced numbers from 0.0 to 2.0, inclusive at both ends.

# In[ ]:

x = np.linspace(0.0, 2.0, 5)
x


# In[ ]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

x = np.linspace(0.0, 2.0, 100)
y = np.sin(x * np.pi) + x ** 2 - np.pi
plt.plot(x, y)


### Many unary operations are implemented as methods.

# In[ ]:

np.random.seed(12345)
a = np.random.random((2, 4))
a


# Sum of all elements, regardless of shape.

# In[ ]:

a.sum()


# Sum of each column.

# In[ ]:

a.sum(axis=0)


# Sum of each row.

# In[ ]:

a.sum(axis=1)


### Changing the shape

# In[ ]:

c = np.arange(10) 
c


# In[ ]:

c.shape = (5, 2)
c


# In[ ]:

d = c.T # transpose() also works
d


# In[ ]:

f = d.flatten() # ravel() also flattens
f


# In[ ]:

f = np.array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
g = f.reshape(5, 2)
g


# Array elements order is, by default, rightmost index *changes the fastest* (C-style). This applies both when the array is flattened and is (re)shaped.

### Slicing works as expected, but returns a view.

# In[ ]:

a = np.arange(15).reshape(3, 5)
a


# A view of the third row and all columns.

# In[ ]:

v = a[2, :]
v


# Change a value of a view like so:

# In[ ]:

v[0] = 999
v


# You change the underlying array data as well.

# In[ ]:

a


### In order to create an independent copy, use copy() method.

# In[ ]:

b = a.copy()
b[2, 0] = 888
b


# a remains unchanged.

# In[ ]:

a 


### Boolean indexing returns an array and works great for setting some cells to certain values.

# In[ ]:

b = a.copy()
idx = np.logical_or(b == 9, b == 999)
idx


# In[ ]:

b[idx] = 0
b


# Selecting the first and the third rows.

# In[ ]:

idx = np.array([True, False, True])
r = b[idx, :]
r


### For matrix multiplication, use `np.dot`.

# In[ ]:

M = np.arange(6.0).reshape(3,2)
M


# In[ ]:

N = 2 * M.T
N


# In[ ]:

M * N


# In[ ]:

np.dot(M, N)


### One dimensional array is a one dimensional array.

# In[ ]:

v = np.ones(2.,)
v


# In[ ]:

np.all(v == v.T)


# In[ ]:

M, v


# In[ ]:

np.dot(M, v)


### Reference for all the NumPy methods and routines found at the [reference page](http://docs.scipy.org/doc/numpy/reference/routines.html)

### An Example: Random Walks

# > This example is mostly based on a section in McKinney (2012, Ch.4), except that a step can be taken one of the three, not two, directions. All the code has been re-written.

# Let's simulate a simple random walk.
# 
# Let $t = 0, 1, \ldots$ denote (discrete) time. Starting from the initial position of 0, a step is taken randomly either to a positive direction (+1), to a neutral direction (0), or to a negative direction (-1) at each time. A $walk$ is a list of positions indexed by time, $t$.

### Without NumPy

# The function, `randint(a, b)`, returns an integer from the range `a` to `b`, inclusive at both ends.

# In[ ]:

import random

rnd = random.Random(1234)
T = range(21)            
walk = []

for t in T:              
    if t == 0:  
        pos = 0
    else:
        step = rnd.randint(0, 2) - 1   
        pos += step
    walk.append(pos)

print "walk = ", walk
plt.plot(T, walk)                              


### Using `np.cumsum()`

# Notice that each element in the $walk$ is simply a cumulative sum of the preceeding (and the current) steps. That is, letting $Step_t$ be the random step taken at time $t$, then:

# $$
# \begin{align}
# walk[t] & = & walk[t-1] + Step_t \\
#         & = & walk[t-2] + Step_{t-1} + Step_t \\
#         & \ldots & \\
#         & = & walk[0] + Step_1 + \ldots + Step_{t-1} + Step_t
# \end{align}
# $$

# Let $walk[0] = Step_0 = 0$ for simplicity, then we define a $walk$ with cummulative sums of steps.
# 
# Conveniently, NumPy ndarray provides a method, `cumsum()`. Let's take advantage of the method and simplify the simulation as so:

# The NumPy `randint(low, high)` returns an integer from low (inclusive) to high (exclusive).

# In[ ]:

np.random.seed(1234) 
T = range(21)        

steps = np.random.randint(0, 3, size=len(T)) - 1 
steps[0] = 0
walk  = steps.cumsum()

print walk
plt.plot(T, walk)


# Notice that we are using a different random number generator. Even with the same random seed, we get a different sequence of random numbers.

### First Crossing Time

# Say we would like to know how long it took a longer random walk to get at least 10 steps away from the origin, 0, in either direction.
# 
# NumPy `ndarray`'s `argmax`() function returns the first index of the maximum value. We are re-using the above code, but are setting the arguments so that we are simulating a longer walk.

# In[ ]:

np.random.seed(1235)
T = range(101)

steps = np.random.randint(0, 3, size=len(T)) - 1
steps[0] = 0
walk  = steps.cumsum()

faraway = (np.abs(walk) >= 10) 
cross = faraway.argmax() 

print cross 
plt.plot(T, walk)
plt.axhline(y = 10.0, ls='dashed')
plt.axvline(x = cross, color='red')


### Simulating Many Random Walks at Once

# Usually we would like to simulate repeatedly.
# 
# In our context, this means generating multiple walks. NumPy `ndarray` can have multiple axes or dimensions. We just introduce the second axis representing the repetition.

# In[ ]:

np.random.seed(1235) 
T = xrange(1000)
num_walks = 5000                              

steps = np.random.randint(0, 3, size=(num_walks, len(T))) - 1
steps[:, 0] = 0
walks = steps.cumsum(1)

import time
from IPython.display import display, clear_output 

f, ax = plt.subplots()
for i in xrange(0, num_walks, 160):   
    time.sleep(0.2)
    ax.plot(T, walks[i, :], color='darkgrey')
    clear_output(wait=True)
    display(f)
plt.close()


### Quiz. 

# Simulate 5,000 random walks, each of which runs from the position of 0 at $t == 0$ to $t == 700$. For each walk, find out the "crossing time" to +30 or -30, that is the first time when the position reaches either +30 or -30. Calculate the mean crossing time over all 3,000 random walks. Notice that not all the walks may reach $\pm 30$. In that case, use only those walks that ever reach $\pm 30$ within the given time $t <= 700$. Make sure to use a random seed so that the simulation can be replicated exactly.

# In[ ]:

# an answer
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)
T = xrange(700)
num_walks = 5000                              

steps = np.random.randint(0, 3, size=(num_walks, len(T))) - 1
steps[:, 0] = 0
walks = steps.cumsum(1)

# filter, get crossing times, and print the mean crossing time
hits30 = (np.abs(walks) >= 30).any(1)
xtimes = (np.abs(walks[hits30]) >= 30).argmax(1)
print "mean crossing time (for walks that reach +30 or -30): ", xtimes.mean()


### References and Resources

# * Bressert, Eli (2012) *SciPy and NumPy: An Overview for Developers* O'Reilly. ISBN:149305466. at [Amazon](http://www.amazon.com/SciPy-NumPy-An-Overview-Developers/dp/1449305466)
# * McKinney, Wes (2012) Chapter 4. "NumPy Basics: Arrays and Vectorized Computation" in *Python for Data Analysis:Data Wrangling with Pandas, NumPy, and IPython* O'Reilly. ISBN:1449319793. at [Amazon](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793)
# * NumPy Reference at [NumPy site](http://docs.scipy.org/doc/numpy/reference/)
# * NumPy User Guide at [NumPy site](http://docs.scipy.org/doc/numpy/user/)
# * Tentative Tutorial at [SciPy site](http://wiki.scipy.org/Tentative_NumPy_Tutorial)
