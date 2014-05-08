
# coding: utf-8

## Data Management with pandas (Python) 1

# May, 2014
# 
# Chang Y. Chung

### Overview

# Pandas provide data structures that are flexible containers for lower dimensional data. Two main objects are `Series` and `DataFrame`:

# * `Series`: 1D labeled NumPy `ndarray`

# * `DataFrame`: 2D labeled, table of potentially heterogeneously-typed `Series`

# Also, there are `TimeSeries` (`Series` indexed by datetimes) and `Panel` (3D labeled table of `DataFrame`s).

### `Series` is a NumPy `ndarray` with an axis `index` and `name`.

# This is an `ndarray`.

# In[ ]:

import numpy as np
import pandas as pd

data = np.array([23, 31, 2, 3])
data


# This is an `ndarray`, super-charged.

# In[ ]:

s = pd.Series(data, index=['a', 'b', 'c', 'd'], name='mySeries')
s


# In[ ]:

s.index


### `Series` is like `ndarray`

# Get by an integer index:

# In[ ]:

s[0]


# Slicing returns a view:

# In[ ]:

s3 = s[:3]
s3[0] = 999
s


# In[ ]:

s


# Boolean indexing works, as well.

# In[ ]:

above_median = s[s > s.median()]
above_median


# Mathematical operations and functions are *vectorized*.

# In[ ]:

s ** 0.5


# In[ ]:

np.sqrt(s)


### `Series` behaves like a dictionary, as well.

# You can create a Series from a dictionary.

# In[ ]:

s = pd.Series({
    'Tom': 0,
    'Mike': 1,
    'Jane': 2,
    'Mary': 3,
    'Claudia': 4
})
s


# In-place sort by value. Default is ascending=True.

# In[ ]:

s.sort()
s


# In[ ]:

s


# `sort_index()` returns a new Series, sorted by index.

# In[ ]:

t = s.sort_index()
t


### Getting and setting are just like a dictionary.

# In[ ]:

s['Claudia'] = 23
s


# In[ ]:

'Tom' in s


# Use `np.nan` for default value for `get()`. Otherwise, `get()` will return `None`, when not found.

# In[ ]:

s.get('NOBODY', np.nan)


### `Series` automatically *aligns* data based on index.

# In[ ]:

s1 = pd.Series({'a':1, 'b':2, 'c':3})
s1


# In[ ]:

s2 = pd.Series({'b':20, 'c':10, 'd':9})
s2


# The resulting index is a *union* of the input indices.

# In[ ]:

s = s1 + s2
s


# Missing values (`np.nan` and `None`) can be dropped easily.

# In[ ]:

s.dropna()


### Example: Age-Specific Mortality

# Following German Rodriguez's nice Stata code available at his [web page](http://tinyurl.com/lndec87), let's graph Age-Specific Mortality.

# We are going to read a bit newer data on White Population in US, 2009, from CDC's ftp server address found in the publication:
# 
# > Arias, Elizabeth (2014) "United States Life Tables, 2009" *National Vital Statistics Reports* Vol. 62, No. 7. Hyattsville, MD: National Center for Health Statistics. Available [here](http://www.cdc.gov/nchs/data/nvsr/nvsr62/nvsr62_07.pdf).

# Download an excel file from CDC's ftp site and write it locally. `urllib.urlretrieve()` returns a tuple of (local) filename, and the header information, if successful.

# In[ ]:

import urllib

nchs = r'ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS' 
ftp  = r'{0}/Publications/NVSR/62_07/Table04.xls'.format(nchs)
xls = 'white2009.xls'
urllib.urlretrieve(ftp, xls)


### `pandas.read_excel` function can read the excel file

# Use zero-based indices for column and row numbers.

# In[ ]:

xls = 'white2009.xls'
options = {'header': 2, 'parse_cols': [2], 'skiprows': 6, 'skip_footer': 2}
df = pd.read_excel(xls, 'Sheet1', **options)


# We then copy only one column (Series) out of the DataFrame returned from `read_excel()`. Let's see first a few rows.

# In[ ]:

lx = df['l(x)'].copy()
lx.head()


# And the last a few rows.

# In[ ]:

lx.tail()


### We do some wrangling (munging, recoding, ...) and graphing

# In[ ]:

get_ipython().magic(u'matplotlib inline')

# convert to per-person
lx /= 100000.0

# cummulative hazard
Hx = - np.log(lx)

# shift(-1) brings up the value of the next row
hx = Hx.shift(-1) - Hx               

# take the mid-range value for age
hx.index += 0.5

# finally
hx.plot(logy=True)


### Let's see how the Series, hx, looks:

# In[ ]:

hx.head()


# In[ ]:

hx.tail()


### Fit a line for those over 30 years old.

# In[ ]:

import statsmodels.api as sm

# more munging
loghx = pd.Series(np.log(hx), name='loghx')[30:-1]
am30 = pd.Series(hx.index, index=hx.index, name='am30')[30:-1] - 30.0

# model fit
model = sm.OLS(loghx, sm.add_constant(am30))
result = model.fit()
print result.params
print 'R^2 : {:6.4f}'.format(result.rsquared)


### Finally, we graph.

# In[ ]:

# predicted value
pred = model.predict(result.params).astype(np.float64, copy=False)
hf = pd.Series(np.exp(pred), index=am30.index, name='hf')

# plot
hx.plot(logy=True)
hf.plot(logy=True)


### Quiz

# There are other life tables in the same [report](http://www.cdc.gov/nchs/data/nvsr/nvsr62/nvsr62_07.pdf):
# 
# * Total population
# * Males
# * Females
# * and so on ...

### String Methods

# `Series` has the built-in string method equivalents. They, however:
# 
# * are *vectorized*, so that it can be called with a whole `Series`;
# * made aware of the missing value (i.e., `np.nan`); and
# * have names that starts with `.str`

# In[ ]:

s = pd.Series(['Aaba', 'Baca', np.nan, 'CcDD'])
s


# In[ ]:

lowered = s.str.lower()
lowered


# `str.replace` and `str.findall` take regular expression, as well!

# * Finding out why Gracie could not medal in Sochi! :-) Notice that the `str.findall` returns a `Series`, whose elements are a list.

# In[ ]:

s = pd.Series(['Adelina', 'Yuna', 'Carolina', 'Gracie'])
ends_with_na = s.str.findall(r'.+na$')
ends_with_na


# * `str.replace()` relies on `re.sub()`.

# In[ ]:

s


# In[ ]:

s.str.replace(r'(.+na)', r'\g<0> medals')


### Summary

# * Important data structures in pandas
# * `Series` has *indexed* values and with a *name.
# * `Series` is (like) a NumPy `ndarray`.
# * `Series` is (like) a dictionary, as well.
# * `Series` automatically aligns data based on index.
# * `Series` has many `.str` vectorized methods.
