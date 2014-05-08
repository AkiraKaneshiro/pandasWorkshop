
# coding: utf-8

## Data Management with pandas (Python) 2

# May, 2014
# 
# Chang Y. Chung

### What is `DataFrame`?

# The reference defines it:
# 
# > Two-dimentional size-mutable, potentially heterogenious tabular data structure with labeled axes (rows and columns)

# Here is a longer way to approach it from the perspective of data structures.

# * NumPy ndarray is an array of `dtype` elements `shape`d. 

# In[ ]:

import numpy as np

rid = np.array([0, 1, 2, 3])
rid, rid.dtype, rid.shape


# * pandas `Series` is an ndarray with an `index` and `name`.

# In[ ]:

import pandas as pd

s1 = pd.Series(
    ['Abe', 'Babe', 'Colombe', 'Daube'],
    index = rid,
    name='firstname')
s1


# Let's create another `Series`, age.

# In[ ]:

s2 = pd.Series([23, 20, 22], index=[0, 1, 3], name='age')
s2


# * `DataFrame` is an aligned dictionary of `Series`'.

# In[ ]:

df = pd.DataFrame({s1.name: s1, s2.name: s2})
df


# We now have two columns nicely aligned by matching the indices. Age did not have a value that was indexed by the number 2. When age was brought into the DataFrame, pandas aligned the indices and put the missing value, `np.nan`, there.

# * Important properties of `DataFrame` includes: `index` (row label or 'observation id') and `columns` (column labels or 'variable name').

# In[ ]:

df.index


# In[ ]:

df.columns


# In[ ]:

df.dtypes


# A NumPy ndarray or a Series has a `dtype`. `DataFrame` has `dtypes`, since each column has its own `dtype` and they can be different. A `dtypes` is a `Series`.

#  * Notice that the age entry in the `dtypes` became *float*. This is because pandas does not have a proper missing value for an integer type. We will talk about this more later when we talk about missing values.

### Quiz

# Create a data frame of Age Specific Fertility Rate (ASFR) for women in Republic of Korea (South Korea) and US, for 2000-2005. Age ranges and ASFRs for five-year age range for both countries are geven below. 
# Data are from *UN World Population Prospect 2006*.

# In[ ]:

start = pd.Series(range(15, 46, 5))
finish = start + 4
age_range = ['{0}-{1}'.format(s, f) for s, f in zip(start, finish)]
age_range


# In[ ]:

korea = [3.2, 33.1, 119.9, 72.8, 16.8, 2.5, 0.2]
usa = [43.7, 104.5, 114.5, 93.5, 42.8, 8.5, 0.5]


# In[ ]:

# create a DataFrame here.


### Creating DataFrame and I/O

# There are many different ways to create a data frame.

# * The constructor is flexible and takes different things as `data` arguments, not just a `Series`:
#     - a dictionary of columns
#     - a list of dictionaries
#     - a list of tuples
#     - another DataFrame
#     - ...

# Below is an example of the `DataFrame` constructor taking in a list of tuples.

# In[ ]:

df2 = pd.DataFrame([
  ('SNSD', 'female', 9),
  ('Big Bang', 'male', 5)
], columns=['band', 'gender', 'size'])
df2


# * There are many ways to create a `DataFrame`, since pandas has many importing functions that return a `DataFrame`.

# | data in this format        | import function |
# |-----------------|-----------------|
# | pickled object  | `read_pickle`   |
# | delimited text  | `read_table`    |
# | CSV             | `read_csv`      |
# | fixed-width text| `read_fwf`      |
# | clipboard       | `read_clipboard`|
# | excel           | `read_excel`    |
# | JSON string     | `read_json`     |
# | HTML table      | `read_html`     |
# | HDFStore(HDF5)  | `read_hdf`      |
# | SQL database    | `read_sql`      |
# | Google BigQuery | `read_gbq`      |
# | Stata           | `read_stata`    |

# * `DataFrame` also has many corresponding methods that go the other way. That is, to *serialize* itself (or to write itself out) in various formats, including `to_pickle`, `to_csv`, `to_hdf`, `to_excel`, `to_json`, `to_stata`, `to_clipboard`, ...

### I/O into and from a `DataFrame`

# Here is an example of:
# 
#     1. Importing a Stata data file (auto.dta) into a pandas DataFrame
#     2. Saving only the foreign rows into a JSON file (foreign.json); and then
#     3. Read it back to `DataFrame` (df_foreign)

# In[ ]:

auto = pd.read_stata('auto.dta')
auto.head(3)


# Save the `columns` for the later use.

# In[ ]:

cols = auto.columns.copy()
cols


# Select foreign makes only.

# In[ ]:

foreign = auto[auto.foreign == 'Foreign']
foreign.head(3)


# And then write it out as a JSON string.

# In[ ]:

foreign.to_json('foreign.json')


# Reading it back using the `read_json` method. While writing to JSON, the order of columns has  been changed. We fix it by re-indexing along the second axis (i.e. columns). And then show the first 3 rows.

# In[ ]:

df_foreign = pd.read_json('foreign.json')
df_foreign.reindex_axis(cols, axis=1).head(3)


### Getting and Setting

# Once you have your data in a `DataFrame`, then you may naturally want to retrieve (*get*) and/or to change (*set*) the value or values.

# Recall when we get/set a value in a NumPy ndarray, we use an integer position (and zero-) based index.

# In[ ]:

import numpy as np

a = np.array([
    [0, 1, 2],
    [3, 4, 5]
])
a[1,2] # get


# In[ ]:

a[1, 2] = 99 # set
a


# pandas data structures have *labels*, that can be non-integers. This permits us getting/setting via labels, in addition to doing so via integer positions. For instance,

# In[ ]:

s = pd.Series([0, 1, 2], index=['Abe', 'Betty', 'Caroline'])
s


# In[ ]:

print 'via label.            s[\'Betty\']=', s['Betty']
print 'via integer position. s[1]=', s[1]


# Getting a value does work. Considering that an index can be an integer, a better way is to tell pandas explicitly which way you intend the *key* should be understood.

# If you mean the key to be:
# 
# * a *label* (or index), then use `.loc`; 
# * an integer *position*, then use `.iloc`.

# In[ ]:

print 'via integer position. s.loc[\'Betty\']=', s.loc['Betty']
print 'via label.            s.iloc[1]=', s.iloc[1]


# `.ix` permits both but assume it a label first, if not found, then a position. It is better to avoid it, if possible.

# In[ ]:

s.ix['Betty'], s.ix[1]


# This works the same way for a `DataFrame`. Only that it is a 2D object and we have a 2-tuple of integer positions with `.iloc` or a 2-tuple of labels with `.loc`.
# 
# That is, given the following DataFrame. 

# In[ ]:

df = pd.DataFrame([
    (23, 'Abe'),
    (20, 'Babe'),
    (22, 'Daube')
], index=['R1', 'R2', 'R3'], columns=['age', 'firstname'])
df


# In[ ]:

df.loc['R1', 'age'], df.iloc[0, 0]


# Setting a value is straight forward as well.

# In[ ]:

df.loc['R1', 'age'] = 100
df


# In[ ]:

df.iloc[0,1] = 'Abraham'
df


# `.loc` can actually do more:
# 
# * Raises a `KeyError` exception when the label was not found.
# * Can take a list or array of labels, in either dimentions or both.
# * Can take a slice with labels, i.e. 'age':'firstname'. Notice that both start and the stop parts are *inclusive*!!!
# * Can take a boolean array.

# In[ ]:

df.loc['R1', 'age']


# In[ ]:

df.loc[['R1', 'R3'], ['age', 'firstname']]


# Remember the *stop* part of a label slice is *inclusive*. That is, 'R2' row is *included* below.

# In[ ]:

df.loc['R1':'R2', 'age':'age']


# `.iloc` is also powerful:
#     
# * Raises an `IndexError` exception when the integer position is out of bounds (0 to length -1).
# * Can take an integer list or array.
# * Can take a slice.
# 
# 

# Let's see the `df` again.

# In[ ]:

df


# In[ ]:

df.iloc[0, 1]


# In[ ]:

df.iloc[[0, 2], [0]]


### Quiz

# How many rows below returns? Assume that the DataFrame df has more than three rows.

# In[ ]:

df.iloc[0:2, :]


# Specifying the key using `.loc` and `.iloc` is the best practice. For the sake of convenience, however, a simple bracketed syntax is often used.

# * A simple bracketed slice gets/sets the rows.

# In[ ]:

df2 = df[:2]
df2


# * A simple bracketed column name (or list of names), gets/sets the columns.

# In[ ]:

df2 = df[['firstname', 'age']] # col order changed
df2


# Here is an Python-like idiom of swapping columns in a DataFrame. In this case, it is rather silly. :-)

# In[ ]:

df[['firstname', 'age']] = df[['age', 'firstname']]
df


### Quiz

# Given a DataFrame df1, create another, df2, which looks like df3.

# In[ ]:

rows = range(3)
cols = list('ABCDE')
df1 = pd.DataFrame(np.arange(15).reshape(3,5), index=rows, columns=cols)
df1


# In[ ]:

df3 = pd.DataFrame([
    ( 0,  1,  -2, 23),
    (10, 11, -12, 45),
], index=[0,2], columns=list('ABCF'))
df3


# An answer:

# In[ ]:

df2 = df1.loc[0:2:2, 'A':'C']
df2['C'] *= -1
df2['F'] = [23, 45]
all(df2 == df3)


### Missing Data

# * Recall that `None` is a Python universal object. It is a placeholder and often evaluated as False, but it compares itself True.

# In[ ]:

None, type(None), None == None


# * `np.nan` is also a placeholder. It is of a float type, representing 'Not a Number', something like log(-1). It *always* compares `False`, including being compared to itself, of which is the defining characteristic. Use `np.isnan()` to identify a value is np.nan.

# In[ ]:

np.log(-1), type(np.nan)


# In[ ]:

np.nan < 1.0, np.nan >= 1.0, np.nan == np.nan


# In[ ]:

np.nan != np.nan, np.isnan(np.nan)


# * pandas uses `np.nan` (and `None`) to represent missing data. They are, by default, *excluded* from calculations.

# pandas automatically promotes int `dtype` to a float, when there is an np.nan within a Series. Similarly, boolean `dtype` is cast to object. Here is an example of the former.

# In[ ]:

left = pd.DataFrame({'a': [23, 13, 25, 42]}, index=range(4))
right = pd.DataFrame({'b': [11, 11, 11]}, index=[0, 2, 3])

together = pd.merge(left, right, how='left', left_index=True, right_index=True)

print together
print together.dtypes


# * pandas provide `isnull()` and `notnull()` functions to identify missing values. They return booleans. 

# In[ ]:

df = pd.DataFrame(np.log(-np.eye(3)))
df


# In[ ]:

pd.isnull(df)


# In[ ]:

pd.notnull(df)


# * `dropna()`, removes either columns (axis=1) or rows (axis=0) with *any* missing values. 

# In[ ]:

df.iloc[1,1] = 23
df


# In[ ]:

dropped = df.dropna(axis=0)
dropped


# * When filling the missing values, `fillna()` and `interpolate()` come in handy.

# In[ ]:

df = pd.DataFrame(np.arange(18.).reshape(6, 3), columns=list('abc'))
df.iloc[2:4, [0, 2]] = np.nan
df.iloc[1, 1] = np.nan
df


# In[ ]:

df['a'] = df['a'].interpolate(method='polynomial')
df['b'] = df['b'].fillna(999)
df


### Some Data Munging Tools: Append, Concat, Group By

# Let's download some World Bank's World Development Indicators.
# 
# > This example is largely based on the "World Bank" section of *pandas 0.13.1 documentation* available [here](http://pandas.pydata.org/pandas-docs/stable/remote_data.html) but was expanded to demonstrate more methods and functions.

# First, we download a GDP per capita series and a fertility rate. The search method shows available series.

# In[ ]:

from pandas.io import wb

wb.search('fertility').iloc[:, :2]


# Let's choose two series: one fore GDP per capita and another for Total Fertility Rate. We request all the available countries and some years.

# In[ ]:

ind = ['NY.GDP.PCAP.KD', 'SP.DYN.TFRT.IN']
df = wb.download(indicator=ind, country='all', start=1950, end=2014)


# Shorten the column labels. and let's see the dataframe. It has a MultiIndex (or hierarchical index).

# In[ ]:

df.columns = ['gdp', 'tfr']
df.head()


# Before we do anything, let's drop any rows that has missing values, and convert both columns to numbers.

# In[ ]:

df = df.dropna()
df = df.convert_objects(convert_numeric=True)
df.to_pickle('df.pkl')
df.dtypes


# How many records do we have? We can get summary data using `describe()` method.

# In[ ]:

df.describe()


# Let's now try some group by using the multiIndex we have. Let's aggregate our data at the country level by calculating a mean over years within each country. Amazinly, it is just one line.

# In[ ]:

country = df.groupby(level=['country']).mean()
print country.describe()


# We save the country data file locally, only after we sort it by the gdp.

# In[ ]:

country.sort(columns=['gdp'], axis=0, inplace=True)
country.to_pickle('country.pkl')


# We can graph gdp and tfr together, 

# In[ ]:

country = pd.read_pickle('country.pkl')
country['log_gdp'] = np.log(country['gdp'])
country = country.drop('gdp', axis=1)
import matplotlib.pyplot as plt
plt.scatter(country['tfr'], country['log_gdp'])


# Another way is to aggregate over countries, so that we end up with yearly data. Aggregation is simple using group by again.

# In[ ]:

df = pd.read_pickle('df.pkl')
year = df.groupby(level='year').mean()
year['ln_gdp'] = np.log(year['gdp'])
year = year.drop('gdp', axis=1)
year.sort(columns=['ln_gdp'], axis=0, inplace=True)
year.to_pickle('year.pkl')
year.plot()


# Suppose that we separate out before and after year 2000 into two datasets, like so. It happened that the index is now a list of Unicode strings.

# In[ ]:

upto2000 = year.loc[:u'2000']
after2000 = year.loc[u'2001':]
print upto2000.tail(3), '\n', after2000.head(3)


# Both `pd.concat()` or `df.append()` can put them together.

# In[ ]:

yearAgain = pd.concat([upto2000, after2000])
print yearAgain.head(3), '\n', yearAgain.tail(3)


# In[ ]:

yearAgain2 = upto2000.append(after2000)
print yearAgain2.head(3), '\n', yearAgain.tail(3)


### Summary

# There are so many other powerful functions and methods remain in pandas and other related packages. For example, we haven't had chance to talk about `merge` and `join`. We have not talked about powerful 'reshaping' (`stack`, and pivot tables), nor Time Series related topics.
# 
# We did study together, though, the following topics:
# 
# * What is a DataFrame?
# * Creating DataFrame and I/O
# * Getting and Setting values
# * Missing Data
# * Append, Concat, Group By

### References

# * McKinney, Wes (2012) Chapter 4. "NumPy Basics: Arrays and Vectorized Computation" in *Python for Data Analysis:Data Wrangling with Pandas, NumPy, and IPython* O'Reilly. ISBN:1449319793. at [Amazon](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793)
# * pandas development team (2014) "pandas: powerful Python data analysis toolkit" Version 0.13.1 Available at [pandas site](http://pandas.pydata.org/pandas-docs/stable/index.html)
