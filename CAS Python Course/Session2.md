---
title: "CAS Python Session II"
xfiller: not used but do not delete
date: "Created 2020-07-03 08:10:27.466236"
author: Stephen J. Mildenhall
numbersections: true
fontsize: 10pt
geometry: margin=1in
cla: --standalone
cla: --variable urlcolor=cyan
cla: -f markdown+smart+yaml_metadata_block+citations+pipe_tables+table_captions
cla: -t latex
cla: -o pdf
debug: true
---

# CAS 2020 Python Workshop: Session II Pandas

## Session Descriptions

Welcome to CAS Python Workshop

| No | Date       |Lead |   Contents  |
|:---|:-----------|:----|:------------------------------------------------------------------|
| 1  |  July 15   | BF  | Python programming basics variables, types, lists, dictionaries, functions, dates, strings, dir, help Simulated transactional data, computing Earned Premium (see 5)
| **2**  |  **July 22**   | **SM**  | **Pandas 1: DataFrame creation and basic data manipulation; make a triangle, make development factors, make an exhibit from the CAS Loss Reserve Database**
| 3  |  July 29   | BF  | Pandas 2: data io with external sources: Excel, CSV, markdown, HTML, web; advanced data manipulation: querying, merging, indexes, stack, unstack, pivot-table, tidydata Prem and loss simulated data…
| 4  |  Aug 5     | SM  | Pandas 3: Visualization and Reporting plotting plus matplotlib, geopandas, jinja, COVID data, NY Auto data
| 5  |  Aug 12    | SM  | Simulation modeling, pandas, numpy, scipy.stats Cat model Creating data for session 1
| 6  |  Aug 19    | BF  | Linear regression, lm, glm, sklearn Triangles analysis

## Session II Agenda: `pandas`

* Recall from Session I: lists, dictionaries, functions
* Two handy Python user-defined functions
* `pandas` Introduction
* Creating DataFrames and Accessing Elements
* Extracting information from DataFrames
* Plotting: Bar Chart, Scatter Plot, Histograms
* Web data access
* Grouping and Aggregation
* Stacking and Pivoting
* Triangles...


### Reference: Functions We Will Discuss
* `DataFrame`, `Series`

* `head`, `tail`

* `unique`, `value_counts`

* `read_csv`

* `loc`, `slices`, `xs`

* `query`

* `pivot`, `stack` `and` `unstack`

* `pivot_table`

* `groupby` (.`groups`, .`get_group`, `as_index`)

* `sum`, `mean`, `std` `etc`.

* `aggregate`

* `plot`

<!--
MISSING
`describe`
`create_index`, `reset_index`
* `MultiIndex`
**`concat`**, `append`, `keys`
|  Slot | Tasks     | Functions                                                                              |
|------:|:----------|:---------------------------------------------------------------------------------------|
|  0-10 | Intro     | sdir, help, np.random (rand, lognormal, binomial choice, poisson), dictionaries        |
| 10-20 | Create df | from dictionary, from array; access row, access column; indexes, name indexes; sorting |
| 20-30 | Claims df | integer indexes, multi index, grouping, histogram; type of claim
| 30-40 | Premium df | merging,
| 40-50 |
| 50-60 |
| 60-70 |
| 70-80 |
| 80-90 |
 -->

## Recall from Session 1: lists and indexing

```python
a = [1,2,3,4,6]

a[1], a[3:], a[-2], a[-2:], a[::-1]
```

### Custom functions

```python

def myfunction(x):
    return x * 10

myfunction(20)
```

### Dictionaries and comprehensions

Count letters in a sentence with a dictionary comprehension. Remember dictionaries are  {key: value}  pairs.

```python

s = "jack and jill went up the hill to fetch a pail of water "

{ i : s.count(i) for i in set(s) }
```

### Custom functions, default arguments
```python

def letter(s, omit=''):
    return { i : s.count(i) for i in s if i not in omit }

letter(s), letter(s, ' aeiou')

```

### Exact same function counts words(!!!)

`split` breaks a string into words.

```python
print(s.split())

letter(s.split())

```

## Handy Utility Functions

* `dir`: what can a function do?
* `sdir`: better version of `dir`
* `all_doc`: all the documentation on a function

```python
dir(str)
```

### (a) What Can a Function Do?

There is no distinction between a variable, data and a function. All equal citizens to Python.

```python
def sdir(x, colwidth=80):
    """
    Directory of useful elements, wrapped
    """
    from textwrap import fill

    # all the work is in this line:
    l = [i for i in dir(x) if i[0] != '_']

    # frills to printout nicely
    mx = max(map(len, l))
    mx += 2
    fs = f'{{:<{mx:d}s}}'
    l = [fs.format(i) for i in l if i[0] != '_']
    print(fill('\t'.join(l), colwidth))

sdir(str)

```

### (b) Get all the Help

`?function` or `help(function)` shows the help on a function. Custom functions can have help: the string immediately after the declaration.

```python
def all_doc(obj):
    """
    print the documentation on every public callable method of obj
    """

    s = f'{str(obj)} (type={type(obj)}) Documentation'
    print(f'{s}\n{"="*len(s)}\n\ntype={type(obj)})\n\nMETHODS\n=======\n')

    # iterate over methods
    for x in dir(obj):
        if x[0] != '_':
            # get the method
            method = getattr(obj, x)
            if callable(method):
                # if it is callable, i.e. a function, show help
                # help lives in obj.__doc__
                print(f'{x}\n{"~"*len(x)}\n{method.__doc__}\n')

all_doc(bytes)

```

## The Zen of Python

`import this`

    The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!

## Module 1: Introduction
* Libraries for today: `numpy` (`np`), `pandas` (`pd`), `matplotlib`, `matplotlib.pyplot` (`plt`)

* `np.random`, `rand`, `lognormal`, `choice`, `poisson`

* Select from list `np.random.choice(list('ABCDE'), 10)`

* Select from list, non-uniform prior `np.random.choice(list('ABCDE'), 10, p=[.4,.3,.2,.05,.05])`

* Dictionaries and dictionary comprehension: count letters in a sentence, `f = {i: s.count(i) for i in set(s)}`; apply to random selection using `''.join()` or convert to `list`

* Start by importing, very important!

```python
import numpy as np

r_letters = np.random.choice(list('ABCDE'), 10)

r_unif = np.random.rand(10)

r_lognorm = np.random.lognormal(10, .2, 10)

r_letters, r_unif, r_lognorm
```

### Exercise

* Simulate random letters from ABCDEF...

* Summarize by letter and check you get distribution you expect, convert sample to list using `list(...)`

* Add a prior distribution

* Summarize again and check you get distribution you expect


### solutions to exercise
```python

letters = list(np.random.choice(list('ABCDE'), 500, p=[.4,.3,.2,.05,.05]))
n = len(letters)

freq = { i: letters.count(i) / n for i in 'ABCDE'}

freq
```

## Module 2: Create a DataFrame and Access Elements

Finally, Pandas: your spreadsheet in Python.


### Creating a DataFrame
* Create from dictionary: keys become column names.

* Create from list of lists

* Allows mixed data types.
workd
* Nice Jupyter Lab output.

* Row and column indexes in bold

* Again, start with import!

```python
import pandas as pd

df  = pd.DataFrame({'a': range(100, 110),
                    'b' : np.random.choice(list('ABCDEF'), 10),
                    'c' : np.random.rand(10),
                    'd': pd.to_datetime('2020/07/05')+pd.to_timedelta(np.arange(10), unit='D')
                    })
df
```

### Accessing Data within a DataFrame

* Access column as item and attribute

* Access row or element using  `loc` for row, both

* Access with logic: `df.c < .25`,  `query`

* Slicing with `loc`, `df.loc[1:4, 'a':'c']` **includes endpoints**; no well defined notion of the *one before* the end

* Integer indexing `iloc`

* `query`

* `display` vs. `print`; intermediate results vs. final result

```python
df['a'], df.a

```

### Accessing Data within a DataFrame: Row Index

```python
df.loc[3]
```

### Accessing Data Within a DataFrame: Row and Column Index

```python
df.loc[3, 'd']
```

### Accessing Data Within a DataFrame: Range of Rows

```python
df.loc[:3]
```

### Accessing Data Within a DataFrame: Range of Rows

```python
df.iloc[::2]
```

### Accessing Data Within a DataFrame: Logic

```python
df.a < 105
```

### Accessing Data Within a DataFrame: Logic

```python
df.loc[df.a < 105]
```

### Accessing Data Within a DataFrame: The Query Operator

* Very powerful, very fast

* SQL like

* Access elements with @

```python
df.query(' .4 < c < .8 ')
```

### Add Data

* Create new columns with math, from old columns

* Create new row

* Can't create on the fly like tidyverse

```python

df['E'] = df.a / df.c

df.loc[100, :] = (110, 'Z', .11223344, pd.to_datetime('2020/11/03'), np.nan)

display(df)
```

### Exercise

Add a column F equal to E * c, check it equals a

Remember everything is case sensitive!

### Solution

```python

df['F'] = df.E * df.c

df.F == df.a

```

...wait, what?

### Sorting

* `sort_values` and `sort_index`: return a new object; `ascending=False` for descending order

```python
df.sort_values('c')
```

### Exercise
* Create function to take a string, make lower case, break it into words, and create a DataFrame with columns `word`  and `freq` counting word frequency

* Reconsider your approach if you go beyond five lines of code...

* *Optionally* make case independent, default arguments `case=False` argument

* Optionally sort output by descending freq

* Extra credit: strip out punctuation

### Solution
```python
def word_count(s):
    """
    always document here!
    """
    word_list = s.lower().split()
    df = pd.DataFrame([[i, word_list.count(i)] for i in set(word_list)], columns=['word', 'freq'])
    return df

def word_count_ex(s, excluded_chars='",\';:()[]!?@#$%&=\\.'):
    """"
    word counter with excluded characters
    """
    for i in excluded_chars:
        s = s.replace(i, ' ')
    # which is kinda yucky
    word_list = s.lower().split()
    df = pd.DataFrame([[i, word_list.count(i)] for i in set(word_list)], columns=['word', 'freq'])
    df['letters'] = df.word.str.len()
    df = df.sort_values('letters', ascending=False)
    return df
```

* Apply to `In[xx]`

## Module 3: Requests (interlude) and Graphics

### Read Longer Document
* read longer document, *The Declaration of Independence* (`di`)

* `requests` library for Internet calls

* create data frame of word count etc., using previous function

```python
import requests

r = requests.get('http://www.mynl.com/RPM/di.txt')

di = r.text

print(di[:100])

df = word_count_ex(di)

df
```

### Exercise

Sort `df` by descending frequency

### Solution

Often helpful to only show `head` or `tail`

```python
df.sort_values('freq', ascending=False).head(10)
```

### Exercise

Show five most common words that occur at least 10 times and that have five or more letters, sorted descending order by frequency.

Note use of `\` for line continuation; nothing can appear after it!

Indentation after first `df` is free-form.

### Solution

```python
df.query(' freq >= 10 and letters >= 5 '). \
    sort_values('freq', ascending=False). \
    head(5)
```


### Graphics! The Bar Chart

* Bar chart of word freq

* `bar` for vertical and `barh` for horizontal

* Subset to longer words using Exercise

* Breakdown `set_index` statement

```python
bit = df.query(' freq >= 5 and letters >= 5 '). \
    sort_values('freq', ascending=False). \
    head(10). \
    set_index('word')

display(bit)
bit.plot(kind='bar', rot=315)
```

### Just Plot Frequency, Not Letters

```python
bit['freq'].plot(kind='barh')
```

### `[x]` vs `[[x]]` is the Same as R

* `bit['freq']` returns a  Pandas `Series` object
* `bit[['freq']]` returns a  Pandas `DataFrame` object

```python
display(bit['freq'])
display(bit[['freq']])
```

### Scatter Plot
* Scatter plot: frequency vs. number of letters

```python
df.plot(kind='scatter', x='letters', y='freq', marker='o', alpha=0.4)
```

### Exercise
* Jitter number of letters and re-plot, i.e., add a new column equal to the number of letters plus a small random number
* Explore alpha, different markers, e.g., `'x'`, change marker size `s=2`

### Solutions

`lw` sets the line width

```python
df['letters_j'] = df.letters + np.random.rand(len(df)) * .8 - 0.4
df.plot(kind='scatter', x='letters_j', y='freq', marker='x', alpha=0.4)
df.plot(kind='scatter', x='letters_j', y='freq', marker='x', s=10, lw=.25, alpha=0.8)
```

### Nicer Plots and Plot Decorations

```python
%config InlineBackend.figure_format = 'svg'

ax = df.plot(kind='scatter', x='letters_j', y='freq', marker='x', s=10, lw=1)
ax.grid()
ax.set(title='My Title', xlabel='(Jitterd) Number of Letters', ylabel='Word Frequency')
```

### Extended Exercise
* Create data frame with [100+] claims, loss=lognormal(10,1), kind=randomly selected from A-E, open=random 0,1

* 30% chance claim closed (`choice` or `np.random.binomial(1, 0.3, n)`)`

* Name index claim_index

* Create new column log_loss using np.log()

* `df = pd.DataFrame({'loss': something, ... })`

* Extra credit: make the mean vary by kind

### Solution to extended exercise
```python
n = 1000
df = pd.DataFrame({'loss': np.random.lognormal(10,1,n),
                   'kind': np.random.choice(list('ABCDE'), n),
                   'open': np.random.binomial(1, 0.3, n)})
df.loc[df.kind=="A", "loss"] *= 1.95
df.loc[df.kind=="B", "loss"] *= 1.45
df.loc[df.kind=="D", "loss"] *= 0.95
df.loc[df.kind=="E", "loss"] *= 0.65
df['log_loss'] = np.log(df.loss)
df.head(10)
```

## Module 4: Grouping and More Charting

### Histograms
* Histogram of claims

* `ec` = edge color, puts nice border around bars

* `bins` determines number of bins or bin boundaries

```python
df.log_loss.hist(bins=50, ec='white', lw=0.5)
```

### Grouping
* Grouping: `group_by` breaks DataFrame into groups

* Apply a function

* `agg` to summarize

* Summary functions include mean, std etc.

```python
df.groupby('kind').mean()
```

### Exercise

Are the claim relativities correct?

### Solution

```python
g = df.groupby('kind').mean()

display(g / g.loc['C'])

print('\n\nBetter\n')
g.loss / g.loc['C', 'loss']
```

### Grouping and Aggregating

* Very flexible processing for applying different functions

```python
stat_fns =  [np.size, np.mean, np.max]

df.groupby('kind').agg(stat_fns)
```

### Flexible Application of Aggregation Functions

```python
df.groupby('kind').agg({'loss': stat_fns, 'log_loss': stat_fns[1:] } )
```

###  Grouping by Two Variables

* First make the open claims more interesting

* Only apply to the loss variable

```python

df.loc[df.open==1, 'loss'] *= 0.9

g = df.groupby(['kind', 'open'])['loss'].agg([np.size, np.mean, np.max, np.std])

g
```

### What is the Group By Object?

* Pull off name and group variables separately

```python
for g, x in df.groupby(['kind', 'open']):
    display(g)
    display(x.head())
```

### Hence We Can Play Games Like

```python
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

f, axs = plt.subplots(2, 5, sharex=True, sharey=False, constrained_layout=True,
    figsize=(8, 3))
axi = iter(axs.flat)

for g, x in df.groupby(['open', 'kind']):
    ax = x.log_loss.hist(bins=20, ax=next(axi))
    ax.set(title=f"{g[1]}: {'Open' if g[0] else 'Closed'}")
```

### New Index and Data Transformation

* Go back to our `g` double grouped data frame

* Usual tidyverse spread (unstack) and gather (stack)

* Stack / unstack from shelving and unshelving books

```python
g = df.groupby(['kind', 'open'])['loss'].agg([np.size, np.mean, np.max, np.std])

display(g)

g.unstack(1)
```

### Stack Different Dimension

* `df.T` for transpose also available

```python
g.unstack(1).stack(0)
```

### Access the Indices

* Note the names for the levels

* Row index is an example of a `MultiIndex`

```python
print(g.columns)

g.index
```

### Exercise

Determine the maximum and minimum claim size by kind and open/closed indicator. Display by kind as rows.

### Solution

```python
df.groupby(['kind', 'open'])['loss'].agg([np.min, np.max]).unstack(1)
```

### Stylin'

```python
df.groupby(['kind', 'open'])['loss'].agg([np.min, np.max]).unstack(1).\
    style.format('{:,.1f}')
```

### More Stylin'

```python
df.groupby(['kind', 'open'])['loss'].agg([np.min, np.max]).\
    unstack(1).style.format('{:,.1f}').\
    background_gradient(subset=[('amax', 0)], cmap='viridis_r').\
    bar(color='#FFA07A', vmin=0, subset=[('amin', 0)], align='zero').\
    set_caption('An Over-Produced DataFrame')
```


## Module 5:  The CAS Loss Reserve Database

* Read CAS loss reserve database (extract)

* Automatically read csv file from URL

* Add some helpful columns

* Summarize

```python
df = pd.read_csv(r'http://www.mynl.com/RPM/masterdata.csv')
df['LR'] = df.UltIncLoss / df['EarnedPrem']
df.loc[:, 'PdLR'] = df.PaidLoss / df.loc[:, 'EarnedPrem']
 # some company names for future use
sfm = 'State Farm Mut Grp'
amg = 'American Modern Ins Grp Inc'
eix = 'Erie Ins Exchange Grp'
fmg = 'Federated Mut Grp'
wbi = 'West Bend Mut Ins Grp'
vnl = 'Vanliner Ins Co'
df.head().T
```

### What Does the DataFrame Contain?

* 10 years development for 10 accident years 1988-97

* Six lines of business

* Variety of companies

```python
print(df.columns)
print('\n\n')
for c in ['AY', 'DY', 'Lag', 'Line']:
    print(c, df[c].unique(), '\n')

for c in ['AY', 'DY', 'Lag', 'Line']:
    print(c, df[c].value_counts(), '\n')
```

### Summarizing The Data

* If **not** analyzing triangles need `Lag==10` subset to avoid double counting!

* Let's give more meaningful index

```python
dfl = df.query(' Lag == 10 ').copy()

dfl = dfl.set_index(['GRName', 'AY', 'Lag', 'Line'], drop=True)

dfl = dfl.drop('GRCode', axis=1)

dfl.head()
```

### Accessing Chunks

* `sfm` defined earlier to be State Farm Mut Grp

```python
dfl.xs([sfm, 'Comm Auto'], axis=0, level=[0,3])
```

### Group by with MultiIndex

```python
dfl.groupby(level=[1,3])[['UltIncLoss', 'EarnedPrem']].sum().unstack(1)
```

### Exercise

* Compute weighted average ultimate loss ratio by line by year

### Solution

```python
s = dfl.groupby(level=[1,3])[['UltIncLoss', 'EarnedPrem']].sum()
s['LR'] = s.UltIncLoss / s.EarnedPrem
s = s['LR'].unstack(1)
s.style.format('{:.1%}')
```

## Module 6: Make A Triangle!

* Standard `pivot_table` functionality

* Extol virtues of zero-based arrays, `lag` starts at 0

```python
bigCos =[sfm, amg, eix, fmg, wbi, vnl]
df['Lag'] -= 1
bit = df.query(f' GRName in @bigCos ').\
    pivot_table(index=['GRName', 'Line', 'AY'], columns='Lag', values='PaidLoss')
bit.tail(20)
```

### Link Ratios

* Use integer indexing...

```python
trg = bit.xs((vnl, 'Comm Auto'))
display(trg)
link = trg.iloc[:, 1:] / trg.iloc[:, :-1]
link
```

* Index-awareness is *usually* helpful!

### Link Ratios...Correctly

* `to_numpy()` or `values` converts into an array, drops index information

* Pick up column index from denominator---retains zero base

```python
link = trg.iloc[:, 1:].to_numpy() / trg.iloc[:, :-1]
link
```

### Trim-Up to an Historical Triangle

* Compute all the triangles at once...

* Drop 1997 year

```python
bit = df.query(f' GRName in @bigCos and Lag + AY <= 1997 ').pivot_table(index=['GRName', 'Line', 'AY'], columns='Lag', values='PaidLoss')
link = bit.iloc[:, 1:].to_numpy() / bit.iloc[:, :-1]
link.drop(1997, axis=0, level=2).head(19)
link.tail(20)
```

## THE END
