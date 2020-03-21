
### 목차

- Data exploration
- Missing Data.
- Imbalanced problem.


- Plots
    - Distribution plots
    - Count plots
    - Unique values
    - Groups


----

### 참조:

- https://www.kaggle.com/artgor/eda-and-models/data
- https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb
- https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda
- https://www.kaggle.com/jesucristo/fraud-complete-eda
<br>


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import catboost
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
%matplotlib inline

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)

import warnings
warnings.filterwarnings("ignore")

import gc
gc.enable()

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import os
print(os.listdir("../input"))

print ("Ready!")
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        



        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


    ['test_identity.csv', 'train_transaction.csv', 'train_identity.csv', 'sample_submission.csv', 'test_transaction.csv']
    Ready!
    

# Data


Data는 **identity** 와 **transaction**으로 나뉘어 있으며,```TransactionID```으로 join이 가능합니다.

> 모든 transaction data가 identity data와 연결되지는 않습니다.

**Categorical Features - Transaction**

- ProductCD
- emaildomain
- card1 - card6
- addr1, addr2
- P_emaildomain
- R_emaildomain
- M1 - M9

**Categorical Features - Identity**

- DeviceType
- DeviceInfo
- id_12 - id_38

**TransactionDT는 지정된 시간으로부터의 델타값입니다.(실제 시간단위가 아닙니다.)**

**Files**

- train_{transaction, identity}.csv - the training set
- test_{transaction, identity}.csv - the test set (**이 파일의 isFraud값을 예측해야합니다.**)
- sample_submission.csv - a sample submission file in the correct format

**Interactive Plots Utils**
> from https://www.kaggle.com/kabure/baseline-fraud-detection-eda-interactive-views (more about Interactive plots there)


```python
# functions from: https://www.kaggle.com/kabure/baseline-fraud-detection-eda-interactive-views

def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def plot_distribution(df, var_select=None, title=None, bins=1.0): 
    # Calculate the correlation coefficient between the new variable and the target
    tmp_fraud = df[df['isFraud'] == 1]
    tmp_no_fraud = df[df['isFraud'] == 0]    
    corr = df['isFraud'].corr(df[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_fraud[var_select].dropna()
    tmp2 = tmp_no_fraud[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Fraud', 'No Fraud']
    colors = ['seagreen','indianred', ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )
    
    fig['layout'].update(title = title+' '+'(corr target ='+ str(corr)+')')

    iplot(fig, filename = 'Density plot')
    
def plot_dist_churn(df, col, binary=None):
    tmp_churn = df[df[binary] == 1]
    tmp_no_churn = df[df[binary] == 0]
    tmp_attr = round(tmp_churn[col].value_counts().sort_index() / df[col].value_counts().sort_index(),2)*100
    print(f'Distribution of {col}: ')
    trace1 = go.Bar(
        x=tmp_churn[col].value_counts().sort_index().index,
        y=tmp_churn[col].value_counts().sort_index().values, 
        name='Fraud',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col].value_counts().sort_index().index,
        y=tmp_no_churn[col].value_counts().sort_index().values,
        name='No Fraud', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2', 
        name='% Fraud', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )
    
    layout = dict(title =  f'Distribution of {str(col)} feature by %Fraud',
              xaxis=dict(type='category'), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [0, 15], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= 'Percentual Fraud Transactions'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    iplot(fig)
```

**Load data**


```python
print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
```

    # File sizes
    test_identity.csv             25.8MB
    train_transaction.csv         683.35MB
    train_identity.csv            26.53MB
    sample_submission.csv         6.08MB
    test_transaction.csv          613.19MB
    


```python
%%time
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')
print ("Data is loaded!")
```

    Data is loaded!
    CPU times: user 33.8 s, sys: 4.85 s, total: 38.7 s
    Wall time: 39 s
    


```python
print('train_transaction shape is {}'.format(train_transaction.shape))
print('test_transaction shape is {}'.format(test_transaction.shape))
print('train_identity shape is {}'.format(train_identity.shape))
print('test_identity shape is {}'.format(test_identity.shape))
```

    train_transaction shape is (590540, 393)
    test_transaction shape is (506691, 392)
    train_identity shape is (144233, 40)
    test_identity shape is (141907, 40)
    

**train_trainsaction**


```python
train_transaction.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>addr1</th>
      <th>addr2</th>
      <th>dist1</th>
      <th>dist2</th>
      <th>P_emaildomain</th>
      <th>R_emaildomain</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>...</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>V322</th>
      <th>V323</th>
      <th>V324</th>
      <th>V325</th>
      <th>V326</th>
      <th>V327</th>
      <th>V328</th>
      <th>V329</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987000</th>
      <td>0</td>
      <td>86400</td>
      <td>68.5</td>
      <td>W</td>
      <td>13926</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>discover</td>
      <td>142.0</td>
      <td>credit</td>
      <td>315.0</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987001</th>
      <td>0</td>
      <td>86401</td>
      <td>29.0</td>
      <td>W</td>
      <td>2755</td>
      <td>404.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>325.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987002</th>
      <td>0</td>
      <td>86469</td>
      <td>59.0</td>
      <td>W</td>
      <td>4663</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>debit</td>
      <td>330.0</td>
      <td>87.0</td>
      <td>287.0</td>
      <td>NaN</td>
      <td>outlook.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987003</th>
      <td>0</td>
      <td>86499</td>
      <td>50.0</td>
      <td>W</td>
      <td>18132</td>
      <td>567.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>117.0</td>
      <td>debit</td>
      <td>476.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yahoo.com</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>112.0</td>
      <td>112.0</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.0</td>
      <td>1758.0</td>
      <td>925.0</td>
      <td>0.0</td>
      <td>354.0</td>
      <td>0.0</td>
      <td>135.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>1404.0</td>
      <td>790.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987004</th>
      <td>0</td>
      <td>86506</td>
      <td>50.0</td>
      <td>H</td>
      <td>4497</td>
      <td>514.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>420.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



**train_identity**


```python
train_identity.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
      <th>id_12</th>
      <th>id_13</th>
      <th>id_14</th>
      <th>id_15</th>
      <th>id_16</th>
      <th>id_17</th>
      <th>id_18</th>
      <th>id_19</th>
      <th>id_20</th>
      <th>id_21</th>
      <th>id_22</th>
      <th>id_23</th>
      <th>id_24</th>
      <th>id_25</th>
      <th>id_26</th>
      <th>id_27</th>
      <th>id_28</th>
      <th>id_29</th>
      <th>id_30</th>
      <th>id_31</th>
      <th>id_32</th>
      <th>id_33</th>
      <th>id_34</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987004</th>
      <td>0.0</td>
      <td>70787.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>-480.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>542.0</td>
      <td>144.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>Android 7.0</td>
      <td>samsung browser 6.2</td>
      <td>32.0</td>
      <td>2220x1080</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>mobile</td>
      <td>SAMSUNG SM-G892A Build/NRD90M</td>
    </tr>
    <tr>
      <th>2987008</th>
      <td>-5.0</td>
      <td>98945.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>49.0</td>
      <td>-300.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>621.0</td>
      <td>500.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>iOS 11.1.2</td>
      <td>mobile safari 11.0</td>
      <td>32.0</td>
      <td>1334x750</td>
      <td>match_status:1</td>
      <td>T</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>mobile</td>
      <td>iOS Device</td>
    </tr>
    <tr>
      <th>2987010</th>
      <td>-5.0</td>
      <td>191631.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>121.0</td>
      <td>NaN</td>
      <td>410.0</td>
      <td>142.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>NaN</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>Windows</td>
    </tr>
    <tr>
      <th>2987011</th>
      <td>-5.0</td>
      <td>221832.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>225.0</td>
      <td>NaN</td>
      <td>176.0</td>
      <td>507.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987016</th>
      <td>0.0</td>
      <td>7460.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>-300.0</td>
      <td>Found</td>
      <td>Found</td>
      <td>166.0</td>
      <td>15.0</td>
      <td>529.0</td>
      <td>575.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>Mac OS X 10_11_6</td>
      <td>chrome 62.0</td>
      <td>24.0</td>
      <td>1280x800</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>MacOS</td>
    </tr>
  </tbody>
</table>
</div>



data를 확인해보면 **NaN** data를 비롯한 여러 알수없는 feature들이 보입니다.

- ``` C1, C2 ... D1, V300, V339 ... ``` 
- ``` id_01 ... id_38``` 

column들의 이름만으로는 어떤 column인지 잘 모르겠습니다.

### 첫번째 관문: NaN

**train_transaction**


```python
missing_values_count = train_transaction.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_transaction.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)
```

    isFraud              0
    TransactionDT        0
    TransactionAmt       0
    ProductCD            0
    card1                0
    card2             8933
    card3             1565
    card4             1577
    card5             4259
    card6             1571
    dtype: int64
    % of missing data =  41.17794374769424
    

**train_identity**


```python
missing_values_count = train_identity.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_identity.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)
```

    id_01         0
    id_02      3361
    id_03     77909
    id_04     77909
    id_05      7368
    id_06      7368
    id_07    139078
    id_08    139078
    id_09     69307
    id_10     69307
    dtype: int64
    % of missing data =  36.47062392101669
    


```python
del missing_values_count, total_cells, total_missing
```

### 두번째 관문 ...

data를 확인해보면 매우 **불균형(imbalanced)** 합니다. 대부분의 transaction이 non-fraud입니다. 이 데이터를 그대로 가져다 모델을 만들면 큰 오차가 생기거나 overfit이 될 가능성이 높아 보입니다. 왜냐하면 대부분의 데이터가 non-fraud라고 "가정"하기 때문입니다. 저희는 모델이 "가정"하기보다는 "탐지"하기를 원했습니다.


```python
x = train_transaction['isFraud'].value_counts().index
y = train_transaction['isFraud'].value_counts().values

trace2 = go.Bar(
     x=x ,
     y=y,
     marker=dict(
         color=y,
         colorscale = 'Viridis',
         reversescale = True
     ),
     name="Imbalance",    
 )
layout = dict(
     title="Data imbalance - isFraud",
     #width = 900, height = 500,
     xaxis=go.layout.XAxis(
     automargin=True),
     yaxis=dict(
         showgrid=False,
         showline=False,
         showticklabels=True,
 #         domain=[0, 0.85],
     ), 
)
fig1 = go.Figure(data=[trace2], layout=layout)
iplot(fig1)

```


<div>
        
        
            <div id="db760452-3813-4dc4-bd3a-e47bc1a7d40e" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("db760452-3813-4dc4-bd3a-e47bc1a7d40e")) {
                    Plotly.newPlot(
                        'db760452-3813-4dc4-bd3a-e47bc1a7d40e',
                        [{"marker": {"color": [569877, 20663], "colorscale": "Viridis", "reversescale": true}, "name": "Imbalance", "type": "bar", "uid": "291af8e8-1483-4d31-b960-4ab497d242a7", "x": [0, 1], "y": [569877, 20663]}],
                        {"title": {"text": "Data imbalance - isFraud"}, "xaxis": {"automargin": true}, "yaxis": {"showgrid": false, "showline": false, "showticklabels": true}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('db760452-3813-4dc4-bd3a-e47bc1a7d40e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
del x,y
gc.collect()
```




    6670



# Time vs FE
**TransactionDT는 지정된 시간으로부터의 델타값입니다.(실제 시간단위가 아닙니다.)**

**[The timespan of the dataset is 1 year ?
](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100071#latest-577632) by Suchith**

```
Train: min = 86400, max = 15811131
Test: min = 18403224, max = 34214345
```

train.min()과 test.max()의 차이는 ```x = 34214345 - 86400 = 34127945``` 입니다. 하지만 이게 어떤 시간 단위(시,분,초)인지는 모릅니다.


만약 이게 '초'단위라면?  ```x/(3600*24*365) = 1.0821``` 년이 됩니다. 약 1년정도 되는걸 보니 합리적으로 보입니다.

그래서 만약 **transactionDT**가 **seconds**단위라면 아래와 같게 됩니다.

```
Time span of the total dataset is 394.9993634259259 days
Time span of Train dataset is  181.99920138888888 days
Time span of Test dataset is  182.99908564814814 days
The gap between train and test is 30.00107638888889 days
```


![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2370491%2Fc9bf5af5e902595b737df5470adc193b%2Fdownload-1.png?generation=1563312982845419&alt=media)

**source: [FChmiel](https://www.kaggle.com/fchmiel)**
<br>


```python
# Here we confirm that all of the transactions in `train_identity`
print(np.sum(train_transaction.index.isin(train_identity.index.unique())))
print(np.sum(test_transaction.index.isin(test_identity.index.unique())))
```

    144233
    141907
    

train data의 TransactionID 중 ```24.4%```(144233 / 590540)가 train_identity와 연결되어있습니다.

test data의 TransactionID 중 ```28.0%``` (144233 / 590540)가 train_identity와 연결되어있습니다.


```python
train_transaction['TransactionDT'].head()
```




    TransactionID
    2987000    86400
    2987001    86401
    2987002    86469
    2987003    86499
    2987004    86506
    Name: TransactionDT, dtype: int64




```python
train_transaction['TransactionDT'].shape[0] , train_transaction['TransactionDT'].nunique()
```




    (590540, 573349)



**TransactionDT** 가 시간단위는 아니지만, 시간을 계산할순 있게 해줍니다.


```python
train_transaction['TransactionDT'].value_counts().head(10)
```




    9474817     8
    11576951    5
    7236588     5
    4397066     5
    8468062     4
    8020359     4
    7337054     4
    5334992     4
    1971839     4
    1962054     4
    Name: TransactionDT, dtype: int64




```python
fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction['TransactionDT'].values

sns.distplot(time_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of TransactionDT', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionDT', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

plt.show()
```


![png](EDA_files/EDA_30_0.png)



```python
fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction.loc[train_transaction['isFraud'] == 1]['TransactionDT'].values

sns.distplot(np.log(time_val), ax=ax[0], color='r')
ax[0].set_title('Distribution of LOG TransactionDT, isFraud=1', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

time_val = train_transaction.loc[train_transaction['isFraud'] == 0]['TransactionDT'].values

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionDT, isFraud=0', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])


plt.show()
```


![png](EDA_files/EDA_31_0.png)



```python
train_transaction['TransactionDT'].plot(kind='hist',
                                        figsize=(15, 5),
                                        label='train',
                                        bins=50,
                                        title='Train vs Test TransactionDT distribution')
test_transaction['TransactionDT'].plot(kind='hist',
                                       label='test',
                                       bins=50)
plt.legend()
plt.show()
```


![png](EDA_files/EDA_32_0.png)


위 그림에 보이듯이 두 data set의 timestamp가 겹치지는 않습니다. 그러니 time-base validation도 가능할 것으로 보이네요.

참조: https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda.

그리고 train와 test사이에 **30일**의 gap이 있는 걸 확인할 수 있습니다.



```python
train_transaction.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>addr1</th>
      <th>addr2</th>
      <th>dist1</th>
      <th>dist2</th>
      <th>P_emaildomain</th>
      <th>R_emaildomain</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>...</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>V322</th>
      <th>V323</th>
      <th>V324</th>
      <th>V325</th>
      <th>V326</th>
      <th>V327</th>
      <th>V328</th>
      <th>V329</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987000</th>
      <td>0</td>
      <td>86400</td>
      <td>68.5</td>
      <td>W</td>
      <td>13926</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>discover</td>
      <td>142.0</td>
      <td>credit</td>
      <td>315.0</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987001</th>
      <td>0</td>
      <td>86401</td>
      <td>29.0</td>
      <td>W</td>
      <td>2755</td>
      <td>404.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>325.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987002</th>
      <td>0</td>
      <td>86469</td>
      <td>59.0</td>
      <td>W</td>
      <td>4663</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>debit</td>
      <td>330.0</td>
      <td>87.0</td>
      <td>287.0</td>
      <td>NaN</td>
      <td>outlook.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987003</th>
      <td>0</td>
      <td>86499</td>
      <td>50.0</td>
      <td>W</td>
      <td>18132</td>
      <td>567.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>117.0</td>
      <td>debit</td>
      <td>476.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yahoo.com</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>112.0</td>
      <td>112.0</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>50.0</td>
      <td>1758.0</td>
      <td>925.0</td>
      <td>0.0</td>
      <td>354.0</td>
      <td>0.0</td>
      <td>135.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>1404.0</td>
      <td>790.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987004</th>
      <td>0</td>
      <td>86506</td>
      <td>50.0</td>
      <td>H</td>
      <td>4497</td>
      <td>514.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>420.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



진행와중에 흥미로운 사실도 알았습니다. [Plotting features over time shows something.... interesting
](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100167#latest-577688)

C와 D사이에 특이한 상관관계가 있었습니다.

### isFraud vs time


```python
i = 'isFraud'
cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
train_transaction.loc[train_transaction['isFraud'] == 0].set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3), label="isFraud=0")
train_transaction.loc[train_transaction['isFraud'] == 1].set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3), label="isFraud=1")
#test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
plt.legend()
plt.show()
```


![png](EDA_files/EDA_37_0.png)


### C features: C1, C2 ... C14


```python
c_features = list(train_transaction.columns[16:30])
for i in c_features:
    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    plt.show()
```


![png](EDA_files/EDA_39_0.png)



![png](EDA_files/EDA_39_1.png)



![png](EDA_files/EDA_39_2.png)



![png](EDA_files/EDA_39_3.png)



![png](EDA_files/EDA_39_4.png)



![png](EDA_files/EDA_39_5.png)



![png](EDA_files/EDA_39_6.png)



![png](EDA_files/EDA_39_7.png)



![png](EDA_files/EDA_39_8.png)



![png](EDA_files/EDA_39_9.png)



![png](EDA_files/EDA_39_10.png)



![png](EDA_files/EDA_39_11.png)



![png](EDA_files/EDA_39_12.png)



![png](EDA_files/EDA_39_13.png)



```python
c_features = list(train_transaction.columns[16:30])
for i in c_features:
    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    plt.show()
```


![png](EDA_files/EDA_40_0.png)



![png](EDA_files/EDA_40_1.png)



![png](EDA_files/EDA_40_2.png)



![png](EDA_files/EDA_40_3.png)



![png](EDA_files/EDA_40_4.png)



![png](EDA_files/EDA_40_5.png)



![png](EDA_files/EDA_40_6.png)



![png](EDA_files/EDA_40_7.png)



![png](EDA_files/EDA_40_8.png)



![png](EDA_files/EDA_40_9.png)



![png](EDA_files/EDA_40_10.png)



![png](EDA_files/EDA_40_11.png)



![png](EDA_files/EDA_40_12.png)



![png](EDA_files/EDA_40_13.png)



```python
del c_features
gc.collect()
```




    47186



### D features: D1 ... D15


```python
d_features = list(train_transaction.columns[30:45])

for i in d_features:
    cor = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i])[0,1]
    train_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].plot(style='.', title=i+" corr= "+str(round(cor,3)), figsize=(15, 3))
    plt.show()
```


![png](EDA_files/EDA_43_0.png)



![png](EDA_files/EDA_43_1.png)



![png](EDA_files/EDA_43_2.png)



![png](EDA_files/EDA_43_3.png)



![png](EDA_files/EDA_43_4.png)



![png](EDA_files/EDA_43_5.png)



![png](EDA_files/EDA_43_6.png)



![png](EDA_files/EDA_43_7.png)



![png](EDA_files/EDA_43_8.png)



![png](EDA_files/EDA_43_9.png)



![png](EDA_files/EDA_43_10.png)



![png](EDA_files/EDA_43_11.png)



![png](EDA_files/EDA_43_12.png)



![png](EDA_files/EDA_43_13.png)



![png](EDA_files/EDA_43_14.png)


문제는 ```D``` feature에 가장 많은 NaN data가 있다는 것입니다.


```python
train_transaction[d_features].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>D15</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987000</th>
      <td>14.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2987001</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2987002</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>315.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>315.0</td>
    </tr>
    <tr>
      <th>2987003</th>
      <td>112.0</td>
      <td>112.0</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>2987004</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Click output to see the number of missing values in each column
missing_values_count = train_transaction[d_features].isnull().sum()
missing_values_count
```




    D1       1269
    D2     280797
    D3     262878
    D4     168922
    D5     309841
    D6     517353
    D7     551623
    D8     515614
    D9     515614
    D10     76022
    D11    279287
    D12    525823
    D13    528588
    D14    528353
    D15     89113
    dtype: int64




```python
# how many total missing values do we have?
total_cells = np.product(train_transaction[d_features].shape)
total_missing = missing_values_count.sum()
# percent of data that is missing
(total_missing/total_cells) * 100
```




    58.15126268612908



D feature에는 약 58%의 missing value가 존재했습니다.


```python
for i in d_features:
    cor_tr = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i].fillna(-1))[0,1]
    cor_te = np.corrcoef(test_transaction['TransactionDT'], test_transaction[i].fillna(-1))[0,1]
    train_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
    test_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+"  || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
    plt.show()
```


![png](EDA_files/EDA_49_0.png)



![png](EDA_files/EDA_49_1.png)



![png](EDA_files/EDA_49_2.png)



![png](EDA_files/EDA_49_3.png)



![png](EDA_files/EDA_49_4.png)



![png](EDA_files/EDA_49_5.png)



![png](EDA_files/EDA_49_6.png)



![png](EDA_files/EDA_49_7.png)



![png](EDA_files/EDA_49_8.png)



![png](EDA_files/EDA_49_9.png)



![png](EDA_files/EDA_49_10.png)



![png](EDA_files/EDA_49_11.png)



![png](EDA_files/EDA_49_12.png)



![png](EDA_files/EDA_49_13.png)



![png](EDA_files/EDA_49_14.png)



```python
del d_features, cor
gc.collect()
```




    17091



### M features: M1 .. M9


```python
m_features = list(train_transaction.columns[45:54])
train_transaction[m_features].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>M1</th>
      <th>M2</th>
      <th>M3</th>
      <th>M4</th>
      <th>M5</th>
      <th>M6</th>
      <th>M7</th>
      <th>M8</th>
      <th>M9</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987000</th>
      <td>T</td>
      <td>T</td>
      <td>T</td>
      <td>M2</td>
      <td>F</td>
      <td>T</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987001</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M0</td>
      <td>T</td>
      <td>T</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987002</th>
      <td>T</td>
      <td>T</td>
      <td>T</td>
      <td>M0</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2987003</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M0</td>
      <td>T</td>
      <td>F</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987004</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
del m_features
gc.collect()
```




    11



## V150


```python
i = "V150"
cor_tr = np.corrcoef(train_transaction['TransactionDT'], train_transaction[i].fillna(-1))[0,1]
cor_te = np.corrcoef(test_transaction['TransactionDT'], test_transaction[i].fillna(-1))[0,1]
train_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+" || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
test_transaction.set_index('TransactionDT')[i].fillna(-1).plot(style='.', title=i+" corr_tr= "+str(round(cor_tr,3))+"  || corr_te= "+str(round(cor_te,3)), figsize=(15, 3))
plt.show()
```


![png](EDA_files/EDA_55_0.png)


<br>
# Groups

```.head(20)```을 제거한 후 확인해봅시다.


```python
train_transaction.loc[:,train_transaction.columns[train_transaction.columns.str.startswith('V')]].isnull().sum().head(20)
```




    V1     279287
    V2     279287
    V3     279287
    V4     279287
    V5     279287
    V6     279287
    V7     279287
    V8     279287
    V9     279287
    V10    279287
    V11    279287
    V12     76073
    V13     76073
    V14     76073
    V15     76073
    V16     76073
    V17     76073
    V18     76073
    V19     76073
    V20     76073
    dtype: int64



<br>
# TransactionAmt


```python
fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction['TransactionAmt'].values

sns.distplot(time_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of TransactionAmt', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionAmt', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

plt.show()
```


![png](EDA_files/EDA_60_0.png)



```python
fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction.loc[train_transaction['isFraud'] == 1]['TransactionAmt'].values

sns.distplot(np.log(time_val), ax=ax[0], color='r')
ax[0].set_title('Distribution of LOG TransactionAmt, isFraud=1', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

time_val = train_transaction.loc[train_transaction['isFraud'] == 0]['TransactionAmt'].values

sns.distplot(np.log(time_val), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionAmt, isFraud=0', fontsize=14)
ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])


plt.show()
```


![png](EDA_files/EDA_61_0.png)



```python
del time_val
```


```python
tmp = train_transaction[['TransactionAmt', 'isFraud']][0:100000]
plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt', 'Transaction Amount Distribution', bins=10.0,)
del tmp
```


<div>
        
        
            <div id="4ea5b7da-f670-4bda-8f0d-112c9620abcf" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("4ea5b7da-f670-4bda-8f0d-112c9620abcf")) {
                    Plotly.newPlot(
                        '4ea5b7da-f670-4bda-8f0d-112c9620abcf',
                        {"barmode": "overlay", "hovermode": "closest", "legend": {"traceorder": "reversed"}, "title": {"text": "Transaction Amount Distribution (corr target =0.012)"}, "xaxis": {"anchor": "y2", "domain": [0.0, 1.0], "zeroline": false}, "yaxis": {"anchor": "free", "domain": [0.35, 1], "position": 0.0}, "yaxis2": {"anchor": "x", "domain": [0, 0.25], "dtick": 1, "showticklabels": false}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('4ea5b7da-f670-4bda-8f0d-112c9620abcf');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# Unique Values

### D Features


```python
plt.figure(figsize=(10, 7))
d_features = list(train_transaction.columns[30:45])
uniques = [len(train_transaction[col].unique()) for col in d_features]
sns.set(font_scale=1.2)
ax = sns.barplot(d_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_66_0.png)



```python
plt.figure(figsize=(10, 7))
d_features = list(test_transaction.columns[30:45])
uniques = [len(test_transaction[col].unique()) for col in d_features]
sns.set(font_scale=1.2)
ax = sns.barplot(d_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TEST')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_67_0.png)


### C features


```python
plt.figure(figsize=(10, 7))
c_features = list(train_transaction.columns[16:30])
uniques = [len(train_transaction[col].unique()) for col in c_features]
sns.set(font_scale=1.2)
ax = sns.barplot(c_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_69_0.png)



```python
plt.figure(figsize=(10, 7))
c_features = list(test_transaction.columns[16:30])
uniques = [len(test_transaction[col].unique()) for col in c_features]
sns.set(font_scale=1.2)
ax = sns.barplot(c_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TEST')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_70_0.png)


### V features


```python
plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[54:120])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_72_0.png)



```python
plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[120:170])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_73_0.png)



```python
plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[170:220])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_74_0.png)



```python
plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[220:270])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_75_0.png)



```python
plt.figure(figsize=(35, 8))
v_features = list(train_transaction.columns[270:320])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_76_0.png)



```python
plt.figure(figsize=(38, 8))
v_features = list(train_transaction.columns[320:390])
uniques = [len(train_transaction[col].unique()) for col in v_features]
sns.set(font_scale=1.2)
ax = sns.barplot(v_features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_77_0.png)


### id_code


```python
train_identity.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
      <th>id_12</th>
      <th>id_13</th>
      <th>id_14</th>
      <th>id_15</th>
      <th>id_16</th>
      <th>id_17</th>
      <th>id_18</th>
      <th>id_19</th>
      <th>id_20</th>
      <th>id_21</th>
      <th>id_22</th>
      <th>id_23</th>
      <th>id_24</th>
      <th>id_25</th>
      <th>id_26</th>
      <th>id_27</th>
      <th>id_28</th>
      <th>id_29</th>
      <th>id_30</th>
      <th>id_31</th>
      <th>id_32</th>
      <th>id_33</th>
      <th>id_34</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987004</th>
      <td>0.0</td>
      <td>70787.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>-480.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>542.0</td>
      <td>144.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>Android 7.0</td>
      <td>samsung browser 6.2</td>
      <td>32.0</td>
      <td>2220x1080</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>mobile</td>
      <td>SAMSUNG SM-G892A Build/NRD90M</td>
    </tr>
    <tr>
      <th>2987008</th>
      <td>-5.0</td>
      <td>98945.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>49.0</td>
      <td>-300.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>621.0</td>
      <td>500.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>iOS 11.1.2</td>
      <td>mobile safari 11.0</td>
      <td>32.0</td>
      <td>1334x750</td>
      <td>match_status:1</td>
      <td>T</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>mobile</td>
      <td>iOS Device</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(35, 8))
features = list(train_identity.columns[0:38])
uniques = [len(train_identity[col].unique()) for col in features]
sns.set(font_scale=1.2)
ax = sns.barplot(features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TRAIN')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_80_0.png)



```python
plt.figure(figsize=(35, 8))
features = list(test_identity.columns[0:38])
uniques = [len(test_identity[col].unique()) for col in features]
sns.set(font_scale=1.2)
ax = sns.barplot(features, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature TEST')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
```


![png](EDA_files/EDA_81_0.png)


<br>
# Categorical Features

- ProductCD
- emaildomain
- card1 - card6
- addr1, addr2
- P_emaildomain
- R_emaildomain
- M1 - M9
- DeviceType
- DeviceInfo
- id_12 - id_38


```python
train_transaction.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>ProductCD</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card4</th>
      <th>card5</th>
      <th>card6</th>
      <th>addr1</th>
      <th>addr2</th>
      <th>dist1</th>
      <th>dist2</th>
      <th>P_emaildomain</th>
      <th>R_emaildomain</th>
      <th>C1</th>
      <th>C2</th>
      <th>C3</th>
      <th>C4</th>
      <th>C5</th>
      <th>C6</th>
      <th>C7</th>
      <th>C8</th>
      <th>C9</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>C13</th>
      <th>C14</th>
      <th>D1</th>
      <th>D2</th>
      <th>D3</th>
      <th>D4</th>
      <th>D5</th>
      <th>D6</th>
      <th>D7</th>
      <th>D8</th>
      <th>D9</th>
      <th>D10</th>
      <th>...</th>
      <th>V300</th>
      <th>V301</th>
      <th>V302</th>
      <th>V303</th>
      <th>V304</th>
      <th>V305</th>
      <th>V306</th>
      <th>V307</th>
      <th>V308</th>
      <th>V309</th>
      <th>V310</th>
      <th>V311</th>
      <th>V312</th>
      <th>V313</th>
      <th>V314</th>
      <th>V315</th>
      <th>V316</th>
      <th>V317</th>
      <th>V318</th>
      <th>V319</th>
      <th>V320</th>
      <th>V321</th>
      <th>V322</th>
      <th>V323</th>
      <th>V324</th>
      <th>V325</th>
      <th>V326</th>
      <th>V327</th>
      <th>V328</th>
      <th>V329</th>
      <th>V330</th>
      <th>V331</th>
      <th>V332</th>
      <th>V333</th>
      <th>V334</th>
      <th>V335</th>
      <th>V336</th>
      <th>V337</th>
      <th>V338</th>
      <th>V339</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987000</th>
      <td>0</td>
      <td>86400</td>
      <td>68.5</td>
      <td>W</td>
      <td>13926</td>
      <td>NaN</td>
      <td>150.0</td>
      <td>discover</td>
      <td>142.0</td>
      <td>credit</td>
      <td>315.0</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987001</th>
      <td>0</td>
      <td>86401</td>
      <td>29.0</td>
      <td>W</td>
      <td>2755</td>
      <td>404.0</td>
      <td>150.0</td>
      <td>mastercard</td>
      <td>102.0</td>
      <td>credit</td>
      <td>325.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>gmail.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2987002</th>
      <td>0</td>
      <td>86469</td>
      <td>59.0</td>
      <td>W</td>
      <td>4663</td>
      <td>490.0</td>
      <td>150.0</td>
      <td>visa</td>
      <td>166.0</td>
      <td>debit</td>
      <td>330.0</td>
      <td>87.0</td>
      <td>287.0</td>
      <td>NaN</td>
      <td>outlook.com</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_identity.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_01</th>
      <th>id_02</th>
      <th>id_03</th>
      <th>id_04</th>
      <th>id_05</th>
      <th>id_06</th>
      <th>id_07</th>
      <th>id_08</th>
      <th>id_09</th>
      <th>id_10</th>
      <th>id_11</th>
      <th>id_12</th>
      <th>id_13</th>
      <th>id_14</th>
      <th>id_15</th>
      <th>id_16</th>
      <th>id_17</th>
      <th>id_18</th>
      <th>id_19</th>
      <th>id_20</th>
      <th>id_21</th>
      <th>id_22</th>
      <th>id_23</th>
      <th>id_24</th>
      <th>id_25</th>
      <th>id_26</th>
      <th>id_27</th>
      <th>id_28</th>
      <th>id_29</th>
      <th>id_30</th>
      <th>id_31</th>
      <th>id_32</th>
      <th>id_33</th>
      <th>id_34</th>
      <th>id_35</th>
      <th>id_36</th>
      <th>id_37</th>
      <th>id_38</th>
      <th>DeviceType</th>
      <th>DeviceInfo</th>
    </tr>
    <tr>
      <th>TransactionID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2987004</th>
      <td>0.0</td>
      <td>70787.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>NaN</td>
      <td>-480.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>542.0</td>
      <td>144.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>Android 7.0</td>
      <td>samsung browser 6.2</td>
      <td>32.0</td>
      <td>2220x1080</td>
      <td>match_status:2</td>
      <td>T</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>mobile</td>
      <td>SAMSUNG SM-G892A Build/NRD90M</td>
    </tr>
    <tr>
      <th>2987008</th>
      <td>-5.0</td>
      <td>98945.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>49.0</td>
      <td>-300.0</td>
      <td>New</td>
      <td>NotFound</td>
      <td>166.0</td>
      <td>NaN</td>
      <td>621.0</td>
      <td>500.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>New</td>
      <td>NotFound</td>
      <td>iOS 11.1.2</td>
      <td>mobile safari 11.0</td>
      <td>32.0</td>
      <td>1334x750</td>
      <td>match_status:1</td>
      <td>T</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>mobile</td>
      <td>iOS Device</td>
    </tr>
    <tr>
      <th>2987010</th>
      <td>-5.0</td>
      <td>191631.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>NotFound</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>121.0</td>
      <td>NaN</td>
      <td>410.0</td>
      <td>142.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Found</td>
      <td>Found</td>
      <td>NaN</td>
      <td>chrome 62.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F</td>
      <td>F</td>
      <td>T</td>
      <td>T</td>
      <td>desktop</td>
      <td>Windows</td>
    </tr>
  </tbody>
</table>
</div>



### ProductCD


```python
fig, ax = plt.subplots(1, 2, figsize=(20,5))

sns.countplot(x="ProductCD", ax=ax[0], hue = "isFraud", data=train_transaction)
ax[0].set_title('ProductCD train', fontsize=14)
sns.countplot(x="ProductCD", ax=ax[1], data=test_transaction)
ax[1].set_title('ProductCD test', fontsize=14)
plt.show()
```


![png](EDA_files/EDA_86_0.png)


### Device Type & Device Info


```python
ax = sns.countplot(x="DeviceType", data=train_identity)
ax.set_title('DeviceType', fontsize=14)
plt.show()
```


![png](EDA_files/EDA_88_0.png)


**Device information**


```python
print ("Unique Devices = ",train_identity['DeviceInfo'].nunique())
train_identity['DeviceInfo'].value_counts().head()
```

    Unique Devices =  1786
    




    Windows        47722
    iOS Device     19782
    MacOS          12573
    Trident/7.0     7440
    rv:11.0         1901
    Name: DeviceInfo, dtype: int64



### Card


```python
cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
for i in cards:
    print ("Unique ",i, " = ",train_transaction[i].nunique())
```

    Unique  card1  =  13553
    Unique  card2  =  500
    Unique  card3  =  114
    Unique  card4  =  4
    Unique  card5  =  119
    Unique  card6  =  4
    


```python
fig, ax = plt.subplots(1, 4, figsize=(25,5))

sns.countplot(x="card4", ax=ax[0], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[0].set_title('card4 isFraud=0', fontsize=14)
sns.countplot(x="card4", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('card4 isFraud=1', fontsize=14)
sns.countplot(x="card6", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('card6 isFraud=0', fontsize=14)
sns.countplot(x="card6", ax=ax[3], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[3].set_title('card6 isFraud=1', fontsize=14)
plt.show()
```


![png](EDA_files/EDA_93_0.png)



```python
cards = train_transaction.iloc[:,4:7].columns

plt.figure(figsize=(18,8*4))
gs = gridspec.GridSpec(8, 4)
for i, cn in enumerate(cards):
    ax = plt.subplot(gs[i])
    sns.distplot(train_transaction.loc[train_transaction['isFraud'] == 1][cn], bins=50)
    sns.distplot(train_transaction.loc[train_transaction['isFraud'] == 0][cn], bins=50)
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(cn))
plt.show()

```


![png](EDA_files/EDA_94_0.png)


위에서 보이듯이, ``` Card 1```은 Categorical feature지만 Continuous feature 마냥 '13553'의 unique value를 가지고 있습니다.

### Email Domain


```python
"emaildomain" in train_transaction.columns, "emaildomain" in train_identity.columns
```




    (False, False)




```python
fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="P_emaildomain", ax=ax[0], data=train_transaction)
ax[0].set_title('P_emaildomain', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('P_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('P_emaildomain isFraud = 0', fontsize=14)
plt.show()
```


![png](EDA_files/EDA_98_0.png)



```python
fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="R_emaildomain", ax=ax[0], data=train_transaction)
ax[0].set_title('R_emaildomain', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[1], data=train_transaction.loc[train_transaction['isFraud'] == 1])
ax[1].set_title('R_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[2], data=train_transaction.loc[train_transaction['isFraud'] == 0])
ax[2].set_title('R_emaildomain isFraud = 0', fontsize=14)
plt.show()
```


![png](EDA_files/EDA_99_0.png)


gmail의 비율이 압도적으로 높았습니다.