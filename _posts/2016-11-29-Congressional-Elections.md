---
layout: post
title: 
feature-img: "img/ted_cruz7.jpg" 
---
### Congressional Elections, Logistic Regression, and Feature Selection

In the spirit of our recent election season (hah), I've decided to do some exploration in using boosted trees to identify features that have the most predictive power. Using those selected features and some propreitary intuition, I'll then use a logistic model to predict whether a congressional district will be democrat or republican. 

For comparison's sake, I'll begin by specifying a logistic model with features that I pick without any empirical evidence to back up my decision.

The data describe all 435 districts in the 105th congress from 1997-1998. It contains demographic and employment data for each district as well as a variable indicating whether the district’srepresentative in the House was a Republican or not. Republican districts were coded as 1 while Democratic/independent districts were coded as 0. There are a total of 31 variables, as described below:

![Variable descriptions](/img/congress_description.png)


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from __future__ import division
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 18.5, 10.5
```

```python
congress = pd.read_stata("http://rlhick.people.wm.edu/econ407/data/congressional_105.dta" )
```

For the naive logit model, I will estimate $$repub = \beta_0 + \beta_1 per\_age65 + \beta_2 per\_black + \beta_3 per\_
bluecllr + \beta_4 city + \beta_5 mdnincm + \beta_6 per\_unemployed + \beta_7 union$$


where per\_ indicates that variable is a percentage of the district. I use percentages so that (presumably) the results are agnostic of the size of a district and can be generalized to any region in the US. 



```python
#making percentages
variables = ['unemplyd', 'age65', 'black', 'blucllr']

for v in variables:
    congress['per_' + v] = congress[[v, 'populatn']].apply(lambda row: (row[0] / row[1])*100, axis = 1)
```

Before I do anything, I check for null values and get a sense of the data


```python
congress.isnull().sum()
```




    state       0
    fipstate    0
    sc          0
    cd          0
    repub       1
    age65       0
    black       0
    blucllr     0
    city        0
    coast       0
    construc    0
    cvllbrfr    0
    enroll      0
    farmer      0
    finance     0
    forborn     0
    gvtwrkr     0
    intrland    0
    landsqmi    0
    mdnincm     0
    miltinst    0
    miltmajr    0
    miltpop     0
    nucplant    0
    popsqmi     0
    populatn    0
    rurlfarm    0
    transprt    0
    unemplyd    0
    union       0
    urban       0
    whlretl     0
    dtype: int64



After doing some research on Oklahoma's first congressional district, they had republican congressman Steve Largent from 1994 to 2002. I will code that row as republican.


```python
congress.repub.fillna(value = 1, inplace = True)
```

Lets see how balanced my data set is 


```python
sns.countplot(x = 'repub', data = congress, palette = 'muted');
```


![png](/img/Congressional%20Elections_files/Congressional%20Elections_10_0.png)



```python
print congress.repub.value_counts()
```

    1.0    228
    0.0    207
    Name: repub, dtype: int64


So our data is fairly balanced, although republicans have the slight majority. Lets check out the relationship between the blue collar workers and black population:


```python
sns.regplot(congress.per_black, congress.per_blucllr);
```


![png](/img/Congressional%20Elections_files/Congressional%20Elections_13_0.png)


There doesn't seem to be a definite relationship between the percentage of blue collar workers and percentage of black population. What about unemployment and the black population?


```python
sns.regplot(congress.per_black, congress.per_unemplyd);
```


![png](/img/Congressional%20Elections_files/Congressional%20Elections_15_0.png)


Those two variables are positively correlated.

Lets take a look at differences in race that elected a republican congressman:


```python
sns.boxplot(x = 'repub', y = 'per_black', data = congress );
sns.swarmplot(x = 'repub', y = 'per_black', data = congress, color = ".15" )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x119de4a10>




![png](/img/Congressional%20Elections_files/Congressional%20Elections_17_1.png)



```python
facetgrid = sns.FacetGrid(congress,hue='repub',size=6)
facetgrid.map(sns.kdeplot,'per_black',shade=True)
facetgrid.set(xlim=(0, congress.per_black.max()))
facetgrid.add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x119e327d0>




![png](/img/Congressional%20Elections_files/Congressional%20Elections_18_1.png)



```python
print "Districts that went Republican: \n",congress[congress.repub == 1].per_black.describe()

print "\nDistricts that went Democrat: \n",congress[congress.repub == 0].per_black.describe()
```

    Districts that went Republican: 
    count    228.000000
    mean       7.139820
    std        7.773183
    min        0.178235
    25%        2.148288
    50%        4.678620
    75%        8.204570
    max       40.724916
    Name: per_black, dtype: float64
    
    Districts that went Democrat: 
    count    207.000000
    mean      17.020763
    std       20.083805
    min        0.131355
    25%        2.776919
    50%        8.122120
    75%       21.369574
    max       73.948710
    Name: per_black, dtype: float64


The median value for districts that went Democrat is 4 % higher than those that went republican. In addition to the districts that went Republican are more closely centered around the mean and median, where as in the districts that went Democrat, there is a significant discrepancy between the mean and median. In fact, 25 % of the data falls between 8.12 % and 21.36 % for Democratic districts, but in Republican districts that number is 4.6 % and 8.2 %.

I expect that the percentage of the population that is black will have significant predicitive power in our model. 


Examine the correlation between our variables of interest:


```python
ind_vars = [ 'per_age65', 'per_black', 'per_blucllr', 'city','mdnincm', 'per_unemplyd', 'union']
congress[ind_vars].corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>per_age65</th>
      <th>per_black</th>
      <th>per_blucllr</th>
      <th>city</th>
      <th>mdnincm</th>
      <th>per_unemplyd</th>
      <th>union</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>per_age65</th>
      <td>1.000000</td>
      <td>-0.135644</td>
      <td>-0.003851</td>
      <td>-0.151725</td>
      <td>-0.176406</td>
      <td>-0.154773</td>
      <td>-0.016208</td>
    </tr>
    <tr>
      <th>per_black</th>
      <td>-0.135644</td>
      <td>1.000000</td>
      <td>0.055332</td>
      <td>0.232024</td>
      <td>-0.312837</td>
      <td>0.549604</td>
      <td>-0.135479</td>
    </tr>
    <tr>
      <th>per_blucllr</th>
      <td>-0.003851</td>
      <td>0.055332</td>
      <td>1.000000</td>
      <td>-0.198259</td>
      <td>-0.470878</td>
      <td>0.110561</td>
      <td>-0.067490</td>
    </tr>
    <tr>
      <th>city</th>
      <td>-0.151725</td>
      <td>0.232024</td>
      <td>-0.198259</td>
      <td>1.000000</td>
      <td>-0.002129</td>
      <td>0.299911</td>
      <td>0.089047</td>
    </tr>
    <tr>
      <th>mdnincm</th>
      <td>-0.176406</td>
      <td>-0.312837</td>
      <td>-0.470878</td>
      <td>-0.002129</td>
      <td>1.000000</td>
      <td>-0.505595</td>
      <td>0.251465</td>
    </tr>
    <tr>
      <th>per_unemplyd</th>
      <td>-0.154773</td>
      <td>0.549604</td>
      <td>0.110561</td>
      <td>0.299911</td>
      <td>-0.505595</td>
      <td>1.000000</td>
      <td>0.167665</td>
    </tr>
    <tr>
      <th>union</th>
      <td>-0.016208</td>
      <td>-0.135479</td>
      <td>-0.067490</td>
      <td>0.089047</td>
      <td>0.251465</td>
      <td>0.167665</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Nothing seriously outrageous to see here. Percent black and unemployment have the highest correlation at 54.9%, followed by city and unemployment at 29.9%. Something to keep in mind when we are doing feature selections.

## quick aside about MLE and Logit

I'm not going to go into a ton detail about logistic regression, because it has been done ad nauseam. If you would like more info about it, the [wiki](https://en.wikipedia.org/wiki/Logistic_regression) is a great place to start. 

I'd like to detail a bit about MLE or maximum likelihood estimation in the context of the logit model. Stats models uses maximum likelihood to estimate the coefficients. In order to use MLE, we assume that the errors in our model are independent and identically Logit distributed. The cumulative logit function is given by: 

$$ P(y = 1 | x_i)= \int_{-\infty}^{x_i \beta} f(t)dt = \frac{e^{x_i \beta}}{1 + e^{x_i \beta}} $$

The log-likelihood is is then:

$$ \Sigma_{i = 1}^{N} P(y = 1 | x_i \beta) \times (y_i) + (1 - P(y = 1 | x_i \beta)) \times (1 - y_i) $$

When I run a logitstic model in statsmodels, it estimates parameters $$b$$ such that the above sum is maximized. 

So now that we know what's going on behind the scenes, lets heckin do it :)



```python
dep_var = 'repub'

x_const = sm.add_constant(congress[ind_vars])
y = congress[dep_var]

#Logit and probit models

logit_results = Logit(y, x_const).fit()
logit_results.summary()
```

    Optimization terminated successfully.
             Current function value: 0.542084
             Iterations 7





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>repub</td>      <th>  No. Observations:  </th>  <td>   435</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   427</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     7</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sun, 15 Jan 2017</td> <th>  Pseudo R-squ.:     </th>  <td>0.2166</td>  
</tr>
<tr>
  <th>Time:</th>              <td>12:59:27</td>     <th>  Log-Likelihood:    </th> <td> -235.81</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -301.01</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>5.158e-25</td>
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>        <td>    9.5500</td> <td>    1.618</td> <td>    5.902</td> <td> 0.000</td> <td>    6.378    12.722</td>
</tr>
<tr>
  <th>per_age65</th>    <td>   -0.1199</td> <td>    0.037</td> <td>   -3.217</td> <td> 0.001</td> <td>   -0.193    -0.047</td>
</tr>
<tr>
  <th>per_black</th>    <td>   -0.0504</td> <td>    0.013</td> <td>   -3.956</td> <td> 0.000</td> <td>   -0.075    -0.025</td>
</tr>
<tr>
  <th>per_blucllr</th>  <td>   -0.0712</td> <td>    0.063</td> <td>   -1.135</td> <td> 0.256</td> <td>   -0.194     0.052</td>
</tr>
<tr>
  <th>city</th>         <td>   -0.6513</td> <td>    0.259</td> <td>   -2.519</td> <td> 0.012</td> <td>   -1.158    -0.145</td>
</tr>
<tr>
  <th>mdnincm</th>      <td>-5.843e-05</td> <td> 1.92e-05</td> <td>   -3.037</td> <td> 0.002</td> <td>-9.61e-05 -2.07e-05</td>
</tr>
<tr>
  <th>per_unemplyd</th> <td>   -1.4488</td> <td>    0.233</td> <td>   -6.210</td> <td> 0.000</td> <td>   -1.906    -0.992</td>
</tr>
<tr>
  <th>union</th>        <td>   -0.0280</td> <td>    0.016</td> <td>   -1.705</td> <td> 0.088</td> <td>   -0.060     0.004</td>
</tr>
</table>




```python
logit_results.get_margeff(dummy = True).summary()
```




<table class="simpletable">
<caption>Logit Marginal Effects</caption>
<tr>
  <th>Dep. Variable:</th>  <td>repub</td> 
</tr>
<tr>
  <th>Method:</th>         <td>dydx</td>  
</tr>
<tr>
  <th>At:</th>            <td>overall</td>
</tr>
</table>
<table class="simpletable">
<tr>
        <th></th>          <th>dy/dx</th>    <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>per_age65</th>    <td>   -0.0221</td> <td>    0.007</td> <td>   -3.363</td> <td> 0.001</td> <td>   -0.035    -0.009</td>
</tr>
<tr>
  <th>per_black</th>    <td>   -0.0093</td> <td>    0.002</td> <td>   -4.215</td> <td> 0.000</td> <td>   -0.014    -0.005</td>
</tr>
<tr>
  <th>per_blucllr</th>  <td>   -0.0122</td> <td>    0.010</td> <td>   -1.269</td> <td> 0.205</td> <td>   -0.031     0.007</td>
</tr>
<tr>
  <th>city</th>         <td>   -0.1201</td> <td>    0.046</td> <td>   -2.586</td> <td> 0.010</td> <td>   -0.211    -0.029</td>
</tr>
<tr>
  <th>mdnincm</th>      <td>-1.077e-05</td> <td>  3.4e-06</td> <td>   -3.163</td> <td> 0.002</td> <td>-1.74e-05  -4.1e-06</td>
</tr>
<tr>
  <th>per_unemplyd</th> <td>   -0.2670</td> <td>    0.035</td> <td>   -7.538</td> <td> 0.000</td> <td>   -0.336    -0.198</td>
</tr>
<tr>
  <th>union</th>        <td>   -0.0052</td> <td>    0.003</td> <td>   -1.725</td> <td> 0.085</td> <td>   -0.011     0.001</td>
</tr>
</table>



Before I look at how well the model is classifying, I'll interpret the marginal effects:

For each variable that is a percentage, the marginal effect is the amount that the probability drops for each percent increase in that variable. Those include per_age65, per_black, per_bluecllr, per_unemplyd, and union. The most notable of these is that for every percentage increase in unemployment, the probability that the district elects a republican congressman or woman  decreases by almost 27%!! Oh how the times have changed.

Another interesting result is that for every 10,000 \$ increase in median income, the probability of a republican goes down by about 10.7 %. This goes against conventional wisdom that says that richer folks prefer republican canidates. Again, I think that if region was controlled for, the effect may be different.

Something that I think would be interesting to explore is how religious preferences within each district affect republican chances. Anyway, on to the classification reports


```python
predictions = np.array([1 if x >= 0.5 else 0 for x in logit_results.predict()])

print classification_report(congress.repub, predictions)
print "Confusion matrix:\n"
print confusion_matrix(congress.repub, predictions)
```

                 precision    recall  f1-score   support
    
            0.0       0.74      0.67      0.70       207
            1.0       0.72      0.78      0.75       228
    
    avg / total       0.73      0.73      0.73       435
    
    Confusion matrix:
    
    [[139  68]
     [ 50 178]]


Okay, so the model isn't *awful*, although it could use some serious improvement. Democratic districts are only classified correctly (139 / (139 + 68)) = 67% of the time. Okay so that's not very good at all. The pseudo R2 is 0.2166, which is okay. A note about that:

**McFadden's (pseudo) R2** = (1 - (log likelihood full model / log likelihood only intercept)). The log likelihood of the intercept model is treated as a total sum of squares, and the log likelihood of the full model is treated as the sum of squared errors (like in approach 1). The ratio of the likelihoods suggests the level of improvement over the intercept model offered by the full model. A higher R2 suggests a better model that explains more of the variance.

We proceed boldy forward by diving in to **Feature Selection**

## Using a Boosted classifier to do Feature Selection

A cool feature of the xgboost library is that after it estimates all the trees, it calculates for each tree the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for. The performance measure may be the purity (Gini index) used to select the split points or another more specific error function. For more information, check out [this](http://stats.stackexchange.com/questions/162162/relative-variable-importance-for-boosting) stack overflow question.

I start with using all variables except those that are identifying information like state, FIPS code etc. I also make the rest of the population count variables in to percentages so that they are comparable to other districts.


```python
needs_convert = ['construc', 'cvllbrfr', 'enroll', 'farmer', 'finance', 'forborn', 'gvtwrkr', 'miltpop',
                 'rurlfarm', 'transprt', 'urban', 'whlretl']
for v in needs_convert:
    congress['per_' + v] = congress[[v, 'populatn']].apply(lambda row: (row[0] / row[1])*100, axis = 1)

all_ind_vars = ['per_age65', 'per_black', 'per_blucllr', 'city','coast', 'per_construc', 'per_cvllbrfr', 
                'per_enroll','per_farmer', 'per_finance', 'per_forborn', 'per_gvtwrkr', 'intrland', 'landsqmi', 
                'mdnincm', 'miltinst', 'miltmajr', 'per_miltpop', 'nucplant', 'popsqmi','populatn', 'per_rurlfarm', 
                'per_transprt', 'per_unemplyd', 'union', 'per_urban','per_whlretl']


```


```python
boosted = XGBClassifier(max_depth= 4, objective= 'binary:logistic')
boosted.fit(congress[all_ind_vars], congress.repub)
```




    XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=4,
           min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
           scale_pos_weight=1, seed=0, silent=True, subsample=1)




```python
xgb.plot_importance(boosted.booster());
```


![png](/img/Congressional%20Elections_files/Congressional%20Elections_31_0.png)


Some surprising results, some expected results.

**Unexpected:**

The percentage of the population that works construction is the most important feature... interesting. Personally, I would consider construction work to be blue collar work, however, they weren't counted as part of the population that constituted blue collar workers. Maybe instead of campaigning with the steel workers, congressmen should pander to construction unions (do those even exist?). The size of the district also plays a significant role. It will be interesting to see whether size contributes positively or negatively to the probability of a republican.

Government, transport and utility workers also contribute significantly. A common theme here is features that are most important are either directly related to government work or somehow often implicated with government. Indeed, whether a person is employed or not seriously colors their view of how the government should be run. 


**Expected**

I was right about unemployment and the black population playing a large role in elections! Sweet validation.

## New Model

The new model I will estimate is:

$$ repub = \beta_0 + \beta_1 per\_construc + \beta_2 per\_black + \beta_3 per\_unemplyd + \beta_4 per\_transprt + \beta_5 per\_gvtwrkr + \beta_6 per\_enroll + \beta_7 per\_age65 + \beta_8 mdnincm + \beta_9 populatn + \beta_{10} intrland + \beta_{11} city + \beta_{12} coast $$

I include all the top four variables that were selected for by the boosted trees. All of those features, with the exception of black, are indicators of employment. To control for age, race, and economic status of the district I include percentage black, over age 65, and median income. I include the amount of National Park land because that land is owned by the government, and provides employment and tourist attractions for citizens. City and coast are to control for urban populations and location respectively. Coast especially should account for the democratic "coastal elite". Enroll is to control for those voters whose children are enrolled in public schools, also a factor that would color their view of the government.

Lets let er rip


```python
new_vars = ['per_construc', 'per_black', 'per_unemplyd', 'per_transprt', 'per_gvtwrkr', 'per_enroll', 'per_age65',
            'mdnincm','populatn',  'intrland', 'city', 'coast' ]
new_x_const = sm.add_constant(congress[new_vars])

new_model = Logit(congress.repub, new_x_const).fit()
new_model.summary()
```

    Optimization terminated successfully.
             Current function value: 0.504768
             Iterations 7





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>repub</td>      <th>  No. Observations:  </th>  <td>   435</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   422</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>    12</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sun, 15 Jan 2017</td> <th>  Pseudo R-squ.:     </th>  <td>0.2705</td>  
</tr>
<tr>
  <th>Time:</th>              <td>12:59:28</td>     <th>  Log-Likelihood:    </th> <td> -219.57</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -301.01</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>1.362e-28</td>
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>        <td>   -2.1656</td> <td>    3.265</td> <td>   -0.663</td> <td> 0.507</td> <td>   -8.566     4.235</td>
</tr>
<tr>
  <th>per_construc</th> <td>    0.3077</td> <td>    0.210</td> <td>    1.466</td> <td> 0.143</td> <td>   -0.104     0.719</td>
</tr>
<tr>
  <th>per_black</th>    <td>   -0.0291</td> <td>    0.013</td> <td>   -2.304</td> <td> 0.021</td> <td>   -0.054    -0.004</td>
</tr>
<tr>
  <th>per_unemplyd</th> <td>   -1.7545</td> <td>    0.268</td> <td>   -6.558</td> <td> 0.000</td> <td>   -2.279    -1.230</td>
</tr>
<tr>
  <th>per_transprt</th> <td>   -0.1287</td> <td>    0.220</td> <td>   -0.585</td> <td> 0.559</td> <td>   -0.560     0.303</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>  <td>   -0.1081</td> <td>    0.064</td> <td>   -1.680</td> <td> 0.093</td> <td>   -0.234     0.018</td>
</tr>
<tr>
  <th>per_enroll</th>   <td>    0.2008</td> <td>    0.080</td> <td>    2.524</td> <td> 0.012</td> <td>    0.045     0.357</td>
</tr>
<tr>
  <th>per_age65</th>    <td>   -0.0035</td> <td>    0.055</td> <td>   -0.063</td> <td> 0.949</td> <td>   -0.111     0.104</td>
</tr>
<tr>
  <th>mdnincm</th>      <td>-1.363e-05</td> <td> 2.27e-05</td> <td>   -0.602</td> <td> 0.547</td> <td> -5.8e-05  3.08e-05</td>
</tr>
<tr>
  <th>populatn</th>     <td> 9.015e-06</td> <td> 3.18e-06</td> <td>    2.837</td> <td> 0.005</td> <td> 2.79e-06  1.52e-05</td>
</tr>
<tr>
  <th>intrland</th>     <td> 2.114e-08</td> <td> 1.15e-08</td> <td>    1.846</td> <td> 0.065</td> <td>-1.31e-09  4.36e-08</td>
</tr>
<tr>
  <th>city</th>         <td>   -0.4814</td> <td>    0.286</td> <td>   -1.685</td> <td> 0.092</td> <td>   -1.041     0.079</td>
</tr>
<tr>
  <th>coast</th>        <td>   -0.6496</td> <td>    0.267</td> <td>   -2.432</td> <td> 0.015</td> <td>   -1.173    -0.126</td>
</tr>
</table>




```python
new_predictions = np.array([1 if x >= 0.5 else 0 for x in new_model.predict()])

print classification_report(congress.repub, new_predictions)
print "Confusion matrix:\n"
print confusion_matrix(congress.repub, new_predictions)
```

                 precision    recall  f1-score   support
    
            0.0       0.76      0.68      0.72       207
            1.0       0.73      0.81      0.77       228
    
    avg / total       0.75      0.74      0.74       435
    
    Confusion matrix:
    
    [[140  67]
     [ 44 184]]


Our pseudo R2 improves significantly from 0.2166 to 0.2705. The precision and recall also improve slightly, although they are still not what I would like them to be. Lets look at the marginal effects: 


```python
new_model.get_margeff(dummy=True).summary()
```




<table class="simpletable">
<caption>Logit Marginal Effects</caption>
<tr>
  <th>Dep. Variable:</th>  <td>repub</td> 
</tr>
<tr>
  <th>Method:</th>         <td>dydx</td>  
</tr>
<tr>
  <th>At:</th>            <td>overall</td>
</tr>
</table>
<table class="simpletable">
<tr>
        <th></th>          <th>dy/dx</th>    <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>per_construc</th> <td>    0.0521</td> <td>    0.035</td> <td>    1.480</td> <td> 0.139</td> <td>   -0.017     0.121</td>
</tr>
<tr>
  <th>per_black</th>    <td>   -0.0049</td> <td>    0.002</td> <td>   -2.351</td> <td> 0.019</td> <td>   -0.009    -0.001</td>
</tr>
<tr>
  <th>per_unemplyd</th> <td>   -0.2972</td> <td>    0.036</td> <td>   -8.170</td> <td> 0.000</td> <td>   -0.368    -0.226</td>
</tr>
<tr>
  <th>per_transprt</th> <td>   -0.0218</td> <td>    0.037</td> <td>   -0.586</td> <td> 0.558</td> <td>   -0.095     0.051</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>  <td>   -0.0183</td> <td>    0.011</td> <td>   -1.701</td> <td> 0.089</td> <td>   -0.039     0.003</td>
</tr>
<tr>
  <th>per_enroll</th>   <td>    0.0340</td> <td>    0.013</td> <td>    2.588</td> <td> 0.010</td> <td>    0.008     0.060</td>
</tr>
<tr>
  <th>per_age65</th>    <td>   -0.0006</td> <td>    0.009</td> <td>   -0.063</td> <td> 0.949</td> <td>   -0.019     0.018</td>
</tr>
<tr>
  <th>mdnincm</th>      <td>-2.309e-06</td> <td> 3.83e-06</td> <td>   -0.603</td> <td> 0.547</td> <td>-9.82e-06   5.2e-06</td>
</tr>
<tr>
  <th>populatn</th>     <td> 1.527e-06</td> <td> 5.22e-07</td> <td>    2.926</td> <td> 0.003</td> <td> 5.04e-07  2.55e-06</td>
</tr>
<tr>
  <th>intrland</th>     <td> 3.604e-09</td> <td> 1.94e-09</td> <td>    1.854</td> <td> 0.064</td> <td>-2.06e-10  7.41e-09</td>
</tr>
<tr>
  <th>city</th>         <td>   -0.0834</td> <td>    0.050</td> <td>   -1.668</td> <td> 0.095</td> <td>   -0.181     0.015</td>
</tr>
<tr>
  <th>coast</th>        <td>   -0.1100</td> <td>    0.044</td> <td>   -2.495</td> <td> 0.013</td> <td>   -0.196    -0.024</td>
</tr>
</table>



Percent unemployment remains the biggest factor for probability of republican, followed by population and then percentage of the population that is black. Interestingly, a larger population benefits republican canidates. So does having children enrolled in public schools.



What would happen if we controlled for region of the country with a series of dummy variables?


```python

# these are regions as defined by the bureau of economic analysis
regions = {'new_england':['CT', 'ME', 'MA', 'NH', 'RI', 'VT'] , 
           'mid_east': ['DE', 'DC', 'MD', 'NJ', 'NY', 'PA'],
           'great_lakes': ['IL', 'IN', 'MI', 'OH', 'WI'], 
           'plains': ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'], 
           'south_east':['AL', 'AR', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'SC', 'TN', 'VA', 'WV'], 
           'south_west': ['AZ', 'NM', 'OK', 'TX'], 
           'rocky_mountain': ['CO', 'ID', 'MT', 'UT', 'WY'], 
           'far_west': ['CA', 'AK', 'HI', 'NV', 'OR', 'WA']}


#assigns 1 to that column if state is in that region
for key,val in regions.iteritems():
    congress['in_' + key] = congress.state.apply(lambda state: 1 if state in regions[key] else 0)
```


```python
controlled_region = ['per_construc', 'per_black', 'per_unemplyd', 'per_transprt', 'per_gvtwrkr', 'per_enroll', 
                     'per_age65', 'mdnincm','populatn',  'intrland', 'city', 'coast', 'in_rocky_mountain', 
                     'in_plains', 'in_new_england', 'in_great_lakes', 'in_mid_east', 'in_south_west', 'in_south_east',
                     'in_far_west']

controlled_model = Logit(congress.repub, sm.add_constant(congress[controlled_region])).fit()
controlled_model.summary()
```

    Optimization terminated successfully.
             Current function value: 0.482721
             Iterations 7


    /Users/benjamindykstra/anaconda/lib/python2.7/site-packages/statsmodels/base/model.py:971: RuntimeWarning: invalid value encountered in sqrt
      return np.sqrt(np.diag(self.cov_params()))
    /Users/benjamindykstra/anaconda/lib/python2.7/site-packages/scipy/stats/_distn_infrastructure.py:875: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    /Users/benjamindykstra/anaconda/lib/python2.7/site-packages/scipy/stats/_distn_infrastructure.py:875: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    /Users/benjamindykstra/anaconda/lib/python2.7/site-packages/scipy/stats/_distn_infrastructure.py:1814: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>repub</td>      <th>  No. Observations:  </th>  <td>   435</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   415</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>    19</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sun, 15 Jan 2017</td> <th>  Pseudo R-squ.:     </th>  <td>0.3024</td>  
</tr>
<tr>
  <th>Time:</th>              <td>12:59:29</td>     <th>  Log-Likelihood:    </th> <td> -209.98</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -301.01</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>1.217e-28</td>
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>             <td>   -3.1413</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>per_construc</th>      <td>    0.2647</td> <td>    0.246</td> <td>    1.076</td> <td> 0.282</td> <td>   -0.217     0.747</td>
</tr>
<tr>
  <th>per_black</th>         <td>   -0.0504</td> <td>    0.016</td> <td>   -3.177</td> <td> 0.001</td> <td>   -0.081    -0.019</td>
</tr>
<tr>
  <th>per_unemplyd</th>      <td>   -1.6023</td> <td>    0.321</td> <td>   -4.997</td> <td> 0.000</td> <td>   -2.231    -0.974</td>
</tr>
<tr>
  <th>per_transprt</th>      <td>   -0.1629</td> <td>    0.239</td> <td>   -0.681</td> <td> 0.496</td> <td>   -0.632     0.306</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>       <td>   -0.1043</td> <td>    0.071</td> <td>   -1.466</td> <td> 0.143</td> <td>   -0.244     0.035</td>
</tr>
<tr>
  <th>per_enroll</th>        <td>    0.2362</td> <td>    0.090</td> <td>    2.637</td> <td> 0.008</td> <td>    0.061     0.412</td>
</tr>
<tr>
  <th>per_age65</th>         <td>   -0.0028</td> <td>    0.058</td> <td>   -0.048</td> <td> 0.962</td> <td>   -0.117     0.111</td>
</tr>
<tr>
  <th>mdnincm</th>           <td> 1.224e-05</td> <td> 2.77e-05</td> <td>    0.442</td> <td> 0.659</td> <td>-4.21e-05  6.65e-05</td>
</tr>
<tr>
  <th>populatn</th>          <td>   8.1e-06</td> <td> 3.12e-06</td> <td>    2.597</td> <td> 0.009</td> <td> 1.99e-06  1.42e-05</td>
</tr>
<tr>
  <th>intrland</th>          <td>  5.69e-08</td> <td> 3.16e-08</td> <td>    1.799</td> <td> 0.072</td> <td>-5.08e-09  1.19e-07</td>
</tr>
<tr>
  <th>city</th>              <td>   -0.3625</td> <td>    0.332</td> <td>   -1.092</td> <td> 0.275</td> <td>   -1.014     0.288</td>
</tr>
<tr>
  <th>coast</th>             <td>   -0.5590</td> <td>    0.283</td> <td>   -1.977</td> <td> 0.048</td> <td>   -1.113    -0.005</td>
</tr>
<tr>
  <th>in_rocky_mountain</th> <td>    0.0798</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_plains</th>         <td>   -0.8275</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_new_england</th>    <td>   -1.3708</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_great_lakes</th>    <td>   -0.0533</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_mid_east</th>       <td>    0.0881</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_south_west</th>     <td>   -0.0952</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_south_east</th>     <td>    0.5559</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_far_west</th>       <td>   -1.5183</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
</table>



There seems to be a problem with the parameter estimates on the regions. Lets check the rank of our matrix and covariance matrix


```python
np.linalg.matrix_rank(sm.add_constant(congress[controlled_region])), sm.add_constant(congress[controlled_region]).shape[1]
```




    (20, 21)




```python
controlled_model.cov_params()[['in_rocky_mountain','in_plains', 'in_new_england', 'in_great_lakes', 'in_mid_east', 
                               'in_south_west', 'in_south_east', 'in_far_west']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>in_rocky_mountain</th>
      <th>in_plains</th>
      <th>in_new_england</th>
      <th>in_great_lakes</th>
      <th>in_mid_east</th>
      <th>in_south_west</th>
      <th>in_south_east</th>
      <th>in_far_west</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>const</th>
      <td>6.386492e+13</td>
      <td>6.386492e+13</td>
      <td>6.386492e+13</td>
      <td>6.386492e+13</td>
      <td>6.386492e+13</td>
      <td>6.386492e+13</td>
      <td>6.386492e+13</td>
      <td>6.386492e+13</td>
    </tr>
    <tr>
      <th>per_construc</th>
      <td>-1.091468e-01</td>
      <td>-2.017659e-02</td>
      <td>9.513134e-03</td>
      <td>-1.286969e-02</td>
      <td>-7.181959e-03</td>
      <td>-4.320845e-02</td>
      <td>3.096794e-03</td>
      <td>4.533879e-02</td>
    </tr>
    <tr>
      <th>per_black</th>
      <td>1.607985e-03</td>
      <td>8.341282e-03</td>
      <td>1.636867e-02</td>
      <td>6.578895e-03</td>
      <td>9.300608e-03</td>
      <td>7.842342e-03</td>
      <td>9.308403e-03</td>
      <td>1.461101e-02</td>
    </tr>
    <tr>
      <th>per_unemplyd</th>
      <td>-3.982602e-01</td>
      <td>-3.711613e-01</td>
      <td>-5.241872e-01</td>
      <td>-4.307269e-01</td>
      <td>-4.567296e-01</td>
      <td>-4.366227e-01</td>
      <td>-4.272142e-01</td>
      <td>-3.913676e-01</td>
    </tr>
    <tr>
      <th>per_transprt</th>
      <td>-4.637134e-01</td>
      <td>-4.552612e-01</td>
      <td>-4.104197e-01</td>
      <td>-4.224048e-01</td>
      <td>-4.154182e-01</td>
      <td>-4.295588e-01</td>
      <td>-4.359294e-01</td>
      <td>-4.114073e-01</td>
    </tr>
    <tr>
      <th>per_gvtwrkr</th>
      <td>-1.343203e-01</td>
      <td>-1.268796e-01</td>
      <td>-1.290734e-01</td>
      <td>-1.220045e-01</td>
      <td>-1.277040e-01</td>
      <td>-1.271874e-01</td>
      <td>-1.251051e-01</td>
      <td>-1.230510e-01</td>
    </tr>
    <tr>
      <th>per_enroll</th>
      <td>-3.792752e-01</td>
      <td>-3.790564e-01</td>
      <td>-3.611060e-01</td>
      <td>-3.638837e-01</td>
      <td>-3.476087e-01</td>
      <td>-3.723100e-01</td>
      <td>-3.569134e-01</td>
      <td>-3.560876e-01</td>
    </tr>
    <tr>
      <th>per_age65</th>
      <td>-2.210222e-01</td>
      <td>-2.252882e-01</td>
      <td>-2.237586e-01</td>
      <td>-2.175076e-01</td>
      <td>-2.148911e-01</td>
      <td>-2.172481e-01</td>
      <td>-2.176205e-01</td>
      <td>-2.048646e-01</td>
    </tr>
    <tr>
      <th>mdnincm</th>
      <td>-1.035904e-04</td>
      <td>-1.033100e-04</td>
      <td>-1.089805e-04</td>
      <td>-1.036229e-04</td>
      <td>-1.024194e-04</td>
      <td>-1.023991e-04</td>
      <td>-9.982727e-05</td>
      <td>-9.807287e-05</td>
    </tr>
    <tr>
      <th>populatn</th>
      <td>-6.235154e-06</td>
      <td>-6.505035e-06</td>
      <td>-6.332263e-06</td>
      <td>-6.145872e-06</td>
      <td>-5.934361e-06</td>
      <td>-6.173450e-06</td>
      <td>-6.200979e-06</td>
      <td>-5.638813e-06</td>
    </tr>
    <tr>
      <th>intrland</th>
      <td>-9.919726e-09</td>
      <td>-1.147271e-09</td>
      <td>4.531544e-09</td>
      <td>6.193129e-10</td>
      <td>1.794346e-09</td>
      <td>-2.347084e-09</td>
      <td>3.778531e-10</td>
      <td>-1.691406e-08</td>
    </tr>
    <tr>
      <th>city</th>
      <td>-2.428479e-01</td>
      <td>-2.577943e-01</td>
      <td>-2.339646e-01</td>
      <td>-2.585814e-01</td>
      <td>-2.425629e-01</td>
      <td>-3.206876e-01</td>
      <td>-2.494138e-01</td>
      <td>-2.831823e-01</td>
    </tr>
    <tr>
      <th>coast</th>
      <td>1.471563e-02</td>
      <td>2.524741e-02</td>
      <td>1.110886e-02</td>
      <td>4.183857e-03</td>
      <td>1.168595e-02</td>
      <td>1.428282e-02</td>
      <td>1.356147e-02</td>
      <td>-2.250626e-02</td>
    </tr>
    <tr>
      <th>in_rocky_mountain</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
    <tr>
      <th>in_plains</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
    <tr>
      <th>in_new_england</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
    <tr>
      <th>in_great_lakes</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
    <tr>
      <th>in_mid_east</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
    <tr>
      <th>in_south_west</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
    <tr>
      <th>in_south_east</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
    <tr>
      <th>in_far_west</th>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
      <td>-6.386492e+13</td>
    </tr>
  </tbody>
</table>
</div>



So the matrix is not quite full rank and there is very little covariance between some of the dummy variable parameters. This is probably the cause of the problems. 

Marginal effects:


```python
controlled_model.get_margeff(dummy = True).summary()
```

    /Users/benjamindykstra/anaconda/lib/python2.7/site-packages/statsmodels/discrete/discrete_margins.py:345: RuntimeWarning: invalid value encountered in sqrt
      return cov_me, np.sqrt(np.diag(cov_me))





<table class="simpletable">
<caption>Logit Marginal Effects</caption>
<tr>
  <th>Dep. Variable:</th>  <td>repub</td> 
</tr>
<tr>
  <th>Method:</th>         <td>dydx</td>  
</tr>
<tr>
  <th>At:</th>            <td>overall</td>
</tr>
</table>
<table class="simpletable">
<tr>
          <th></th>             <th>dy/dx</th>    <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>per_construc</th>      <td>    0.0426</td> <td>    0.039</td> <td>    1.081</td> <td> 0.280</td> <td>   -0.035     0.120</td>
</tr>
<tr>
  <th>per_black</th>         <td>   -0.0081</td> <td>    0.002</td> <td>   -3.357</td> <td> 0.001</td> <td>   -0.013    -0.003</td>
</tr>
<tr>
  <th>per_unemplyd</th>      <td>   -0.2576</td> <td>    0.047</td> <td>   -5.535</td> <td> 0.000</td> <td>   -0.349    -0.166</td>
</tr>
<tr>
  <th>per_transprt</th>      <td>   -0.0262</td> <td>    0.038</td> <td>   -0.682</td> <td> 0.495</td> <td>   -0.101     0.049</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>       <td>   -0.0168</td> <td>    0.011</td> <td>   -1.472</td> <td> 0.141</td> <td>   -0.039     0.006</td>
</tr>
<tr>
  <th>per_enroll</th>        <td>    0.0380</td> <td>    0.014</td> <td>    2.797</td> <td> 0.005</td> <td>    0.011     0.065</td>
</tr>
<tr>
  <th>per_age65</th>         <td>   -0.0004</td> <td>    0.009</td> <td>   -0.047</td> <td> 0.962</td> <td>   -0.019     0.018</td>
</tr>
<tr>
  <th>mdnincm</th>           <td> 1.968e-06</td> <td> 4.43e-06</td> <td>    0.444</td> <td> 0.657</td> <td>-6.72e-06  1.07e-05</td>
</tr>
<tr>
  <th>populatn</th>          <td> 1.302e-06</td> <td> 4.76e-07</td> <td>    2.734</td> <td> 0.006</td> <td> 3.69e-07  2.24e-06</td>
</tr>
<tr>
  <th>intrland</th>          <td> 9.066e-09</td> <td> 4.82e-09</td> <td>    1.882</td> <td> 0.060</td> <td>-3.77e-10  1.85e-08</td>
</tr>
<tr>
  <th>city</th>              <td>   -0.0595</td> <td>    0.055</td> <td>   -1.075</td> <td> 0.282</td> <td>   -0.168     0.049</td>
</tr>
<tr>
  <th>coast</th>             <td>   -0.0920</td> <td>    0.047</td> <td>   -1.963</td> <td> 0.050</td> <td>   -0.184    -0.000</td>
</tr>
<tr>
  <th>in_rocky_mountain</th> <td>    0.0128</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_plains</th>         <td>   -0.1341</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_new_england</th>    <td>   -0.2272</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_great_lakes</th>    <td>   -0.0086</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_mid_east</th>       <td>    0.0141</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_south_west</th>     <td>   -0.0154</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_south_east</th>     <td>    0.0883</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
<tr>
  <th>in_far_west</th>       <td>   -0.2441</td> <td>      nan</td> <td>      nan</td> <td>   nan</td> <td>      nan       nan</td>
</tr>
</table>




```python
new_predictions = np.array([1 if x >= 0.5 else 0 for x in controlled_model.predict()])

print classification_report(congress.repub, new_predictions)
print "Confusion matrix:\n"
print confusion_matrix(congress.repub, new_predictions)
```

                 precision    recall  f1-score   support
    
            0.0       0.80      0.71      0.75       207
            1.0       0.76      0.83      0.80       228
    
    avg / total       0.78      0.78      0.78       435
    
    Confusion matrix:
    
    [[148  59]
     [ 38 190]]


The model improves both true negatives and true positives. The R2 is also at a high of 0.3024 because of the better log likelihood value. It seems controlling for region of the country really helped out the model with its predictive accuracy. According to the marginal effects, a district in the south east has a 8 % chance higher of being republican, all else equal. Employment is still the most significant factore in the model and has the highest effect on probability as in the other models.

## Using Sklearn's Recursive Feature Elimination for Feature Selection

For comparison's sake, I will use recursive feature elimination to select features from the dataset. From sklearn's documentation: 
> Features whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

In order to use this method, I will need to use sklearn's logistic regression instead of statsmodels. Something to be aware of is the regularization term 'C' in sklearn - it needs to be set to an extremely high value so that the parameter estimates are the same as statsmodels.


```python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

new_ind_vars = ['per_age65', 'per_black', 'per_blucllr', 'city', 'coast', 'per_construc','per_cvllbrfr',
                'per_enroll','per_farmer','per_finance','per_forborn','per_gvtwrkr','intrland','landsqmi',
                'mdnincm','miltinst','miltmajr','per_miltpop','nucplant','popsqmi','populatn','per_rurlfarm',
                'per_transprt','per_unemplyd','union','per_urban','per_whlretl','in_rocky_mountain', 'in_plains', 
                'in_new_england','in_great_lakes', 'in_mid_east', 'in_south_west', 'in_south_east','in_far_west']

x = sm.add_constant(congress[new_ind_vars])
y = congress.repub

#cross validated recursive feature elimination, with auc scoring
rfecv = RFECV(estimator= LogisticRegression(C = 1e10),scoring = 'roc_auc' )
x_new = rfecv.fit_transform(x, y)

print x_new.shape, x.shape


#get the selected features
good_features = []
for i in range(len(x.columns.values)):
    if rfecv.support_[i]:
        good_features.append(x.columns.values[i])

print "The selected features: \n", good_features
```

    (435, 14) (435, 36)
    The selected features: 
    ['const', 'per_age65', 'city', 'coast', 'per_construc', 'per_cvllbrfr', 'per_finance', 'per_gvtwrkr', 'miltinst', 'per_miltpop', 'per_transprt', 'per_unemplyd', 'in_rocky_mountain', 'in_new_england']


A lot of the same features that were previously included by xgboost were selected for. Thats rather encouraging. I'll fit the statsmodels logit to the data using the new features now.


```python
selected_model = Logit(congress.repub, x[good_features]).fit()
selected_model.summary()

```

    Optimization terminated successfully.
             Current function value: 0.497313
             Iterations 7





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>repub</td>      <th>  No. Observations:  </th>  <td>   435</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   421</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>    13</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sun, 15 Jan 2017</td> <th>  Pseudo R-squ.:     </th>  <td>0.2813</td>  
</tr>
<tr>
  <th>Time:</th>              <td>12:59:33</td>     <th>  Log-Likelihood:    </th> <td> -216.33</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -301.01</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>2.489e-29</td>
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>             <td>   18.5263</td> <td>    3.283</td> <td>    5.644</td> <td> 0.000</td> <td>   12.092    24.960</td>
</tr>
<tr>
  <th>per_age65</th>         <td>   -0.1854</td> <td>    0.050</td> <td>   -3.703</td> <td> 0.000</td> <td>   -0.284    -0.087</td>
</tr>
<tr>
  <th>city</th>              <td>   -0.6637</td> <td>    0.299</td> <td>   -2.220</td> <td> 0.026</td> <td>   -1.250    -0.078</td>
</tr>
<tr>
  <th>coast</th>             <td>   -0.5589</td> <td>    0.267</td> <td>   -2.090</td> <td> 0.037</td> <td>   -1.083    -0.035</td>
</tr>
<tr>
  <th>per_construc</th>      <td>    0.4259</td> <td>    0.209</td> <td>    2.039</td> <td> 0.041</td> <td>    0.017     0.835</td>
</tr>
<tr>
  <th>per_cvllbrfr</th>      <td>   -0.2094</td> <td>    0.046</td> <td>   -4.595</td> <td> 0.000</td> <td>   -0.299    -0.120</td>
</tr>
<tr>
  <th>per_finance</th>       <td>    0.3566</td> <td>    0.142</td> <td>    2.518</td> <td> 0.012</td> <td>    0.079     0.634</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>       <td>   -0.2279</td> <td>    0.068</td> <td>   -3.355</td> <td> 0.001</td> <td>   -0.361    -0.095</td>
</tr>
<tr>
  <th>miltinst</th>          <td>    0.3400</td> <td>    0.117</td> <td>    2.899</td> <td> 0.004</td> <td>    0.110     0.570</td>
</tr>
<tr>
  <th>per_miltpop</th>       <td>   -0.4299</td> <td>    0.123</td> <td>   -3.485</td> <td> 0.000</td> <td>   -0.672    -0.188</td>
</tr>
<tr>
  <th>per_transprt</th>      <td>   -0.3541</td> <td>    0.230</td> <td>   -1.538</td> <td> 0.124</td> <td>   -0.806     0.097</td>
</tr>
<tr>
  <th>per_unemplyd</th>      <td>   -1.9399</td> <td>    0.283</td> <td>   -6.857</td> <td> 0.000</td> <td>   -2.494    -1.385</td>
</tr>
<tr>
  <th>in_rocky_mountain</th> <td>    1.1965</td> <td>    0.909</td> <td>    1.316</td> <td> 0.188</td> <td>   -0.585     2.978</td>
</tr>
<tr>
  <th>in_new_england</th>    <td>   -1.0359</td> <td>    0.673</td> <td>   -1.538</td> <td> 0.124</td> <td>   -2.356     0.284</td>
</tr>
</table>




```python
selected_model.get_margeff(dummy = True).summary()
```




<table class="simpletable">
<caption>Logit Marginal Effects</caption>
<tr>
  <th>Dep. Variable:</th>  <td>repub</td> 
</tr>
<tr>
  <th>Method:</th>         <td>dydx</td>  
</tr>
<tr>
  <th>At:</th>            <td>overall</td>
</tr>
</table>
<table class="simpletable">
<tr>
          <th></th>             <th>dy/dx</th>    <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>per_age65</th>         <td>   -0.0168</td> <td>    0.001</td> <td>  -18.297</td> <td> 0.000</td> <td>   -0.019    -0.015</td>
</tr>
<tr>
  <th>city</th>              <td>   -0.1133</td> <td>    0.051</td> <td>   -2.205</td> <td> 0.027</td> <td>   -0.214    -0.013</td>
</tr>
<tr>
  <th>coast</th>             <td>   -0.0928</td> <td>    0.044</td> <td>   -2.130</td> <td> 0.033</td> <td>   -0.178    -0.007</td>
</tr>
<tr>
  <th>per_construc</th>      <td>    0.0707</td> <td>    0.034</td> <td>    2.073</td> <td> 0.038</td> <td>    0.004     0.138</td>
</tr>
<tr>
  <th>per_cvllbrfr</th>      <td>   -0.0348</td> <td>    0.007</td> <td>   -5.029</td> <td> 0.000</td> <td>   -0.048    -0.021</td>
</tr>
<tr>
  <th>per_finance</th>       <td>    0.0592</td> <td>    0.023</td> <td>    2.584</td> <td> 0.010</td> <td>    0.014     0.104</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>       <td>   -0.0378</td> <td>    0.011</td> <td>   -3.518</td> <td> 0.000</td> <td>   -0.059    -0.017</td>
</tr>
<tr>
  <th>miltinst</th>          <td>    0.0565</td> <td>    0.019</td> <td>    3.004</td> <td> 0.003</td> <td>    0.020     0.093</td>
</tr>
<tr>
  <th>per_miltpop</th>       <td>   -0.0714</td> <td>    0.019</td> <td>   -3.672</td> <td> 0.000</td> <td>   -0.109    -0.033</td>
</tr>
<tr>
  <th>per_transprt</th>      <td>   -0.0588</td> <td>    0.038</td> <td>   -1.553</td> <td> 0.120</td> <td>   -0.133     0.015</td>
</tr>
<tr>
  <th>per_unemplyd</th>      <td>   -0.0394</td> <td>    0.014</td> <td>   -2.767</td> <td> 0.006</td> <td>   -0.067    -0.011</td>
</tr>
<tr>
  <th>in_rocky_mountain</th> <td>    0.1838</td> <td>    0.123</td> <td>    1.500</td> <td> 0.134</td> <td>   -0.056     0.424</td>
</tr>
<tr>
  <th>in_new_england</th>    <td>   -0.1720</td> <td>    0.111</td> <td>   -1.554</td> <td> 0.120</td> <td>   -0.389     0.045</td>
</tr>
</table>



Wow, so in this model percent unemployment doesn't have nearly the same maginitude effect that it does in the other models. However, I like what I see with regions - New England has always been very liberal, and that is reflected in the dummy variables 'in_new_england' and 'coast'. 

Interestingly, percentage black was not selected for.. I still believe that that is a very important feature and should be included. Lets add it in and reexamine the results.


```python
good_features.append('per_black')
new_selected_model = Logit(congress.repub, x[good_features]).fit()
new_selected_model.summary()

```

    Optimization terminated successfully.
             Current function value: 0.488322
             Iterations 7





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>repub</td>      <th>  No. Observations:  </th>  <td>   435</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   420</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>    14</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sun, 15 Jan 2017</td> <th>  Pseudo R-squ.:     </th>  <td>0.2943</td>  
</tr>
<tr>
  <th>Time:</th>              <td>12:59:33</td>     <th>  Log-Likelihood:    </th> <td> -212.42</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -301.01</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>2.411e-30</td>
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>             <td>   17.5971</td> <td>    3.363</td> <td>    5.233</td> <td> 0.000</td> <td>   11.006    24.188</td>
</tr>
<tr>
  <th>per_age65</th>         <td>   -0.1827</td> <td>    0.050</td> <td>   -3.620</td> <td> 0.000</td> <td>   -0.282    -0.084</td>
</tr>
<tr>
  <th>city</th>              <td>   -0.7272</td> <td>    0.304</td> <td>   -2.390</td> <td> 0.017</td> <td>   -1.324    -0.131</td>
</tr>
<tr>
  <th>coast</th>             <td>   -0.6234</td> <td>    0.271</td> <td>   -2.302</td> <td> 0.021</td> <td>   -1.154    -0.093</td>
</tr>
<tr>
  <th>per_construc</th>      <td>    0.4754</td> <td>    0.214</td> <td>    2.223</td> <td> 0.026</td> <td>    0.056     0.895</td>
</tr>
<tr>
  <th>per_cvllbrfr</th>      <td>   -0.2059</td> <td>    0.046</td> <td>   -4.429</td> <td> 0.000</td> <td>   -0.297    -0.115</td>
</tr>
<tr>
  <th>per_finance</th>       <td>    0.3566</td> <td>    0.142</td> <td>    2.505</td> <td> 0.012</td> <td>    0.078     0.636</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>       <td>   -0.1804</td> <td>    0.071</td> <td>   -2.545</td> <td> 0.011</td> <td>   -0.319    -0.041</td>
</tr>
<tr>
  <th>miltinst</th>          <td>    0.3094</td> <td>    0.118</td> <td>    2.617</td> <td> 0.009</td> <td>    0.078     0.541</td>
</tr>
<tr>
  <th>per_miltpop</th>       <td>   -0.3837</td> <td>    0.123</td> <td>   -3.120</td> <td> 0.002</td> <td>   -0.625    -0.143</td>
</tr>
<tr>
  <th>per_transprt</th>      <td>   -0.3083</td> <td>    0.233</td> <td>   -1.323</td> <td> 0.186</td> <td>   -0.765     0.149</td>
</tr>
<tr>
  <th>per_unemplyd</th>      <td>   -1.7545</td> <td>    0.294</td> <td>   -5.959</td> <td> 0.000</td> <td>   -2.332    -1.177</td>
</tr>
<tr>
  <th>in_rocky_mountain</th> <td>    0.8609</td> <td>    0.931</td> <td>    0.925</td> <td> 0.355</td> <td>   -0.963     2.685</td>
</tr>
<tr>
  <th>in_new_england</th>    <td>   -1.2589</td> <td>    0.674</td> <td>   -1.869</td> <td> 0.062</td> <td>   -2.579     0.062</td>
</tr>
<tr>
  <th>per_black</th>         <td>   -0.0331</td> <td>    0.012</td> <td>   -2.662</td> <td> 0.008</td> <td>   -0.057    -0.009</td>
</tr>
</table>




```python
new_selected_model.get_margeff().summary()
```




<table class="simpletable">
<caption>Logit Marginal Effects</caption>
<tr>
  <th>Dep. Variable:</th>  <td>repub</td> 
</tr>
<tr>
  <th>Method:</th>         <td>dydx</td>  
</tr>
<tr>
  <th>At:</th>            <td>overall</td>
</tr>
</table>
<table class="simpletable">
<tr>
          <th></th>             <th>dy/dx</th>    <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>per_age65</th>         <td>   -0.0298</td> <td>    0.008</td> <td>   -3.815</td> <td> 0.000</td> <td>   -0.045    -0.014</td>
</tr>
<tr>
  <th>city</th>              <td>   -0.1185</td> <td>    0.048</td> <td>   -2.450</td> <td> 0.014</td> <td>   -0.213    -0.024</td>
</tr>
<tr>
  <th>coast</th>             <td>   -0.1016</td> <td>    0.043</td> <td>   -2.357</td> <td> 0.018</td> <td>   -0.186    -0.017</td>
</tr>
<tr>
  <th>per_construc</th>      <td>    0.0775</td> <td>    0.034</td> <td>    2.268</td> <td> 0.023</td> <td>    0.011     0.144</td>
</tr>
<tr>
  <th>per_cvllbrfr</th>      <td>   -0.0336</td> <td>    0.007</td> <td>   -4.810</td> <td> 0.000</td> <td>   -0.047    -0.020</td>
</tr>
<tr>
  <th>per_finance</th>       <td>    0.0581</td> <td>    0.023</td> <td>    2.570</td> <td> 0.010</td> <td>    0.014     0.102</td>
</tr>
<tr>
  <th>per_gvtwrkr</th>       <td>   -0.0294</td> <td>    0.011</td> <td>   -2.615</td> <td> 0.009</td> <td>   -0.051    -0.007</td>
</tr>
<tr>
  <th>miltinst</th>          <td>    0.0504</td> <td>    0.019</td> <td>    2.698</td> <td> 0.007</td> <td>    0.014     0.087</td>
</tr>
<tr>
  <th>per_miltpop</th>       <td>   -0.0625</td> <td>    0.019</td> <td>   -3.257</td> <td> 0.001</td> <td>   -0.100    -0.025</td>
</tr>
<tr>
  <th>per_transprt</th>      <td>   -0.0502</td> <td>    0.038</td> <td>   -1.333</td> <td> 0.183</td> <td>   -0.124     0.024</td>
</tr>
<tr>
  <th>per_unemplyd</th>      <td>   -0.2859</td> <td>    0.041</td> <td>   -7.044</td> <td> 0.000</td> <td>   -0.365    -0.206</td>
</tr>
<tr>
  <th>in_rocky_mountain</th> <td>    0.1403</td> <td>    0.151</td> <td>    0.928</td> <td> 0.353</td> <td>   -0.156     0.437</td>
</tr>
<tr>
  <th>in_new_england</th>    <td>   -0.2051</td> <td>    0.108</td> <td>   -1.898</td> <td> 0.058</td> <td>   -0.417     0.007</td>
</tr>
<tr>
  <th>per_black</th>         <td>   -0.0054</td> <td>    0.002</td> <td>   -2.737</td> <td> 0.006</td> <td>   -0.009    -0.002</td>
</tr>
</table>




```python
new_selected_model.bic, selected_model.bic, controlled_model.bic
```




    (515.97027448372125, 517.71751551732223, 541.47425095794426)



The percentage of unemployment has been restored to it's former prominence. Other important variables (statistically and magnitude) include coast, construction workers, and New England. New England has the largest magnitude with a 30% reduction in probability of being republican if the district is in the region. The new model also has the lowest BIC.

Lets look at the predictive power:


```python
new_selected_predictions = np.array([1 if x >= 0.5 else 0 for x in new_selected_model.predict()])

print classification_report(congress.repub, new_selected_predictions)
print "Confusion matrix:\n"
print confusion_matrix(congress.repub, new_selected_predictions)
```

                 precision    recall  f1-score   support
    
            0.0       0.75      0.70      0.72       207
            1.0       0.74      0.79      0.76       228
    
    avg / total       0.74      0.74      0.74       435
    
    Confusion matrix:
    
    [[144  63]
     [ 49 179]]


These are actually worse predictions that the previous controlled model. We choose that model instead.


# Conclusion

While there are certainly more powerful (prediction wise) algorithms out there, logistic regression is nice because it can be interpreted meaningfully and actionably. With the help of xgboost, I picked out some features that I wouldn't have otherwise and was able to improve the logit model. After controlling for region, I achieved the best results, with in class predictions at 71% and 83% for democrats and republicans, respectively. With more detailed information on things like district religious preferences, age distributions, previous voting preferences etc, I'm sure that the model would make much better predictions.

Thanks for following along! :)


