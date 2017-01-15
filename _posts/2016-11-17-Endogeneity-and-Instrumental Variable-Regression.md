---
layout: post
title: Endogenous Variables and IV Regression in Python
---


Endogeneity occurs when it is impossible to establish a chain of causality among variables. An instance of this might be AIDS funding in Uganda and AIDS occurence in Uganda. The problem here is that the amount of funding is a function of the number of AIDS cases in Uganda, however the number of cases is also affected by funding - what came first, chicken or the egg?

In this notebook I use a fertility data set to explore factors that might affect the age of a woman when she has her first child. The data is from James Heakins, a former undergraduate student of Professor Wooldridge at Michigan State University, for a term project.  They come from Botswana’s 1988 Demographic and Health Survey.

Here's my roadmap for evaluating this data:

1. I begin by looking at some scatterplots of the variables so that I can begin to get an idea of their relationship to one another. 
2. Estimate a naive equation with a possibly endogenous variable. The dependent variable will always be age at first birth.
3. Identify the endogenous variable and pick an appropiate instrument for it. Test for the relevancy of this instrument using an f-test.
4. Use 2-stage least squares regression to estimate a new OLS model with the proper instrument included. I use IV2SLS written by the wonderful people at statsmodels. 
5. As an exercise, replicate part 4 using matrix algebra in Numpy
6. Test for exogeneity of (supposedly) endogenous variable using the Hausman-Wu test.
7. Add another instrument to the mix, repeat step 3 for both instruments.
8. Test for overidentification using the Sagan test.



```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from __future__ import division
import seaborn as sns

%matplotlib inline
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 18.5, 10.5

def print_resids(preds, resids):
    ax = sns.regplot(preds, resids);
    ax.set(xlabel = 'Predicted values', ylabel = 'errors', title = 'Predicted values vs. Errors')
    plt.show();
```

## Part 1

### Load data, do some exploration on it


```python
fertility = pd.read_stata("http://rlhick.people.wm.edu/econ407/data/fertility.dta")
fertility.shape
```




    (4361, 27)







Take a peek at the relationship between education and age of first birth, as well as the number of children and age of first birth.


```python
fertility.plot.scatter('educ', 'agefbrth'); fertility.plot.scatter( 'agefbrth', 'children');
```


![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_6_0.png)



![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_6_1.png)


If you squint a bit, there seems to be a small positive relationship between how much education a woman recieves and the age of her first birth. As for the relationship between age of first birth and total number of children, women who have their first child at a young age seem to have more children overall.

Lets compare the first birth age of women who use a method of contraception to those who don't.

These women have used a method of birth control at least once:


```python
print fertility[fertility['usemeth'] == 1]['agefbrth'].dropna().describe()
sns.distplot(fertility[fertility['usemeth'] == 1]['agefbrth'].dropna() );
```

    count    2182.000000
    mean       18.952795
    std         2.834758
    min        11.000000
    25%        17.000000
    50%        19.000000
    75%        20.000000
    max        38.000000
    Name: agefbrth, dtype: float64



![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_8_1.png)


These women have never used any method of birth control:


```python
print fertility[fertility['usemeth'] == 0]['agefbrth'].dropna().describe()
sns.distplot(fertility[fertility['usemeth'] == 0]['agefbrth'].dropna() );
```

    count    1031.000000
    mean       19.115421
    std         3.591423
    min        10.000000
    25%        17.000000
    50%        19.000000
    75%        21.000000
    max        38.000000
    Name: agefbrth, dtype: float64



![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_10_1.png)


Somewhat interesting... the mean age of 1st birth for women who have *ever* used a method of birth control is lower than those that have *never* used one. However, there is slightly more variation in the women who don't use birth control.

## Part 2

## estimate naive equation with a possibly endogenous variable

The equation I will estimate is:

$$ agefbrth = \beta_0 + \beta_1 educ + \beta_2 monthfm + \beta_3 ceb + \beta_4 idlnchld $$

We assume that there is no relationship between education and the number of children ever born or education and month of marriage. We also assume that there is no relationship between month of first marriage and number of children born.


Theres a huge problem with missing data in this dataset, roughly a half of agefbrth data a missing from the total amount of observations. I get rid of the data now that has null values for age of first birth, education, month of first marriage, and children ever born.



```python
#gets all columns that aren't null
no_null = fertility[(fertility['agefbrth'].notnull()) & (fertility['educ'].notnull()) & 
                    (fertility.monthfm.notnull()) & (fertility['ceb'].notnull()) & (fertility['idlnchld'].notnull())] 

print "lost {} samples of data out of a total {} samples".format(fertility.shape[0] - no_null.shape[0],
                                                                 fertility.shape[0] )

ind_vars = ['monthfm', 'ceb', 'educ', 'idlnchld']
dep_var = 'agefbrth'
x = no_null[ind_vars] 
y = no_null[dep_var]

x_const = sm.add_constant(x)

first_model_results = sm.OLS(y, x_const, missing = 'drop').fit()

#results = first_model.fit()

first_model_results.summary()
```

    lost 2489 samples of data out of a total 4361 samples





<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>agefbrth</td>     <th>  R-squared:         </th> <td>   0.058</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.056</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   28.83</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 15 Jan 2017</td> <th>  Prob (F-statistic):</th> <td>2.78e-23</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:51:50</td>     <th>  Log-Likelihood:    </th> <td> -4811.1</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1872</td>      <th>  AIC:               </th> <td>   9632.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1867</td>      <th>  BIC:               </th> <td>   9660.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>    <td>   18.9236</td> <td>    0.294</td> <td>   64.289</td> <td> 0.000</td> <td>   18.346    19.501</td>
</tr>
<tr>
  <th>monthfm</th>  <td>    0.0413</td> <td>    0.020</td> <td>    2.043</td> <td> 0.041</td> <td>    0.002     0.081</td>
</tr>
<tr>
  <th>ceb</th>      <td>   -0.1588</td> <td>    0.034</td> <td>   -4.640</td> <td> 0.000</td> <td>   -0.226    -0.092</td>
</tr>
<tr>
  <th>educ</th>     <td>    0.1303</td> <td>    0.019</td> <td>    6.830</td> <td> 0.000</td> <td>    0.093     0.168</td>
</tr>
<tr>
  <th>idlnchld</th> <td>   -0.0101</td> <td>    0.034</td> <td>   -0.298</td> <td> 0.766</td> <td>   -0.077     0.056</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>508.807</td> <th>  Durbin-Watson:     </th> <td>   1.920</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1674.738</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.339</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.781</td>  <th>  Cond. No.          </th> <td>    43.9</td>
</tr>
</table>




```python
print_resids(first_model_results.predict(x_const), first_model_results.resid)
```


![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_13_0.png)



```python
print "the descriptive statistics for the errors and a histogram of them:\n\n", first_model_results.resid.describe()
sns.distplot(first_model_results.resid);
```

    the descriptive statistics for the errors and a histogram of them:
    
    count    1872.000000
    mean       -0.000002
    std         3.162461
    min        -8.951200
    25%        -2.016429
    50%        -0.468160
    75%         1.414013
    max        19.689455
    dtype: float64



![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_14_1.png)


There's definitely some linear structure to the errors here, caused by the discrete nature of the dependent variable. (Maybe build a classification model for this?)

The exogeneity assumptions are not valid here. It's reasonable to believe that amount of education recieved is correlated with errors in age of first birth. Education and the month of the first marriage are possibly *weakly* related. I'm not sure how the school years in Botswana are structured, but if a woman is in school for part of a year, she may not want to get married during any of those months, thus affecting the month she is married. We proceed with caution.

## Part 3: Pick an instrument and test for relevancy and strength

I hypothesize that the most endogenous variable is education. If a child is born at a young age, there is less time for education, and it is impossible to determine which is the causal variable.

I will use electricity as an instrumental variable. There is no reason to believe that errors in age of birth and electricity are directly related to each other. However, education and electricity are probably related because places that have electricity are probably more developed and thus more likely to have a school. So, electricity is related to age of first birth only via education.

test for the relevancy of electricity as an instrumental variable:
   1. run relevancy equation where exogenous variables and instrument predict the endogenous variable.
   2. test whether the coefficient on the instrument is 0 via an F-test with one degree of freedom
    



```python
rel = ['monthfm', 'ceb', 'electric']
endog = 'educ'

dropped_na = fertility[(fertility.monthfm.notnull()) & (fertility.ceb.notnull()) & (fertility.electric.notnull())
                    & (fertility.educ.notnull())]

only_exog = sm.add_constant(dropped_na[rel])
relevancy_results = sm.OLS(dropped_na[endog], only_exog).fit()

relevancy_results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>educ</td>       <th>  R-squared:         </th> <td>   0.269</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.268</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   253.9</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 15 Jan 2017</td> <th>  Prob (F-statistic):</th> <td>2.70e-140</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:51:53</td>     <th>  Log-Likelihood:    </th> <td> -5605.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2076</td>      <th>  AIC:               </th> <td>1.122e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2072</td>      <th>  BIC:               </th> <td>1.124e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>    <td>    5.7469</td> <td>    0.205</td> <td>   27.973</td> <td> 0.000</td> <td>    5.344     6.150</td>
</tr>
<tr>
  <th>monthfm</th>  <td>    0.0220</td> <td>    0.022</td> <td>    1.007</td> <td> 0.314</td> <td>   -0.021     0.065</td>
</tr>
<tr>
  <th>ceb</th>      <td>   -0.4456</td> <td>    0.032</td> <td>  -13.811</td> <td> 0.000</td> <td>   -0.509    -0.382</td>
</tr>
<tr>
  <th>electric</th> <td>    4.6753</td> <td>    0.216</td> <td>   21.634</td> <td> 0.000</td> <td>    4.251     5.099</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>38.041</td> <th>  Durbin-Watson:     </th> <td>   1.527</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  23.335</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.100</td> <th>  Prob(JB):          </th> <td>8.57e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.521</td> <th>  Cond. No.          </th> <td>    24.4</td>
</tr>
</table>




Run the hypothesis test that the coefficient on electric is 0:

[this](http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.RegressionResults.f_test.html) is where I found this test


```python
hypothesis = '(electric = 0)'
print relevancy_results.f_test(hypothesis)
```

    <F test: F=array([[ 468.01322886]]), p=9.5448237988e-94, df_denom=2072, df_num=1>


With an F-statistic of 440.417, this is surely a relevant and strong instrument. The F-statistic must be at least 10 in order to be a strong instrument.


## Part 4: Instrumenting using two-stage least squares
Some background and information about two-stage least squares regression: It's called two stage because there are actually two stages of regression done (earth shattering I know). 

### First stage
In the first stage, the matrix $X$, which contains the endogenous information, is projected on to $Z$. $Z$ is the matrix without endogenous information that includes the variable(s) that are our instruments. Mathematically:
$$ X = \gamma Z + V $$ where $V$ is the error, and $\hat\gamma = (Z'  Z)^{-1}Z' X$.

The projection of X on to Z is then:
$$ \hat X = \hat\gamma Z = Z (Z'  Z)^{-1}Z' X $$
$$ = P_z X, \text{ where } P_Z = Z (Z'  Z)^{-1}Z'$$

### Second stage

We repeat the same process as above using

$$ Y = \hat X \beta + \epsilon $$


### Specifying the two stage least squares model

The documentation for IV2SLS in statsmodels is somewhat confusing and conflicts with some of the terminology that I've used in my classes. So, for clarification: 
   1. endog is the dependent variable, y
   2. exog is the x matrix that has the endogenous information in it. Include the endogenous variables in it.
   3. instrument is the z matrix. Include all the variables that are not endogenous and replace the endogenous variables from the exog matrix (above) with what ever instruments you choose for them.


```python
no_null_iv = fertility[(fertility['agefbrth'].notnull()) & (fertility['electric'].notnull()) & 
                    (fertility['monthfm'].notnull()) & (fertility['ceb'].notnull()) & (fertility['educ'].notnull())
                      & (fertility['idlnchld'].notnull())]
endog = no_null_iv['agefbrth']
exog = no_null_iv[['monthfm', 'ceb', 'idlnchld', 'educ']]
instr = no_null_iv[['monthfm', 'ceb', 'idlnchld', 'electric']]
dep_var_iv = no_null_iv['agefbrth']

exog_constant = sm.add_constant(exog)
instr_constant = sm.add_constant(instr)
no_endog_results = IV2SLS(endog, exog_constant, instrument = instr_constant).fit()

no_endog_results.summary()
```




<table class="simpletable">
<caption>IV2SLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>agefbrth</td>     <th>  R-squared:         </th> <td>   0.005</td>
</tr>
<tr>
  <th>Model:</th>                 <td>IV2SLS</td>      <th>  Adj. R-squared:    </th> <td>   0.003</td>
</tr>
<tr>
  <th>Method:</th>               <td>Two Stage</td>    <th>  F-statistic:       </th> <td>   28.06</td>
</tr>
<tr>
  <th></th>                    <td>Least Squares</td>  <th>  Prob (F-statistic):</th> <td>1.16e-22</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 15 Jan 2017</td> <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:51:56</td>     <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1870</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1865</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>    <td>   17.0973</td> <td>    0.507</td> <td>   33.696</td> <td> 0.000</td> <td>   16.102    18.092</td>
</tr>
<tr>
  <th>monthfm</th>  <td>    0.0411</td> <td>    0.021</td> <td>    1.973</td> <td> 0.049</td> <td>    0.000     0.082</td>
</tr>
<tr>
  <th>ceb</th>      <td>   -0.0635</td> <td>    0.041</td> <td>   -1.540</td> <td> 0.124</td> <td>   -0.144     0.017</td>
</tr>
<tr>
  <th>idlnchld</th> <td>    0.0780</td> <td>    0.040</td> <td>    1.949</td> <td> 0.051</td> <td>   -0.000     0.156</td>
</tr>
<tr>
  <th>educ</th>     <td>    0.3276</td> <td>    0.048</td> <td>    6.829</td> <td> 0.000</td> <td>    0.234     0.422</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>530.134</td> <th>  Durbin-Watson:     </th> <td>   1.907</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1901.731</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.366</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.116</td>  <th>  Cond. No.          </th> <td>    43.8</td>
</tr>
</table>



The effect of education on the age of first birth is fairly large. 

On average, every year of education increases age of first birth by .327 years. This speaks to the positive effects of education. Interestingly, it is the only statistically significant variable at the .01 level.


```python
print_resids(no_endog_results.predict(), no_endog_results.resid)
```


![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_22_0.png)



```python
print "the descriptive statistics for the errors and a histogram of them:\n\n", no_endog_results.resid.describe()
sns.distplot(no_endog_results.resid);
```

    the descriptive statistics for the errors and a histogram of them:
    
    count    1.870000e+03
    mean    -7.649794e-07
    std      3.252076e+00
    min     -8.714241e+00
    25%     -2.057206e+00
    50%     -4.409256e-01
    75%      1.484385e+00
    max      2.060653e+01
    dtype: float64



![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_23_1.png)


## Part 5: replicate using matrix algebra

first, replicate OLS estimates:
$$ b = (x'  x)^{-1}x' y $$


```python
x_mat_ols = np.matrix(x_const)
y_mat_ols = np.matrix(y)
y_mat_ols = np.reshape(y_mat_ols, (-1, 1)) #reshape so that its a single column vector, not row vector
b_ols = np.linalg.inv(x_mat_ols.T*x_mat_ols)*x_mat_ols.T*y_mat_ols
print b_ols
```

    [[  1.89236317e+01]
     [  4.13089916e-02]
     [ -1.58784062e-01]
     [  1.30279362e-01]
     [ -1.00923479e-02]]


those check out with our original naive estimates with endogeneity.

Now, the IV estimates, using the z-matrix:

$$ b = (z'  x)^{-1}z' y $$


```python
y_iv_mat = np.matrix(endog)
y_iv_mat = np.reshape(y_iv_mat, (-1, 1))
z_mat = np.matrix(instr_constant)
x_mat_iv = np.matrix(exog_constant) 
np.linalg.inv(z_mat.T * x_mat_iv)*z_mat.T*y_iv_mat
```




    matrix([[ 17.09725189],
            [  0.04106776],
            [ -0.06348502],
            [  0.07799444],
            [  0.32765239]], dtype=float32)



Yay, everything checks out! For clarity, the z matrix only contains exogenous information, while x constains our endogenous variable education.

## Part 6: Hausman-Wu test for endogeneity

Steps:
   1. Run relevancy regression (endog variable ~ exogenous variables + instrument(s) + error)
   2. Get the predicted residuals from this regression ($\hat r$)
   3. Run regression $Y = X\gamma + \hat r \beta + u$
   4. Test whether the coefficient on $\hat r$ is significantly different than 0 using an F-test with 1 degree of freedom
  



```python
# add relevancy equation residuals on to the endogenous matrix
x_const['relevancy_resids'] = relevancy_results.resid

# run endogenous regression now with residuals added in
endog_test_results = sm.OLS(y, x_const, missing = 'drop').fit()

endog_test_results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>agefbrth</td>     <th>  R-squared:         </th> <td>   0.069</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.067</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   27.83</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 15 Jan 2017</td> <th>  Prob (F-statistic):</th> <td>2.91e-27</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:51:59</td>     <th>  Log-Likelihood:    </th> <td> -4795.4</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1870</td>      <th>  AIC:               </th> <td>   9603.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1864</td>      <th>  BIC:               </th> <td>   9636.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>            <td>   17.6155</td> <td>    0.407</td> <td>   43.296</td> <td> 0.000</td> <td>   16.818    18.413</td>
</tr>
<tr>
  <th>monthfm</th>          <td>    0.0412</td> <td>    0.020</td> <td>    2.047</td> <td> 0.041</td> <td>    0.002     0.081</td>
</tr>
<tr>
  <th>ceb</th>              <td>   -0.0618</td> <td>    0.040</td> <td>   -1.541</td> <td> 0.123</td> <td>   -0.140     0.017</td>
</tr>
<tr>
  <th>educ</th>             <td>    0.3102</td> <td>    0.043</td> <td>    7.213</td> <td> 0.000</td> <td>    0.226     0.394</td>
</tr>
<tr>
  <th>idlnchld</th>         <td>   -0.0005</td> <td>    0.034</td> <td>   -0.014</td> <td> 0.989</td> <td>   -0.067     0.066</td>
</tr>
<tr>
  <th>relevancy_resids</th> <td>   -0.2192</td> <td>    0.047</td> <td>   -4.654</td> <td> 0.000</td> <td>   -0.312    -0.127</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>494.869</td> <th>  Durbin-Watson:     </th> <td>   1.931</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1630.863</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.302</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.761</td>  <th>  Cond. No.          </th> <td>    61.3</td>
</tr>
</table>




```python
null_hypothesis = '(relevancy_resids = 0)'
print endog_test_results.f_test(null_hypothesis)
```

    <F test: F=array([[ 21.65873982]]), p=3.48680999408e-06, df_denom=1864, df_num=1>


We reject the null hypothesis that education is exogenous and conclude that education is indeed an endogenous variable.

The thinking behind this test is that the residuals should only include endogenous information of education because we explained all the exogenous information with monthfm and ceb. If we can then use that endogenous information to predict y in a meaningful way (i.e. the coefficient isn't zero), then that is evidence that education is correlated with age of first birth via the error term.

## Part 7: Add another instrument

Now we instrument for education using more than one instrumental variable. Living in an urban area should not be related to differences in the age of first birth, however, it will affect educational attainment. Again, more developed areas should (presumably) have better access to schools and education.


```python
two_ivs = fertility[(fertility['agefbrth'].notnull()) & (fertility['electric'].notnull()) & 
                    (fertility['monthfm'].notnull()) & (fertility['ceb'].notnull()) & (fertility['educ'].notnull())
                      & (fertility['idlnchld'].notnull()) & (fertility['urban'].notnull())]

endog = two_ivs['agefbrth']
exog = two_ivs[['monthfm', 'ceb', 'idlnchld', 'educ']]
instr = two_ivs[['monthfm', 'ceb', 'idlnchld', 'electric', 'urban']]


exog_constant = sm.add_constant(exog)
instr_constant = sm.add_constant(instr)
two_iv_results = IV2SLS(endog, exog_constant, instrument = instr_constant).fit()

two_iv_results.summary()
```




<table class="simpletable">
<caption>IV2SLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>agefbrth</td>     <th>  R-squared:         </th> <td>   0.033</td>
</tr>
<tr>
  <th>Model:</th>                 <td>IV2SLS</td>      <th>  Adj. R-squared:    </th> <td>   0.031</td>
</tr>
<tr>
  <th>Method:</th>               <td>Two Stage</td>    <th>  F-statistic:       </th> <td>   25.28</td>
</tr>
<tr>
  <th></th>                    <td>Least Squares</td>  <th>  Prob (F-statistic):</th> <td>2.01e-20</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 15 Jan 2017</td> <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:52:03</td>     <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1870</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1865</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>    <td>   17.6755</td> <td>    0.488</td> <td>   36.215</td> <td> 0.000</td> <td>   16.718    18.633</td>
</tr>
<tr>
  <th>monthfm</th>  <td>    0.0412</td> <td>    0.021</td> <td>    2.011</td> <td> 0.044</td> <td>    0.001     0.081</td>
</tr>
<tr>
  <th>ceb</th>      <td>   -0.0939</td> <td>    0.040</td> <td>   -2.335</td> <td> 0.020</td> <td>   -0.173    -0.015</td>
</tr>
<tr>
  <th>idlnchld</th> <td>    0.0501</td> <td>    0.039</td> <td>    1.281</td> <td> 0.200</td> <td>   -0.027     0.127</td>
</tr>
<tr>
  <th>educ</th>     <td>    0.2655</td> <td>    0.046</td> <td>    5.795</td> <td> 0.000</td> <td>    0.176     0.355</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>531.978</td> <th>  Durbin-Watson:     </th> <td>   1.918</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1891.653</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.374</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.089</td>  <th>  Cond. No.          </th> <td>    43.8</td>
</tr>
</table>




```python
print_resids(two_iv_results.predict(), two_iv_results.resid)
```


![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_34_0.png)



```python
print two_iv_results.resid.describe()
sns.distplot(two_iv_results.resid);
```

    count    1.870000e+03
    mean     2.600930e-07
    std      3.204986e+00
    min     -8.368925e+00
    25%     -2.044604e+00
    50%     -4.985476e-01
    75%      1.411119e+00
    max      2.031738e+01
    dtype: float64



![png](/img/Endogeneity%20and%20Instrumental%20Variable%20Regression_files/Endogeneity%20and%20Instrumental%20Variable%20Regression_35_1.png)


Now, we test for the relevancy and strength of our instruments:


```python
rel = ['monthfm', 'ceb', 'electric', 'urban']
endog = 'educ'

only_exog = sm.add_constant(fertility[rel])
relevancy_results = sm.OLS(fertility[endog], only_exog, missing = 'drop').fit()

relevancy_results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>educ</td>       <th>  R-squared:         </th> <td>   0.281</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.280</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   202.7</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 15 Jan 2017</td> <th>  Prob (F-statistic):</th> <td>7.76e-147</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:52:06</td>     <th>  Log-Likelihood:    </th> <td> -5587.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2076</td>      <th>  AIC:               </th> <td>1.118e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2071</td>      <th>  BIC:               </th> <td>1.121e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>const</th>    <td>    5.1261</td> <td>    0.228</td> <td>   22.450</td> <td> 0.000</td> <td>    4.678     5.574</td>
</tr>
<tr>
  <th>monthfm</th>  <td>    0.0247</td> <td>    0.022</td> <td>    1.137</td> <td> 0.255</td> <td>   -0.018     0.067</td>
</tr>
<tr>
  <th>ceb</th>      <td>   -0.4105</td> <td>    0.033</td> <td>  -12.626</td> <td> 0.000</td> <td>   -0.474    -0.347</td>
</tr>
<tr>
  <th>electric</th> <td>    4.2665</td> <td>    0.225</td> <td>   18.979</td> <td> 0.000</td> <td>    3.826     4.707</td>
</tr>
<tr>
  <th>urban</th>    <td>    1.0167</td> <td>    0.169</td> <td>    6.019</td> <td> 0.000</td> <td>    0.685     1.348</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>32.600</td> <th>  Durbin-Watson:     </th> <td>   1.546</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  21.735</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.118</td> <th>  Prob(JB):          </th> <td>1.91e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.558</td> <th>  Cond. No.          </th> <td>    25.7</td>
</tr>
</table>




```python
null_hypotheses = '(electric = 0), (urban = 0)'
print relevancy_results.f_test(null_hypotheses)
```

    <F test: F=array([[ 256.10258471]]), p=4.11102682888e-100, df_denom=2071, df_num=2>


I conclude that these are indeed strong and relevant instrumental variables.

## Conclusion

While the predictive power of our model may not be stellar with an $R^2$ of 0.033, we can be sure that our estimates for $\beta$ are unbiased and that there is not a problem with endogeneity. Education, instrumented with access to electricity and urban area, remains the most important factor in predicting the age at which a woman will have her first birth.

Statsmodels does a good job of IV regression, and all results match the output given by Stata. However, some features of Stata are lacking in statsmodels. A robust testing API for hausman-wu and Sargan's test of over identification would be very nice. In stata, those tests are as simple as typing "estat overid". Also, the examples on the statsmodels wiki are not stellar and could be expanded upon to include an econometric use case that I'm sure many data scientists and econometricians would find useful.
