#!/usr/bin/env python
# coding: utf-8

# # Lead Score - Case Study

# ## Problem Statement
# An X Education need help to select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires us to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%. <br>
# ## Goals and Objectives
# There are quite a few goals for this case study.
# - Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.
# - There are some more problems presented by the company which your model should be able to adjust to if the company's requirement changes in the future so you will need to handle these as well. These problems are provided in a separate doc file. Please fill it based on the logistic regression model you got in the first step. Also, make sure you include this in your final PPT where you'll make recommendations.

# ___All the outcomes and understandings are written in <font color= green> GREEN</font>___

# In[61]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
#Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 1 : Loading and Cleaning Data
# 
# ##  1.1  Import Data 

# In[62]:


# Loading the data using Pandas
leads = pd.read_csv("Leads.csv")


# ## 1.2 Inspect the dataframe
# This helps to give a good idea of the dataframes.

# In[63]:


# The .info() code gives almost the entire information that needs to be inspected, so let's start from there
leads.info()


# In[64]:


#To get the idea of how the table looks like we can use .head() or .tail() command
leads.head()


# In[65]:


# The .shape code gives the no. of rows and columns
leads.shape


# In[66]:


#To get an idea of the numeric values, use .describe()
leads.describe()


# ## 1.3 Cleaning the dataframe

# In[67]:


# Converting all the values to lower case
leads = leads.applymap(lambda s:s.lower() if type(s) == str else s)


# In[68]:


# Replacing 'Select' with NaN (Since it means no option is selected)
leads = leads.replace('select',np.nan)


# In[69]:


# Checking if there are columns with one unique value since it won't affect our analysis
leads.nunique()


# In[70]:


# Dropping unique valued columns
leads1= leads.drop(['Magazine','Receive More Updates About Our Courses','I agree to pay the amount through cheque','Get updates on DM Content','Update me on Supply Chain Content'],axis=1)


# In[71]:


# Checking the percentage of missing values
round(100*(leads1.isnull().sum()/len(leads1.index)), 2)


# In[72]:


# Removing all the columns that are no required and have 35% null values
leads2 = leads1.drop(['Asymmetrique Profile Index','Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Score','Lead Profile','Tags','Lead Quality','How did you hear about X Education','City','Lead Number'],axis=1)
leads2.head()


# In[73]:


# Rechecking the percentage of missing values
round(100*(leads2.isnull().sum()/len(leads2.index)), 2)


# <font color= green>___There is a huge value of null variables in 4 columns as seen above. But removing the rows with the null value will cost us a lot of data and they are important columns. So, instead we are going to replace the NaN values with 'not provided'. This way we have all the data and almost no null values. In case these come up in the model, it will be of no use and we can drop it off then.___</font>

# In[74]:


leads2['Specialization'] = leads2['Specialization'].fillna('not provided') 
leads2['What matters most to you in choosing a course'] = leads2['What matters most to you in choosing a course'].fillna('not provided')
leads2['Country'] = leads2['Country'].fillna('not provided')
leads2['What is your current occupation'] = leads2['What is your current occupation'].fillna('not provided')
leads2.info()


# In[75]:


# Rechecking the percentage of missing values
round(100*(leads2.isnull().sum()/len(leads2.index)), 2)


# In[76]:


leads2["Country"].value_counts()


# In[77]:


def slots(x):
    category = ""
    if x == "india":
        category = "india"
    elif x == "not provided":
        category = "not provided"
    else:
        category = "outside india"
    return category

leads2['Country'] = leads2.apply(lambda x:slots(x['Country']), axis = 1)
leads2['Country'].value_counts()


# In[78]:


# Rechecking the percentage of missing values
round(100*(leads2.isnull().sum()/len(leads2.index)), 2)


# In[79]:


# Checking the percent of lose if the null values are removed
round(100*(sum(leads2.isnull().sum(axis=1) > 1)/leads2.shape[0]),2)


# In[80]:


leads3 = leads2[leads2.isnull().sum(axis=1) <1]


# In[81]:


# Code for checking number of rows left in percent
round(100*(leads3.shape[0])/(leads.shape[0]),2)


# In[82]:


# Rechecking the percentage of missing values
round(100*(leads3.isnull().sum()/len(leads3.index)), 2)


# In[83]:


# To familiarize all the categorical values
for column in leads3:
    print(leads3[column].astype('category').value_counts())
    print('----------------------------------------------------------------------------------------')


# In[84]:


# Removing Id values since they are unique for everyone
leads_final = leads3.drop('Prospect ID',1)
leads_final.shape


# ## 2. EDA

# ### 2.1. Univariate Analysis

# #### 2.1.1. Categorical Variables

# In[85]:


leads_final.info()


# In[100]:


plt.figure(figsize = (20,40))

plt.subplot(6,2,1)
sns.countplot(x='Lead Origin', data=leads_final)
plt.title('Lead Origin')

plt.subplot(6,2,2)
sns.countplot(x='Do Not Email', data=leads_final)
plt.title('Do Not Email')

plt.subplot(6,2,3)
sns.countplot(x='Do Not Call', data=leads_final)
plt.title('Do Not Call')

plt.subplot(6,2,4)
sns.countplot(x='Country', data=leads_final)
plt.title('Country')

plt.subplot(6,2,5)
sns.countplot(x='Search', data=leads_final)
plt.title('Search')

plt.subplot(6,2,6)
sns.countplot(x='Newspaper Article', data=leads_final)
plt.title('Newspaper Article')

plt.subplot(6,2,7)
sns.countplot(x='X Education Forums', data=leads_final)
plt.title('X Education Forums')

plt.subplot(6,2,8)
sns.countplot(x='Newspaper', data=leads_final)
plt.title('Newspaper')

plt.subplot(6,2,9)
sns.countplot(x='Digital Advertisement', data=leads_final)
plt.title('Digital Advertisement')

plt.subplot(6,2,10)
sns.countplot(x='Through Recommendations', data=leads_final)
plt.title('Through Recommendations')

plt.subplot(6,2,11)
sns.countplot(x='A free copy of Mastering The Interview', data=leads_final)
plt.title('A free copy of Mastering The Interview')

plt.subplot(6,2,12)
sns.countplot(x='Last Notable Activity', data=leads_final)
plt.xticks(rotation=90)
plt.title('Last Notable Activity')


plt.show()


# In[102]:


sns.countplot(x='Lead Source', data=leads_final)
plt.xticks(rotation=90)
plt.title('Lead Source')
plt.show()


# In[106]:


plt.figure(figsize = (20,30))

plt.subplot(2,2,1)
sns.countplot(x='Specialization', data=leads_final)
plt.xticks(rotation=90)
plt.title('Specialization')

plt.subplot(2,2,2)
sns.countplot(x='What is your current occupation', data=leads_final)
plt.xticks(rotation=90)
plt.title('Current Occupation')

plt.subplot(2,2,3)
sns.countplot(x='What matters most to you in choosing a course', data=leads_final)
plt.xticks(rotation=90)
plt.title('What matters most to you in choosing a course')

plt.subplot(2,2,4)
sns.countplot(x='Last Activity', data=leads_final)
plt.xticks(rotation=90)
plt.title('Last Activity')

plt.show()


# In[107]:


sns.countplot(leads['Converted'])
plt.title('Converted("Y variable")')
plt.show()


# #### 2.1.1. Numerical Variables

# In[108]:


leads_final.info()


# In[109]:


plt.figure(figsize = (10,10))
plt.subplot(221)
plt.hist(leads_final['TotalVisits'], bins = 200)
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(222)
plt.hist(leads_final['Total Time Spent on Website'], bins = 10)
plt.title('Total Time Spent on Website')

plt.subplot(223)
plt.hist(leads_final['Page Views Per Visit'], bins = 20)
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show()


# ### 2.1. Relating all the categorical variables to Converted

# In[110]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Lead Origin', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')

plt.subplot(1,2,2)
sns.countplot(x='Lead Source', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[111]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Do Not Email', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Do Not Email')

plt.subplot(1,2,2)
sns.countplot(x='Do Not Call', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Do Not Call')
plt.show()


# In[112]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Last Activity', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')

plt.subplot(1,2,2)
sns.countplot(x='Country', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Country')
plt.show()


# In[113]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()


# In[114]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='What matters most to you in choosing a course', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')

plt.subplot(1,2,2)
sns.countplot(x='Search', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Search')
plt.show()


# In[115]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper Article', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Newspaper Article')

plt.subplot(1,2,2)
sns.countplot(x='X Education Forums', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('X Education Forums')
plt.show()


# In[116]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Newspaper')

plt.subplot(1,2,2)
sns.countplot(x='Digital Advertisement', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Digital Advertisement')
plt.show()


# In[117]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Through Recommendations', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Through Recommendations')

plt.subplot(1,2,2)
sns.countplot(x='A free copy of Mastering The Interview', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('A free copy of Mastering The Interview')
plt.show()


# In[118]:


sns.countplot(x='Last Notable Activity', hue='Converted', data= leads_final).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')
plt.show()


# In[119]:


# To check the correlation among varibles
plt.figure(figsize=(10,5))
sns.heatmap(leads_final.corr())
plt.show()


# <font color= green>___It is understandable from the above EDA that there are many elements that have very little data and so will be of less relevance to our analysis.___</font>

# In[120]:


numeric = leads_final[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# <font color= green>___There aren't any major outliers, so moving on to analysis___</font>

# ## 3. Dummy Variables

# In[121]:


leads_final.info()


# In[123]:


leads_final.loc[:, leads_final.dtypes == 'object'].columns


# In[124]:


# Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(leads_final[['Lead Origin','Specialization' ,'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation','A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)
# Add the results to the master dataframe
leads_final_dum = pd.concat([leads_final, dummy], axis=1)
leads_final_dum


# In[125]:


leads_final_dum = leads_final_dum.drop(['What is your current occupation_not provided','Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call','Last Activity', 'Country', 'Specialization', 'Specialization_not provided','What is your current occupation','What matters most to you in choosing a course', 'Search','Newspaper Article', 'X Education Forums', 'Newspaper','Digital Advertisement', 'Through Recommendations','A free copy of Mastering The Interview', 'Last Notable Activity'], 1)
leads_final_dum


# ## 4. Test-Train Split

# In[127]:


# Import the required library
from sklearn.model_selection import train_test_split


# In[128]:


X = leads_final_dum.drop(['Converted'], 1)
X.head()


# In[129]:


# Putting the target variable in y
y = leads_final_dum['Converted']
y.head()


# In[130]:


# Split the dataset into 70% and 30% for train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)


# In[131]:


# Import MinMax scaler
from sklearn.preprocessing import MinMaxScaler
# Scale the three numeric features
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()


# In[132]:


# To check the correlation among varibles
plt.figure(figsize=(20,30))
sns.heatmap(X_train.corr())
plt.show()


# <font color= green>___Since there are a lot of variables it is difficult to drop variable. We'll do it after RFE___</font>

# ## 5. Model Building 

# In[136]:


# Import 'LogisticRegression'
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()


# In[134]:





# In[142]:


# Running RFE with 15 variables as output
rfe = RFE(estimator=logreg, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[143]:


# Features that have been selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[144]:


# Put all the columns selected by RFE in the variable 'col'
col = X_train.columns[rfe.support_]


# <font color= green>___All the variables selected by RFE, next statistics part (p-values and the VIFs).___</font>

# In[145]:


# Selecting columns selected by RFE
X_train = X_train[col]


# In[146]:


# Importing statsmodels
import statsmodels.api as sm


# In[147]:


X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[148]:


# Importing 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[149]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= green>___The VIF values seem fine but the p-values aren't. So removing 'Last Notable Activity had a phone conversation'___</font>

# In[150]:


X_train.drop('Last Notable Activity_had a phone conversation', axis = 1, inplace = True)


# In[151]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[152]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= green>___The VIF values seem fine but the p-values aren't. So removing 'What is your current occupation housewife'___</font>

# In[153]:


X_train.drop('What is your current occupation_housewife', axis = 1, inplace = True)


# In[154]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[155]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= green>___The VIF values seem fine but the p-values aren't. So removing 'What is your current occupation other'___</font>

# In[156]:


X_train.drop('What is your current occupation_other', axis = 1, inplace = True)


# In[157]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[158]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <font color= green>___All the VIF values are good and all the p-values are below 0.05. So we can fix model.___</font>

# ## 6. Creating Prediction

# In[159]:


# Predicting the probabilities on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[160]:


# Reshaping to an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[161]:


# Data frame with given convertion rate and probablity of predicted ones
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[162]:


# Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# ## 7. Model Evaluation

# In[163]:


# Importing metrics from sklearn for evaluation
from sklearn import metrics


# In[164]:


# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[165]:


# Predicted     not_churn    churn
# Actual
# not_churn        3403       492
# churn             729      1727


# In[166]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# <font color= green>___That's around 81% accuracy with is a very good value___</font>

# In[167]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[168]:


# Calculating the sensitivity
TP/(TP+FN)


# In[169]:


# Calculating the specificity
TN/(TN+FP)


# <font color= green>___With the current cut off as 0.5 we have around 81% accuracy, sensitivity of around 70% and specificity of around 87%.___</font>

# ## 7. Optimise Cut off (ROC Curve)

# The previous cut off was randomely selected. Now to find the optimum one

# In[170]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[171]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[172]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# <font color= green>___The area under ROC curve is 0.87 which is a very good value.___</font>

# In[173]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[174]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[175]:


# Plotting it
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# <font color= green>___From the graph it is visible that the optimal cut off is at 0.35.___</font>

# In[176]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.35 else 0)
y_train_pred_final.head()


# In[177]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[178]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[179]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[180]:


# Calculating the sensitivity
TP/(TP+FN)


# In[181]:


# Calculating the specificity
TN/(TN+FP)


# <font color= green>___With the current cut off as 0.35 we have accuracy, sensitivity and specificity of around 80%.___</font>

# ## 8. Prediction on Test set

# In[182]:


# Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[183]:


# Substituting all the columns in the final train model
col = X_train.columns


# In[184]:


# Select the columns in X_train for X_test as well
X_test = X_test[col]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm
X_test_sm


# In[185]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[186]:


# Making prediction using cut off 0.35
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final


# In[187]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[188]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[189]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[190]:


# Calculating the sensitivity
TP/(TP+FN)


# In[191]:


# Calculating the specificity
TN/(TN+FP)


# <font color= green>___With the current cut off as 0.35 we have accuracy, sensitivity and specificity of around 80%.___</font>

# ## 9. Precision-Recall

# In[192]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[193]:


# Precision = TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[194]:


#Recall = TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# <font color= green>___With the current cut off as 0.35 we have Precision around 78% and Recall around 70%___</font>

# ### 9.1. Precision and recall tradeoff

# In[195]:


from sklearn.metrics import precision_recall_curve


# In[196]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[197]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[198]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[199]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_train_pred_final.head()


# In[200]:


# Accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[201]:


# Creating confusion matrix again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[202]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[203]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[204]:


#Recall = TP / TP + FN
TP / (TP + FN)


# <font color= green>___With the current cut off as 0.41 we have Precision around 74% and Recall around 76%___</font>

# ## 10. Prediction on Test set

# In[205]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[206]:


# Making prediction using cut off 0.41
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_pred_final


# In[207]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[208]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[209]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[210]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[211]:


#Recall = TP / TP + FN
TP / (TP + FN)


# <font color= green>___With the current cut off as 0.41 we have Precision around 73% and Recall around 75%___</font>

# ## Conclusion
# It was found that the variables that mattered the most in the potential buyers are (In descending order) :
# 1.	The total time spend on the Website.
# 2.	Total number of visits.
# 3.	When the lead source was: <br>
# a.	Google<br>
# b.	Direct traffic<br>
# c.	Organic search<br>
# d.	Welingak website<br>
# 4.	When the last activity was:<br>
# a.	SMS<br>
# b.	Olark chat conversation<br>
# 5.	When the lead origin is Lead add format.
# 6.	When their current occupation is as a working professional.<br>
# Keeping these in mind the X Education can flourish as they have a very high chance to get almost all the potential buyers to change their mind and buy their courses.

# In[ ]:




