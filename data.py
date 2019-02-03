#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.model_selection import GridSearchCV


# In[22]:


train_data=pd.read_csv("train.csv")


# In[23]:


del train_data['std_wtd']


# In[24]:


train_data.head()


# In[25]:


train_data['DateOfDeparture'] = pd.to_datetime(train_data['DateOfDeparture'])


# In[26]:


train_data['year']=train_data['DateOfDeparture'].dt.year
train_data['month']=train_data['DateOfDeparture'].dt.month
train_data['day']=train_data['DateOfDeparture'].dt.day
train_data['dayofyear']=train_data['DateOfDeparture'].dt.dayofyear
train_data['dayofweek']=train_data['DateOfDeparture'].dt.dayofweek


# In[27]:


train_data.head()


# In[28]:


train_data['DateOfDeparture']=train_data['DateOfDeparture'].astype(np.int64)


# In[29]:


train_data['WeeksToDeparture']=np.log1p(train_data['WeeksToDeparture'])


# In[30]:


from math import sin, cos, sqrt, atan2, radians
import mpu
dx=[]
for index, row in train_data.iterrows():
    # Point one
    lat1 = radians(row['LatitudeDeparture'])
    lon1 = radians(row['LongitudeDeparture'])

    # Point two
    lat2 = radians(row['LatitudeArrival'])
    lon2 = radians(row['LongitudeArrival'])

    # What you were looking for
    dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
    dx.append(dist)


# In[31]:


train_data['featureX']=dx
train_data.head()


# In[32]:


train_data.dtypes


# In[33]:


import seaborn as sns
corr = train_data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[34]:


features=list(train_data.columns)
features.remove("PAX")
X=train_data.loc[:,features]
Y=train_data.loc[:,"PAX"]


# In[35]:


test_data=pd.read_csv("test.csv")
del test_data['std_wtd']
test_data['DateOfDeparture'] = pd.to_datetime(test_data['DateOfDeparture'])
test_data['year']=test_data['DateOfDeparture'].dt.year
test_data['month']=test_data['DateOfDeparture'].dt.month
test_data['day']=test_data['DateOfDeparture'].dt.day
test_data['dayofyear']=test_data['DateOfDeparture'].dt.dayofyear
test_data['dayofweek']=test_data['DateOfDeparture'].dt.dayofweek
test_data['DateOfDeparture']=test_data['DateOfDeparture'].astype(np.int64)
test_data['WeeksToDeparture']=np.log1p(test_data['WeeksToDeparture'])
from math import sin, cos, sqrt, atan2, radians
import mpu
dx=[]
for index, row in test_data.iterrows():
    # Point one
    lat1 = radians(row['LatitudeDeparture'])
    lon1 = radians(row['LongitudeDeparture'])

    # Point two
    lat2 = radians(row['LatitudeArrival'])
    lon2 = radians(row['LongitudeArrival'])

    # What you were looking for
    dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
    dx.append(dist)
test_data['featureX']=dx
test_data.head()


# In[43]:


temp=pd.concat([X,test_data]).reset_index(drop=True)


# In[45]:


featuresC=['Departure','CityDeparture','Arrival','CityArrival']


# In[46]:


le = LabelEncoder()
for each in featuresC:
    temp[each] = le.fit_transform(temp[each])


# In[49]:


X.shape,test_data.shape


# In[50]:


rowstemp=temp.shape[0]
X=temp.loc[0:X.shape[0]-1,:]
test_data=temp.loc[X.shape[0]:,:]


# In[51]:


X.shape,test_data.shape


# In[52]:


#Train and test split stratified
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=50)
for train_index,test_index in sss.split(X,Y):
    X_train=X.loc[train_index,:].reset_index(drop=True)
    X_test=X.loc[test_index,:].reset_index(drop=True)
    y_train=Y[train_index].reset_index(drop=True)
    y_test=Y[test_index].reset_index(drop=True)


# In[53]:


import lightgbm
trainD=lightgbm.Dataset(X_train,y_train)
testD=lightgbm.Dataset(X_test,y_test)
params = {'num_leaves': 15,
         'min_data_in_leaf': 10, 
         'num_class':8,
         'objective':'multiclass',
         'max_depth': 12,
         'learning_rate': 0.1,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "lambda_l1": 0.1,
         "random_state": 133,
         "verbosity": -1}

clf = lightgbm.train(params, trainD, 800,valid_sets = [trainD, testD])


# In[54]:


print("Train: ",sklearn.metrics.f1_score(y_train,clf.predict(X_train).argmax(axis=1),average='weighted'))
print("Test: ",sklearn.metrics.f1_score(y_test,clf.predict(X_test).argmax(axis=1),average='weighted'))


# In[ ]:


#F1 Score on test 0.52


# In[19]:


#Submission


# In[56]:


test_data.reset_index(drop=True,inplace=True)


# In[58]:


P=clf.predict(test_data).argmax(axis=1)


# In[62]:


idd=np.arange(0,test_data.shape[0],1)


# In[64]:


idd.shape,len(P)


# In[65]:


output=pd.DataFrame()
output['Id']=idd
output['Label']=P


# In[67]:


output.to_csv("output_last.csv",index=False)


# In[ ]:





# In[ ]:




