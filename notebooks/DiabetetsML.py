#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# to standardise data into a common range
# 

# train-test-split funct to split our data into training and test data.

# **now data collection and analysis**

# In[ ]:


#loading the diabetes dataset to a pandas dataframe


# In[10]:


diabetes_dataset= pd.read_csv("../data/diabetes.csv")
#this will load csv file into a pandas dataframe


# In[4]:


#pd.read_csv? #what does a func does to know


# In[5]:


#print first 5rows
diabetes_dataset.head()


# In[6]:


# get the no of rows and cols of the diabetes_dataset  --1st rows --then cols here we have 8 features and 1output cols
diabetes_dataset.shape


# In[7]:


# get the statistical measures of the data
diabetes_dataset.describe()
#50 percentile means 50% is below 72. percentile and percentage is difff


# In[11]:


diabetes_dataset['Outcome'].value_counts()
#counts how many 0 and 1 in outcome col  label0- non diabetic and 1-diabetic ache


# In[12]:


#get mean, u can read as mean value of glucose for all people who are diabetic is 141.25, its important it tells if a new data comes, ota kar modhey jabe 0 or 1
diabetes_dataset.groupby('Outcome').mean()


# In[13]:


#now seperate outcome from other features  if u a dropping a col axis=1, if row- axis=0
x=diabetes_dataset.drop(columns='Outcome',axis=1)
y=diabetes_dataset['Outcome']


# In[11]:


print(x)


# In[14]:


x


# In[15]:


#DAta Standardisation   this means creating a instance of standardscaler and storing it in scalar
scalar=StandardScaler()
scalar.fit(x)


# In[16]:


#now transform the data or scalar.firtransform 2steps in 1steps.. so that it fits the dtaa and transform in a range
standarised_data=scalar.transform(x)


# In[17]:


standarised_data  # now all datas are in similar range 0-1


# In[18]:


standarised_data.shape


# In[19]:


x=standarised_data


# In[20]:


#now split data into test and train , this funct will give 4outputs so we need 4vars , 0.2 means 20% we will take data for testing, stratify is to make sure
# not all diabetic patients go to xtrain means disproportane in spliting. to ensure proportionate spliting. how data is getting splitted. give same no as sir to get same kind of split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[20]:


print(x.shape,xtrain.shape,xtest.shape)


# In[21]:


#training the model we are using a  linear model SVC-suuport vector classifier
classifier=svm.SVC(kernel='linear') #load the svm model into classifier var  . our model is classifier named


# In[22]:


#now we will fit our data into classfier, taining our model
classifier.fit(xtrain,ytrain)


# now we can evaluate our model. how many times our model is predicting correctly

# In[23]:


#Accuracy score on the training data

xtrain_prediction=classifier.predict(xtrain)
training_data_accuracy=accuracy_score(xtrain_prediction,ytrain)


# In[24]:


print("accuracy score of training data: ", training_data_accuracy) # out of 100, our model is predicting 79 times correctly. 79% is good, as our dataset is less, we can increase acc by optimisation technique


# our model has already seen xtrain data, but now is imp to test our model by ytest.

# In[25]:


xtest_prediction=classifier.predict(xtest)


# In[26]:


test_data_accuracy=accuracy_score(xtest_prediction,ytest)


# In[27]:


print("Accuracy for my unseen test data: ", test_data_accuracy)


# making a predictive system where i can give input

# In[28]:


input_data=(10,168,74,0,0,38,0.537,34) # this is in list datatype

#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance, we trained model on 768 rows, so it will expect i give 768 but i give only 1 so reshape
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standardise the input data  as our model is trained on stand data
std_data=scalar.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if(prediction[0]==0): #prediction is a list. it has only 1element to access the first element[0]. our model gives a list not int.
  print("the person is not diabetic")
else:
  print("the person is diabetic\n")


