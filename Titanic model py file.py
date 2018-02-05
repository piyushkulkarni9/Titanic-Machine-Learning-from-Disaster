
# coding: utf-8

# ### Titanic Machine Learning from Disaster

# In[ ]:


# Loading Modules


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# In[ ]:


# Loading Datasets

# Loading train and test dataset


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[ ]:


# Looking into the training dataset


# In[3]:


train.head()


# In[4]:


train.shape


# In[5]:


# Describing training dataset


# In[6]:


train.describe()


# In[7]:


train.describe(include=['O'])


# In[8]:


train.info()


# In[9]:


train.isnull().sum()


# In[ ]:


# There are 177 rows with missing *Age*, 687 rows with missing *Cabin* and 2 rows with missing *Embarked* information.


# In[10]:


## Looking into the testing dataset


# In[11]:


test.shape


# In[12]:


test.head()


# In[13]:


test.info()


# In[14]:


#There are missing entries for *Age* in Test dataset as well.


# In[15]:


test.isnull().sum()


# In[ ]:


# There are 86 rows with missing *Age*, 327 rows with missing *Cabin* and 1 row with missing *Fare* information.


# In[ ]:


## Relationship between Features and Survival


# In[16]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i"%len(train))


# In[ ]:


### Pclass vs. Survival


# In[17]:


train.Pclass.value_counts()


# In[18]:


train.groupby('Pclass').Survived.value_counts()


# In[19]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[20]:


#train.groupby('Pclass').Survived.mean().plot(kind='bar')
sns.barplot(x='Pclass', y='Survived', data=train)


# In[21]:


### Sex vs. Survival


# In[22]:


train.Sex.value_counts()


# In[23]:


train.groupby('Sex').Survived.value_counts()


# In[24]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[25]:


#train.groupby('Sex').Survived.mean().plot(kind='bar')
sns.barplot(x='Sex', y='Survived', data=train)


# In[26]:


# Pclass & Sex vs. Survival


# In[27]:


tab = pd.crosstab(train['Pclass'], train['Sex'])
print (tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')


# In[28]:


sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)


# In[29]:


# From the above plot, it can be seen that:
# Women from 1st and 2nd Pclass have almost 100% survival chance. 
# Men from 2nd and 3rd Pclass have only around 10% survival chance.


# In[30]:


# Pclass, Sex & Embarked vs. Survival


# In[31]:


sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)


# In[32]:


# From the above plot, it can be seen that:
# Almost all females from Pclass 1 and 2 survived.
# Females dying were mostly from 3rd Pclass.
# Males from Pclass 1 only have slightly higher survival chance than Pclass 2 and 3.


# In[ ]:


# Embarked vs. Survived


# In[33]:


train.Embarked.value_counts()


# In[34]:


train.groupby('Embarked').Survived.value_counts()


# In[35]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[36]:


#train.groupby('Embarked').Survived.mean().plot(kind='bar')
sns.barplot(x='Embarked', y='Survived', data=train)


# In[37]:


#Parch vs. Survival


# In[38]:


train.Parch.value_counts()


# In[39]:


train.groupby('Parch').Survived.value_counts()


# In[40]:


train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()


# In[41]:


sns.barplot(x='Parch', y='Survived', ci=None, data=train)


# In[42]:


# SibSp vs. Survival


# In[43]:


train.SibSp.value_counts()


# In[44]:


train.groupby('SibSp').Survived.value_counts()


# In[45]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()


# In[46]:


sns.barplot(x='SibSp', y='Survived', ci=None, data=train)


# In[47]:


# Age vs. Survival


# In[48]:


fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)


# In[ ]:


# From Pclass violinplot, we can see that:
# 1st Pclass has very few children as compared to other two classes.
# 1st Plcass has more old people as compared to other two classes.
# Almost all children (between age 0 to 10) of 2nd Pclass survived.
# Most children of 3rd Pclass survived.
# Younger people of 1st Pclass survived as compared to its older people.

# From Sex violinplot, we can see that:
# Most male children (between age 0 to 14) survived.
# Females with age between 18 to 40 have better survival chance.


# In[50]:


total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')


# In[ ]:


# From the above figures, we can see that:
# Combining both male and female, we can see that children with age between 0 to 5 have better chance of survival.
# Females with age between "18 to 40" and "50 and above" have higher chance of survival.
# Males with age between 0 to 14 have better chance of survival.


# In[ ]:


# Correlating Features


# In[ ]:


# Heatmap of Correlation between different features:

#Positive numbers = Positive correlation, i.e. increase in one feature will increase the other feature & vice-versa.

#Negative numbers = Negative correlation, i.e. increase in one feature will decrease the other feature & vice-versa.

# In our case, we focus on which features have strong positive or negative correlation with the *Survived* feature.


# In[51]:


corr=train.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# In[ ]:


## Feature Extraction


# In[ ]:


# Name Feature

# Exreacting titles from Name column.


# In[52]:


train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')


# In[53]:


train.head()


# In[54]:


pd.crosstab(train['Title'], train['Sex'])


# In[55]:


# The number of passengers with each *Title* is shown above.

# Replacing some less common titles with the name "Other".


# In[56]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',  	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[57]:


# Converting the categorical *Title* values into numeric form.


# In[58]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[59]:


train.head()


# In[60]:


### Sex Feature

# Converting the categorical value of Sex into numeric.


# In[61]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[62]:


train.head()


# In[ ]:


# Embarked Feature


# In[63]:


train.Embarked.unique()


# In[64]:


# Checking number of passengers for each Embarked category.


# In[65]:


train.Embarked.value_counts()


# In[ ]:


# Category "S" has maximum passengers. Replacing "nan" values with "S".


# In[66]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[67]:


train.head()


# In[69]:


# Converting categorical value of Embarked into numeric.


# In[70]:


for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[71]:


train.head()


# In[ ]:


# Age Feature


# In[72]:


# We first fill the NULL values of *Age* with a random number between (mean_age - std_age) and (mean_age + std_age). 

# We then create a new column named *AgeBand*. This categorizes age into 5 different age range.


# In[73]:


for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())


# In[74]:


train.head()


# In[75]:


# Mapping Age according to AgeBand.


# In[76]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# In[77]:


train.head()


# In[78]:


# Fare Feature


# In[79]:


# Replace missing *Fare* values with the median of Fare.


# In[80]:


for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# In[81]:


# Creating FareBand.


# In[82]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())


# In[83]:


train.head()


# In[84]:


# Mapping Fare according to FareBand


# In[86]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[87]:


train.head()


# In[88]:


# SibSp & Parch Feature

#Combining SibSp & Parch feature, we create a new feature named FamilySize.


# In[89]:


for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# In[90]:


#Data shows that: 

# Having FamilySize upto 4 (from 2 to 4) has better survival chance. 
# FamilySize = 1, i.e. travelling alone has less survival chance.
# Large FamilySize (size of 5 and above) also have less survival chance.


# In[ ]:


# Let's create a new feature named IsAlone. This feature is used to check how is the survival chance while travelling alone as compared to travelling with family.


# In[91]:


for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# In[92]:


train.head(1)


# In[93]:


test.head(1)


# In[ ]:


# Feature Selection


# In[94]:


features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)


# In[95]:


train.head()


# In[96]:


test.head()


# In[ ]:


# Training classifier


# In[ ]:


# Classification & Accuracy 


# In[ ]:


# Defining training and testing set


# In[98]:


X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape


# In[99]:


# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# In[ ]:


# Logistic Regression


# In[100]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')


# In[ ]:


# k-Nearest Neighbors


# In[101]:


clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)


# In[102]:


# Decision Tree


# In[103]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)


# In[104]:


# Random Forest


# In[108]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)


# In[ ]:


# Confusion Matrix


# In[114]:


from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d', cmap='YlGnBu',linecolor="white")

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True, cmap='YlGnBu',linecolor="white")



# In[ ]:


# Comparing Models


# In[113]:


models = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'Decision Tree', 'Random Forest'] ,
              
    
    'Score': [acc_log_reg, acc_knn, acc_decision_tree, acc_random_forest]  
              
    })

models.sort_values(by='Score', ascending=False)


# In[ ]:


#From the above table, we can see that Decision Tree and Random Forest classfiers have the highest accuracy score.

#Among these two, we choose Random Forest classifier as it has the ability to limit overfitting as compared to Decision Trees.

