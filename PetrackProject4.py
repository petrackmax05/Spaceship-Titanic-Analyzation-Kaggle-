

# Spaceship Titantic
#Kaggle Competition

#Dataset: This dataset contrains passanger information such as demographics, cabin location, and onboard spending.

#Objective: Predict whether a passanger was transported to another dimension.

#Important Takeaways:
#Feature engineering significantly improves performance
#Spending behavior and cabin location are strong predictors
#Handling missing data is critical
#Random forrest provided the most solid classification performance results.


# In[38]:


train = pd.read_csv("C:/Users/Max Petrack/OneDrive/Desktop/train.csv")
test = pd.read_csv("C:/Users/Max Petrack/OneDrive/Desktop/test.csv")


train.head()


# In[39]:


# Age → median
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

# Spending columns
spending_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

for col in spending_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# Cabin
train['Cabin'] = train['Cabin'].fillna("Unknown/0/U")
test['Cabin'] = test['Cabin'].fillna("Unknown/0/U")


# In[40]:


train[['Deck','Num','Side']] = train['Cabin'].str.split('/', expand=True)
test[['Deck','Num','Side']] = test['Cabin'].str.split('/', expand=True)


# In[41]:


spending_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

train['TotalSpend'] = train[spending_cols].sum(axis=1)
test['TotalSpend'] = test[spending_cols].sum(axis=1)


# In[42]:


#The distribution between passangers transported.
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Transported', data=train)
plt.title("Transported Distribution")
plt.show


# In[43]:


#Spending v. Transported
sns.boxplot(x='Transported', y='TotalSpend', data=train)
plt.title("Spending vs Transported")
plt.show()


# In[44]:


#Age Distribution

sns.histplot(train['Age'], bins=30)
plt.title("Age Distribution")
plt.show


# In[45]:


#Deck v. Transported passangers

sns.countplot(x='Deck', hue='Transported', data=train)
plt.title("Deck vs Transported")
plt.show()


# In[46]:


#Removing Unused Columns that were not necessary
train = train.drop(['Name','Cabin'], axis=1)
test = test.drop(['Name','Cabin'], axis=1)


# In[47]:


#Categorial Data setup
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = le.fit_transform(train[col].astype(str))

       
        test[col] = le.fit_transform(test[col].astype(str))

train.dtypes


# In[48]:


#Model Setup

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = train.drop(['Transported', 'PassengerId'], axis=1)
y = train['Transported'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[49]:


#Training Model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)


# In[50]:


#Final Evaluation of model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))


# In[52]:


# Kaggle Submission

test_ids = test['PassengerId']

X_test = test.drop(['PassengerId'], axis=1)

predictions = model.predict(X_test)

output = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": predictions.astype(bool)
})

output.to_csv("submission.csv", index=False)


# In[ ]:


## Conclusion
The model successfully predicted passanger transportation using engineerd features such as spending and cabin location.
Feature engineering significantly improved model performance, demonstrating the importance of structured datasets.
This project highlights how machine learning can be applied to real world classification problems.


# In[ ]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')

