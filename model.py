import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

insurance = pd.read_csv("insurance.csv")
print(insurance.head())

#String columns are "sex", "smoker" and "region"
#Print the number of unique values of string columns
###print("Number of unique values in sex column: " + str(insurance.sex.nunique()))
###print("Number of unique values in smoker column: " + str(insurance.smoker.nunique()))
###print("Number of unique values in region column: " + str(insurance.region.nunique()))


# Replacing string values to numbers
insurance['sex'] = insurance['sex'].apply({'male':0,      'female':1}.get) 
insurance['smoker'] = insurance['smoker'].apply({'yes':1, 'no':0}.get)
insurance['region'] = insurance['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)

###print(insurance.head())

###print(insurance.columns)

# features
X = insurance[['age', 'sex', 'bmi', 'children','smoker','region']]
# predicted variable
y = insurance['charges']

###print(X.head())
###print(y.head())


# splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

###print(len(X_train))
###print(len(X_test))
###print(len(insurance))

model = LinearRegression()
# Fit linear model by passing training dataset
model.fit(X_train,y_train)
# Predicting the target variable for test datset
predictions = model.predict(X_test)

'''# Predict charges for new customer : Name- Frank
data = {'age' : 40,
        'sex' : 1,
        'bmi' : 45.50,
        'children' : 4,
        'smoker' : 1,
        'region' : 3}
########index = [1]
########frank_df = pd.DataFrame(data,index)
########print(frank_df)'''

pickle.dump(model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[40, 1, 45.5, 4, 1, 3]]))

