import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('height-weight.csv')

# plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")
# plt.show()

# Divide the dataset into independent and dependent
X = df[['Weight']]
y = df[['Height']]

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# Standardize the dataset to scale down to minimize the error
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # For test dataset we always do transform w.r.t X_train and not fit_transform 
plt.scatter(X_train,y_train)
# plt.show()


# Train the model using Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train,y_train)    # This finds out the intercept(theta0) and all the slopes(theta1) values

print(regressor.coef_)          # For coefficient or slope
print(regressor.intercept_)     # For intercept

plt.plot(X_train,regressor.predict(X_train),"r")   # To create best fit line
plt.show()