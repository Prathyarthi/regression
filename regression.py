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


# Predicting the test results
y_pred = regressor.predict(X_test)
print(y_pred,y_test)
# test_dataset
# plt.scatter(X_test,y_test)
# plt.plot(X_test,regressor.predict(X_test),"r")
# plt.show()



# Performance metrics -> R squared and adjusted R squared
# Cost functions -> MSE, MAE, RMSE

from sklearn.metrics import mean_squared_error,mean_absolute_error
mse = mean_squared_error(y_test,y_pred)   # mse between y_test and y_pred
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(mse)
print(mae)
print(rmse)

# To get the score of the model:
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
print("Accuracy :",score)


# For any new data point, let's say 82
new_data = [[82]]
w = scaler.transform(new_data)
print("The predicted height for the weight 82 is:",regressor.predict([w[0]]))

  