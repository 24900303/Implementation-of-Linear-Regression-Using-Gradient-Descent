# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Loading and Preprocessing
2. Extract Features and Target Variable:
3. using StandardScaler. This ensures that the features have zero mean and unit variance, which helps gradient descent converge faster.
4. Implementing Linear Regression with Gradient Descent
5. Initialize the model parameters (weights) theta to zeros. The size of theta will match the number of features (including the intercept term).
6. Set the learning rate (step size for updates) and the number of iterations for the gradient descent process.
7. After the gradient descent converges (i.e., the specified number of iterations is completed), return the final optimized values of theta.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("/content/50_Startups.csv")
print(data.info())
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/111bc302-a62d-4fd8-bf7a-b7cf1998de92)
![Screenshot 2025-03-08 113424](https://github.com/user-attachments/assets/21f11ec6-e59b-42af-94a5-ce4471cdf6b1)
![Screenshot 2025-03-08 113559](https://github.com/user-attachments/assets/6c2d6b1d-2fb7-4aa3-81fd-9e4f34628467)
![Screenshot 2025-03-08 114106](https://github.com/user-attachments/assets/7661160d-e627-44fc-9f04-e4994b6d9977)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
