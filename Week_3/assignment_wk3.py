import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso


#load in data
df = pd.read_csv('week3.csv', header=None)
print(df.head())
X1 = df.iloc[:,0].values
X2 = df.iloc[:,1].values
X = np.column_stack([X1, X2])
y = df.iloc[:,2].values
print(X.shape, y.shape)

#plot the dataset on a 3D scatter plot - first feature on x, second on y and target on z
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c='b', marker='o')

ax.set_xlabel("Feature 1 (x1)")
ax.set_ylabel("Feature 2 (x2)")
ax.set_zlabel("Target (y)")
ax.set_title("3D Scatter: x1 vs x2 vs y")

plt.show()

#contour map of y against x1 and x2 to assess if plane or curved surface
plt.figure()
plt.tricontourf(X[:,0], X[:,1], y, levels=12)
plt.scatter(X[:,0], X[:,1], s=12, alpha=0.6)
plt.xlabel("x1"); plt.ylabel("x2"); plt.title("y contours"); plt.colorbar(label="y")
plt.show()

#in addition to the two features in data file add extra polynomial features equal to all combinations of powers 
#of x1 and x2 up to the 5th power
poly = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly.fit_transform(X)
print(X_poly.shape)  #check new shape of the features 
print(poly.get_feature_names_out())  #check the names of new features for the craic

#train lasso regression model on the data with polynomial features with changing c values
#c value should start small enough that all trained coeffs are zero then increase accordingly
c_val = [0.001, 0.01, 0.1, 1, 10, 100] #TO:DO: change these values to ensure it starts with all coeffs zero
for c in c_val: #iterate through the c value array
    lasso_model = Lasso(alpha=c, max_iter=10000) #create lasso model with c value
    lasso_model.fit(X_poly, y) #fit the model to the data
    coeffs = lasso_model.coef_ #get the coeffs
    print(f"Coefficients: {coeffs}\n")
    
    #create a grid of x1 and x2 values for prediction
    x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    x2_range = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    X_grid_poly = poly.transform(X_grid)
    y_pred = lasso_model.predict(X_grid_poly)
    y_pred_grid = y_pred.reshape(x1_grid.shape)
    plt.figure()
    plt.contourf(x1_grid, x2_grid, y_pred_grid, levels=12, cmap='viridis', alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c='r', s=12, alpha=0.6)
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title(f"Lasso Regression Predictions (c={c})"); plt.colorbar(label="Predicted y")
    plt.show()

