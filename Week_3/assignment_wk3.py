import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


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
#c_val = [0.001, 0.01, 0.1, 1, 10, 100] #TO:DO: change these values to ensure it starts with all coeffs zero
c_val = [100,10,1,0.1,0.01,0.001,0.0001] #changed these values to ensure it starts with all coeffs zero
#for lasso the bigger alpha (c) the more coeffs are zero

#store models here for use later
models = []

for c in c_val: #iterate through the c value array
    lasso_model = Lasso(alpha=c, max_iter=10000) #create lasso model with c value
    lasso_model.fit(X_poly, y) #fit the model to the data
    coeffs = lasso_model.coef_ #get the coeffs
    print(f"Coefficients: {coeffs}\n")
    
    #store model currently used
    models.append((c, lasso_model))
    
    
    #create a grid of x1 and x2 values for prediction
    #x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    #x2_range = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    #x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    #X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    #X_grid_poly = poly.transform(X_grid)
    #y_pred = lasso_model.predict(X_grid_poly)
    #y_pred_grid = y_pred.reshape(x1_grid.shape)
    #plt.figure()
    #plt.contourf(x1_grid, x2_grid, y_pred_grid, levels=12, cmap='viridis', alpha=0.8)
    #plt.scatter(X[:,0], X[:,1], c='r', s=12, alpha=0.6)
    #plt.xlabel("x1"); plt.ylabel("x2"); plt.title(f"Lasso Regression Predictions (c={c})"); plt.colorbar(label="Predicted y")
    #plt.show()
    
#for each model generate predicitions for the target variable.
#Generate these predictions on a grid of feature values
#use some nested for loops here is one provided for me to use

#all models have been stored in a list with their c value
Xtest = []
grid=np.linspace(-5,5)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)
#expand with polynomial transformer
Xtest_poly = poly.transform(Xtest)

#plot the surface for each model
X1grid, X2grid = np.meshgrid(grid, grid)

gridLen = len(grid)

for c, lasso_model in models:
    y_pred = lasso_model.predict(Xtest_poly)
    y_pred_shaped= y_pred.reshape(gridLen, gridLen)

    fig = plt.figure()                      
    ax = fig.add_subplot(111, projection='3d')        
    model_surface=ax.plot_surface(X1grid, X2grid, y_pred_shaped, alpha=0.55, linewidth=0)
    ax.scatter(X[:,0], X[:,1], y, s=15, c='r',label='Training data')  # also plot the training points
    model_surface.set_label("predicted surface")
    ax.legend()
    ax.set_xlabel("x1"); 
    ax.set_ylabel("x2"); 
    ax.set_zlabel("predicted y")
    ax.set_title(f"Lasso surface (alpha={c})")
    #ax.view_init(elev=22, azim=35)
    views = [(20,35), (10,35), (20,0), (10,0)]
    for elev, az in views:
        ax.view_init(elev=elev, azim=az)
        plt.draw(); 
        plt.pause(0.6)                  # short delay so you can see it

    plt.show()
    plt.close(fig)

#(d) - explain this part in report 

#(e) - redo but with redige regressing instead of lasso regression
#add extra polynomials
#train ridge models with different c values
#gnerate predictions and plot surfaces

#c values for ridge regression - smaller values mean more regularisation
#reuse c_val from before but iterate backwards
#use x_poly from before
c_val_ridge = c_val[::-1] #reverse the order of c values
models_ridge = [] #store ridge models here
for c in c_val_ridge:
    ridge_model = Ridge(alpha=c, max_iter=10000) #ridge model with changing c value 
    ridge_model.fit(X_poly, y) #fit the model to the data
    coeffs = ridge_model.coef_ #get the coeffs
    print(f"Ridge Coefficients (c={c}): {coeffs}\n")
    models_ridge.append((c, ridge_model)) #store the model
    
#get predictions and plot the surfaces for each model
Xtest = []
grid=np.linspace(-5,5)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)
#expand with polynomial transformer
Xtest_poly = poly.transform(Xtest)

#plot the surface for each model
X1grid, X2grid = np.meshgrid(grid, grid)

gridLen = len(grid)

for c, ridge_model in models_ridge:
    y_pred = ridge_model.predict(Xtest_poly)
    y_pred_shaped= y_pred.reshape(gridLen, gridLen)

    fig = plt.figure()                      
    ax = fig.add_subplot(111, projection='3d')        
    model_surface=ax.plot_surface(X1grid, X2grid, y_pred_shaped, alpha=0.55, linewidth=0)
    ax.scatter(X[:,0], X[:,1], y, s=15, c='r',label='Training data')  # also plot the training points
    model_surface.set_label("predicted surface")
    ax.legend()
    ax.set_xlabel("x1"); 
    ax.set_ylabel("x2"); 
    ax.set_zlabel("predicted y")
    ax.set_title(f"Ridge surface (alpha={c})")
    #ax.view_init(elev=22, azim=35)
    views = [(20,35), (10,35), (20,0), (10,0)]
    for elev, az in views:
        ax.view_init(elev=elev, azim=az)
        plt.draw(); 
        plt.pause(0.6)                  # short delay so you can see it

    plt.show()
    plt.close(fig)  

#use 5 fold cross validation to plot mean and standard dev of the predicition error vs C
#use errorbar funtion
mean_errors_lasso = []
std_errors_lasso = []
mean_errors_ridge = []
std_errors_ridge = []
for c in c_val:
    lasso_model = Lasso(alpha=c, max_iter=10000)
    #cross_val_score uses negative MSE
    neg_mse_scores = cross_val_score(lasso_model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    #convert to pos
    mse_scores = -neg_mse_scores
    ##append to array 
    mean_errors_lasso.append(np.mean(mse_scores))
    std_errors_lasso.append(np.std(mse_scores))
    
    #do the same for ridge
for c in c_val:
    ridge_model = Ridge(alpha=c, max_iter=10000)
    #cross_val_score uses negative MSE
    neg_mse_scores = cross_val_score(ridge_model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    #convert to pos
    mse_scores = -neg_mse_scores
    ##append to array 
    mean_errors_ridge.append(np.mean(mse_scores))
    std_errors_ridge.append(np.std(mse_scores))
    

#make the plots 
plt.figure()
plt.errorbar(c_val, mean_errors_lasso, yerr=std_errors_lasso, fmt='-o', label='Lasso', capsize=5)
plt.errorbar(c_val, mean_errors_ridge, yerr=std_errors_ridge, fmt='-s', label='Ridge', capsize=5)
plt.xscale('log')
plt.xlabel('C (alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs C for Lasso and Ridge Regression')
plt.legend()
plt.show()
