import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


#open csv and read first line -. needs to be included in the submission
with open("week2.csv", "r") as f:
    dataset_ID_line = f.readline().strip()
print("Dataset ID line:", dataset_ID_line)   # required for submissio

#code given to read the data set
df = pd.read_csv("week2.csv" , comment='#', header=None)
#print(df.head())
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1 ,X2))
y=df.iloc[:,2]

#a(i) - visulaise data by placing marker on 2D plot 
#marker is a plus if the target value is 1 and and a circle if the target value is -1

X1_pos = X1[y == 1]
X1_neg = X1[y == -1]
X2_pos = X2[y == 1]
X2_neg = X2[y == -1]

plt.scatter(X1_pos, X2_pos, marker='+', color='g', label='Positive')
plt.scatter(X1_neg, X2_neg, marker='o', facecolors='none', edgecolors='b', s=30, linewidths=1, label='Negative')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.title('a(i) - 2D Scatter Plot')
plt.legend()
plt.show()

#a(ii) - train a logistic regression classifier on the data

#penalty - l2 regularization can discourage large weights but doesn't force them to be zero
#C -inverse of regularisation - 1 is standard
model = LogisticRegression(penalty='l2', C=1, solver='liblinear')
model.fit(X, y)

w1, w2 = model.coef_[0] #weights for X1 and X2
b = model.intercept_[0] #bias term

#print weights and biases that were learned 
#p^​(y=1∣x)=σ(b+w1​x1​+w2​x2​)
#p^​(y=1∣x)=1/(1+exp(−(b+w1​x1​+w2​x2​)))
print(f"a(ii) - Weights: w1 = {w1}, w2 = {w2}, Bias: b = {b}")

y_pred = model.predict(X)

#a(iii) - plot the training data and the decision boundary

plt.figure(figsize=(8, 6))

#plot OG data points 
plt.scatter(X1_pos, X2_pos, marker='+', color='g', label='True Positive')
plt.scatter(X1_neg, X2_neg, marker='o', facecolors='none', edgecolors='b', s=30, linewidths=1, label='True Negative')

#plot the predicted points
plt.scatter(X1[y_pred == 1], X2[y_pred == 1], marker='x', color='orange', label='Predicted Positive', alpha=0.5)
plt.scatter(X1[y_pred==-1], X2[y_pred==-1], marker='.', color='purple', label='Predicted Negative', alpha=0.5)

#create decision boundary

#p^(y=1|x) >= 0.5 when b + wx1 + wx2 >= 0

x1_range = np.linspace(X1.min(), X1.max(), 100) #100 evenly spaced points between min and max of X1
#solve for x2
ys = -(w1/w2) * x1_range - (b/w2) #corresponding x2 values for the decision boundary line
plt.plot(x1_range, ys, color='red', linestyle='--', label='Decision Boundary')

# Labels
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Logistic Regression Classifier (a(ii) & a(iii))")
plt.legend()
plt.show()

#b(i) - use sklearn to train svm classifier on the data 
#use LinearSVC in sklearn

#train linear svm classifier for a wind range of values and penealties (C=0.001, C=1, C=100)
#give predictions and reort parameter values

#change this value as needed to test different penalties
C_values = [0.001, 1, 100]

#train and predict and log the weights and bias for C
print(len(C_values))

#TO-DO: loop through C_values and train/predict for each
for i in range(len(C_values)-1):
    svm_model = LinearSVC(C=C_values[i], max_iter=10000)
    svm_model.fit(X, y)
    y_svm_pred = svm_model.predict(X)
    w1_svm, w2_svm = svm_model.coef_[0]
    b_svm = svm_model.intercept_[0]
    print(f"b(i) - SVM Weights: w1 = {w1_svm}, w2 = {w2_svm}, Bias: b = {b_svm}")

    #plot the training and predicted points and decision boundary for SVM
    plt.figure(figsize=(8, 6))
    #plot OG data points
    plt.scatter(X1_pos, X2_pos, marker='+', color='g', label='True Positive')
    plt.scatter(X1_neg, X2_neg, marker='o', facecolors='none', edgecolors='b', s=30, linewidths=1, label='True Negative')
    #plot the predicted points
    plt.scatter(X1[y_svm_pred == 1], X2[y_svm_pred == 1], marker='x', color='orange', label='Predicted Positive', alpha=0.5)
    plt.scatter(X1[y_svm_pred == -1], X2[y_svm_pred == -1], marker='.', color='purple', label='Predicted Negative', alpha=0.5)
    #decision boundary for SVM
    x1_range = np.linspace(X1.min(), X1.max(), 100)
    ys_svm = -(w1_svm/w2_svm) * x1_range - (b_svm/w2_svm)
    plt.plot(x1_range, ys_svm, color='red', linestyle='--', label='SVM Decision Boundary')
    # Labels
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("SVM Classifier C = " + str(C_values[i]) + " (b(i))")
    plt.legend()
    plt.show()
    
#c(i) - create two additional features a=-> add square of each feature (four features in total). Train logistic classifier give the model and the trained parameters values
    
#create and store x1^2 and x2^2
X1_squared = X1 ** 2
X2_squared = X2 ** 2
X_ext = np.column_stack((X1, X2, X1_squared, X2_squared))
#train logistic regression on extended feature set
model_ext = LogisticRegression(penalty='l2', C=1, solver='liblinear')
model_ext.fit(X_ext, y)

w = model_ext.coef_[0] #weights for X1, X2, X1^2, X2^2
b_ext = model_ext.intercept_[0] #bias term
print(f"c(i) - Extended Logistic Regression Weights: w1 = {w[0]}, w2 = {w[1]}, w3 = {w[2]}, w4 = {w[3]}, Bias: b = {b_ext}")
y_ext_pred = model_ext.predict(X_ext)

#plot the true data points and the predicted points
plt.figure(figsize=(8, 6))
#plot OG data points
plt.scatter(X1_pos, X2_pos, marker='+', color='g', label='True Positive')
plt.scatter(X1_neg, X2_neg, marker='o', facecolors='none', edgecolors='b', s=30, linewidths=1, label='True Negative')
#plot the predicted points
plt.scatter(X1[y_ext_pred == 1], X2[y_ext_pred == 1], marker='x', color='orange', label='Predicted Positive', alpha=0.5)
plt.scatter(X1[y_ext_pred == -1], X2[y_ext_pred == -1], marker='.', color='purple', label='Predicted Negative', alpha=0.5)
# Labels
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("Extended Logistic Regression Classifier (c(i))")
plt.legend()
plt.show()
    
    