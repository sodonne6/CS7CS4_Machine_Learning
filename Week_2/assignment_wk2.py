import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


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