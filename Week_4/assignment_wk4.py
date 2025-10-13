import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

## data id 
#Dataset 1:
## # id:2-2-2-1 ##
#Dataset 2:
## ## id:2--2-2-1

#function to read in data and make inital plot- return train test splits
def read_in_data(dataset):
    #use week4.csv for the first dataset

    #load in data
    df = pd.read_csv(f'{dataset}.csv', header=None)
    print(df.head())
    X1 = df.iloc[:,0].values
    X2 = df.iloc[:,1].values
    X = np.column_stack([X1, X2])
    y = df.iloc[:,2].values
    print(X.shape, y.shape)

    #split data according to y class label +1 and -1
    X1_pos = X1[y==1]
    X2_pos = X2[y==1]
    X1_neg = X1[y==-1]
    X2_neg = X2[y==-1]


    #visualised data as scatter plot
    plt.figure()
    plt.scatter(X1_pos,X2_pos,c='blue',label='Positive class')
    plt.scatter(X1_neg,X2_neg,c='red',label='Negative class')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Data visualisation')
    plt.legend()
    plt.show()

    #split data into train test sets 0.8:0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train , X_test, y_train, y_test

def log_reg_model_train_eval(X_train, X_test, y_train, y_test):
    #augment two input features with polynomials
    #how to decide degree of polynomial q?
    #-scan across different values of q and do cross validation for each value of q and pick the one with the least error
    q_val = [1,2,3,4,5,6,7,8,9,10,11,12]
    c_val = [0.01,0.1,1,10,100,1000]
    #need to store accuracy results for each (q,c) pair
    #this will store a 2D array with the accuracy results for each q and c_values
    acc_results = np.zeros((len(q_val), len(c_val)))
    std_dev_results = np.zeros((len(q_val), len(c_val)))

    for q in q_val:
        for c in c_val:
            #set polynomial features to current q value
            poly = PolynomialFeatures(degree=q)
            #transform train and test sets
            x_train_poly = poly.fit_transform(X_train)
            x_test_poly = poly.transform(X_test)
            #fit model
            logRegModel = LogisticRegression(penalty = "l2", C=c, solver='liblinear', max_iter=1000)
            #fit data to model
            logRegModel.fit(x_train_poly,y_train)
            coeffs = logRegModel.coef_
            #print(f"Logistic Regression model polynomial order={q} and regularisation penalty = {c} : coeffs: {coeffs}")

            x_test_poly = poly.transform(X_test)

            #need to think of best metric for classification accuracy
            #use cross validation to get accuracy score
            scores = cross_val_score(logRegModel, x_test_poly, y_test, cv=5, scoring='accuracy')

            #also get standard deviation of scores
            std_dev = np.std(scores)
            #store std dev to use as error bars in plots
            std_dev_results[q_val.index(q), c_val.index(c)] = std_dev

            #need to manage how to store results in 2D array
            #rows are q values, columns are c values
            acc_results[q_val.index(q), c_val.index(c)] = np.mean(scores)

    #from the results array i want the q and c combination that gives the highest accuracy
    #then print out the best q and c values
    max_index = np.unravel_index(np.argmax(acc_results, axis=None), acc_results.shape)
    best_q = q_val[max_index[0]]
    best_c = c_val[max_index[1]]
    best_accuracy = acc_results[max_index]
    print(f"Best polynomial order q: {best_q}, Best regularisation penalty c: {best_c}, with accuracy: {best_accuracy}")

    #make plots showing accuracy at different q and c values
    #have lines according to each q value and the x axis is c

    for i, q in enumerate(q_val):
        plt.plot(c_val, acc_results[i, :], marker='o', label=f'q={q}')
        #add error bars - remove for now and maybe use on individual plots
        #plt.errorbar(c_val, acc_results[i, :], yerr=std_dev_results[i, :], fmt='o', capsize=5)
    plt.xscale('log')
    plt.xlabel('Regularisation penalty C (log scale)')
    plt.ylabel('Cross-validated Accuracy')
    plt.title('Model Accuracy for different polynomial orders and regularisation penalties')
    plt.legend()
    plt.grid()
    plt.show()
    #to be used for ROC curve for best model
    return best_q, best_c

#TODO - make individual plots for each q value with error bars


#(b) - train a kNN classifier and use cross val to find best k val
def knn_model_train_eval(X_train, X_test, y_train, y_test):
    k_val = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #create array to store accuracy results 
    knn_acc_results = np.zeros(len(k_val))
    knn_std_dev_results = np.zeros(len(k_val))
    for k in k_val:
        #iterate through the k values chosen to test
        knnModel = KNeighborsClassifier(n_neighbors=k)
        #fit model to train data
        knnModel.fit(X_train, y_train)
        #use cross val to get accuracy - could change this metric in the future
        scores = cross_val_score(knnModel, X_test, y_test, cv=5, scoring='accuracy')
        #get std dev of accuracy scores
        std_dev = np.std(scores)
        knn_std_dev_results[k_val.index(k)] = std_dev
        knn_acc_results[k_val.index(k)] = np.mean(scores)

    #find the best accuracy score in the 2d array and print the k val abd the accuracy
    best_k_index = np.argmax(knn_acc_results)
    best_k = k_val[best_k_index]
    best_k_accuracy = knn_acc_results[best_k_index]
    best_k_std_dev = knn_std_dev_results[best_k_index]
    print(f"Best k value for kNN: {best_k}, with accuracy: {best_k_accuracy} +/- {best_k_std_dev}")
    
    return best_k

def train_best_params_log_reg(X_train,X_test,y_train,y_test,best_q,best_c):
    poly = PolynomialFeatures(degree=best_q)
    x_train_poly = poly.fit_transform(X_train)
    x_test_poly = poly.transform(X_test)
    model = LogisticRegression(penalty = "l2", C=best_c, solver='liblinear', max_iter=1000)
    model.fit(x_train_poly,y_train)
    y_pred = model.predict(x_test_poly)
    return y_pred

def train_best_params_knn(X_train,X_test,y_train,y_test,best_k):
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return y_pred
#calculate confusion matrices for log and knn models with best parameters
def confusion_matrices(y_test, y_pred_log, y_pred_knn):
    cm_log = confusion_matrix(y_test, y_pred_log)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    return cm_log, cm_knn

#create confusion matrices for baseline classifiers (one that always predicts the most popular class in the dataset)
#(one that chooses randomly (1 or -1))
def baseline_model_conf_mat(y_train,y_test):
    #find most popular class 
    if(np.sum(y_train == 1) >= np.sum(y_train == -1)):
        most_freq_class = 1
    else:
        most_freq_class = -1
    baseline_pred = np.full(y_test.shape, most_freq_class)
    
    #random number gen
    # --- 2. Random baseline ---
    # randomly choose between +1 or -1 with equal probability
    rng = np.random.default_rng(seed=42)  # for reproducibility
    baseline_pred_random = rng.choice([-1, 1], size=y_test.shape)

    # --- 3. Compute confusion matrices ---
    cm_majority = confusion_matrix(y_test, baseline_pred)
    cm_random = confusion_matrix(y_test, baseline_pred_random)

    return cm_majority, cm_random
    



#create main function to call other functions
def main():
    
    print("=================dataset 1==================")
    #read in data and get train test splits
    X_train, X_test, y_train, y_test = read_in_data('week4')

    #train and evaluate logistic regression model
    best_q, best_c = log_reg_model_train_eval(X_train, X_test, y_train, y_test)

    #train and evaluate kNN model
    best_k = knn_model_train_eval(X_train, X_test, y_train, y_test)
    
    #call functions to train best models for both log and knn
    y_pred_log = train_best_params_log_reg(X_train,X_test,y_train,y_test,best_q,best_c)
    y_pred_knn = train_best_params_knn(X_train,X_test,y_train,y_test,best_k)
    
    #get confusion matrices for both
    cm_log, cm_knn = confusion_matrices(y_test, y_pred_log, y_pred_knn)
    print(f"Confusion matrix for Logistic Regression model:\n{cm_log}")
    print(f"Confusion matrix for kNN model:\n{cm_knn}")
    
    #get confusion matrices for baseline tests
    cm_most_common, cm_random = baseline_model_conf_mat(y_train,y_test)
    
    #plot ROC curve using conf matrices
    #plot_ROC(cm_log)
    
    #================dataset 2==================
    print("=================dataset 2==================")
    X_train, X_test, y_train, y_test = read_in_data('week4_dataset2')
    log_reg_model_train_eval(X_train, X_test, y_train, y_test)
    knn_model_train_eval(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()

    







