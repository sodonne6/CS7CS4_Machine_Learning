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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


## data id 
#Dataset 1:
## # id:2-2-2-1 ##
#Dataset 2:
## ## id:2--2-2-1

#function to read in data and make inital plot- return train test splits
def read_in_data(dataset,dataset_num):
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
    plt.title(f'Data visualisation - Dataset {dataset_num} ')
    plt.legend()
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    #split data into train test sets 0.8:0.2
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X, y, X_train, X_test, y_test, y_train


def log_reg_model_train_eval(X_train, y_train):
    q_val = [1,2,3,4,5]
    c_val = [0.01,0.1,1,10,100,1000]
    #store f1 scores and std_dev
    acc_results_f1 = np.zeros((len(q_val), len(c_val)))
    std_dev_results_f1 = np.zeros((len(q_val), len(c_val)))
    
    acc_results_auc = np.zeros((len(q_val), len(c_val)))
    std_dev_results_auc = np.zeros((len(q_val), len(c_val)))

    #5 fold data
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, q in enumerate(q_val):
        for j, c in enumerate(c_val):
            #fold_scores_f1 = []
            fold_score_auc = []

            ##manual cross val 
            #for train_index, valid_index in cv.split(X_train, y_train):
            #    x_trn, x_val = X_train[train_index], X_train[valid_index]
            #    y_trn, y_val = y_train[train_index], y_train[valid_index]
#
            poly = PolynomialFeatures(degree=q)
            x_trn_poly = poly.fit_transform(X_train)   # fit on fold-train
            #x_val_poly = poly.transform(x_val)
            #f1_scores = cross_val_score()
#
            logRegModel = LogisticRegression(penalty="l2", C=c, solver='liblinear', max_iter=1000)
            #    logRegModel.fit(x_trn_poly, y_trn)
            #    y_val_pred = logRegModel.predict(x_val_poly)
            fold_scores_f1 = cross_val_score(logRegModel,x_trn_poly,y_train,cv=5,scoring='f1')
            #    #calculate f1 score - more appropriate for classification
            #    fold_scores_f1.append(f1_score(y_val, y_val_pred, pos_label=1))
            #    #print(f1_score)
                

            acc_results_f1[i, j] = np.mean(fold_scores_f1)
            std_dev_results_f1[i, j] = np.std(fold_scores_f1)
            #print(acc_results_f1)
            #print(std_dev_results_f1)

    #find the best f1 score achieved and the q and c value associated 
    max_idx = np.unravel_index(np.argmax(acc_results_f1, axis=None), acc_results_f1.shape)
    best_q = q_val[max_idx[0]]
    best_c = c_val[max_idx[1]]
    best_accuracy = acc_results_f1[max_idx]
    print(f"Best polynomial order q: {best_q}, Best regularisation penalty c: {best_c}, with F1 score: {best_accuracy}, with a standard deviation of {std_dev_results_f1[max_idx]}")

    
    ##plot f1 score with for each q/c partnership
    for i, q in enumerate(q_val):
        #plt.plot(c_val, acc_results_f1[i, :], marker='o', label=f'q={q}')
        plt.errorbar(c_val, acc_results_f1[i, :], yerr=std_dev_results_f1[i,:], fmt="-o", capsize=5, label=f"q={q}")
    plt.xscale('log')
    plt.xlabel('Regularisation penalty C (log scale)')
    plt.ylabel('Mean F1 score with 5 fold cross validation')
    plt.title('Model Accuracy for different polynomial orders and regularisation penalties')
    plt.legend()
    plt.grid()
    plt.show()

    return best_q, best_c, acc_results_f1, std_dev_results_f1, q_val, c_val


#TODO - make another plot with best performers and include errors bars because it'll be less noisy now
def plot_best_log_regs(acc_results, std_dev_results, q_vals, c_vals, best_num=3):
    acc_per_q=acc_results.max(axis=1)
    top_acc_q = np.argsort(acc_per_q)[-best_num:][::-1]
    
    plt.figure()
    #for i in top_acc_q:
    q = q_vals[0]
    mean_acc_vals = acc_results[0,:]
    std = std_dev_results[0,:]
    plt.errorbar(c_vals,mean_acc_vals,yerr=std,fmt='-o',capsize=5,label=f"q={q}")
    
    #q = q_vals[10]
    #mean_acc_vals = acc_results[10,:]
    #std = std_dev_results[10,:]
    #plt.errorbar(c_vals,mean_acc_vals,yerr=std,fmt='-o',capsize=5,label=f"q={q}")
    
    q = q_vals[1]
    mean_acc_vals = acc_results[3,:]
    std = std_dev_results[3,:]
    plt.errorbar(c_vals,mean_acc_vals,yerr=std,fmt='-o',capsize=5,label=f"q={q}")
    
    q = q_vals[2]
    mean_acc_vals = acc_results[3,:]
    std = std_dev_results[3,:]
    plt.errorbar(c_vals,mean_acc_vals,yerr=std,fmt='-o',capsize=5,label=f"q={q}")
        
        #plt.plot(c_vals,mean_acc_vals,marker='o,label')
    plt.xscale('log')
    plt.xlabel('Regularisation penalty C (log scale)')
    plt.ylabel('F1 score')
    plt.title(f'Logistic Regression: Top {best_num} performing polynomial degrees by peak CV F1 score')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title='Polynomial degree')
    plt.tight_layout()
    plt.show()

#(b) - train a kNN classifier and use cross val to find best k val
def knn_model_train_eval(X_train, X_test, y_train, y_test,X,y):
    k_val = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    #create array to store accuracy results 
    knn_acc_results = np.zeros(len(k_val))
    knn_std_dev_results = np.zeros(len(k_val))
    for k in k_val:
        #iterate through the k values chosen to test
        knnModel = KNeighborsClassifier(n_neighbors=k)
        #fit model to train data
        #knnModel.fit(X_train, y_train)
        #use cross val to get accuracy - could change this metric in the future
        scores = cross_val_score(knnModel, X_train, y_train, cv=5, scoring='f1')
        #get std dev of accuracy scores
        std_dev = np.std(scores)
        knn_std_dev_results[k_val.index(k)] = std_dev
        knn_acc_results[k_val.index(k)] = np.mean(scores)
        

    plt.figure(figsize=(7, 4))
    plt.errorbar(k_val, knn_acc_results, yerr=knn_std_dev_results, fmt='-o', capsize=3)
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Cross-validated F1 score')
    plt.title('kNN: cross validated F1 score vs k')
    plt.xticks(k_val)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

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
    return y_pred, model, x_test_poly

def train_best_params_knn(X_train,X_test,y_train,y_test,best_k):
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return y_pred , model
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
    rng = np.random.default_rng(seed=42)  # for reproducibility
    baseline_pred_random = rng.choice([-1, 1], size=y_test.shape)

    #compute confusion matrix
    cm_majority = confusion_matrix(y_test, baseline_pred)
    cm_random = confusion_matrix(y_test, baseline_pred_random)

    return cm_majority, cm_random
    
def plot_roc(y_test, y_score=None, conf_matrix=None, label='Model'):
    #this will plot the log reg and knn
    if y_score is not None:
        y_true = np.where(y_test == -1, 0, y_test)  
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f'{label} (AUC={auc(fpr,tpr):.2f})')
    #this will plot single point for the 2 baseline models
    elif conf_matrix is not None:
        TN, FP, FN, TP = conf_matrix.ravel()
        FPR = FP / (FP + TN) if (FP + TN) else 0.0
        TPR = TP / (TP + FN) if (TP + FN) else 0.0
        plt.scatter(FPR, TPR, s=80, label=f'{label} (point)')


#plot og dataset with the y_pred values with it according to report brief
#logic taken from the wk2 assignemtn
def plot_true_vs_pred(X, y, X_plot, y_pred, title="Predictions over True Data"):
    X1, X2 = X[:, 0], X[:, 1]
    X1_pos, X2_pos = X1[y == 1],  X2[y == 1]
    X1_neg, X2_neg = X1[y == -1], X2[y == -1]

    
    plt.figure(figsize=(7, 5))
    plt.scatter(X1_pos, X2_pos, marker='+', label='True +1')
    plt.scatter(X1_neg, X2_neg, facecolors='none', edgecolors='k', label='True -1')
    #prediction data
    plt.scatter(X_plot[y_pred ==1,0],  X_plot[y_pred==1, 1],  marker='x', label='Pred +1', alpha=0.6)
    plt.scatter(X_plot[y_pred ==-1,0], X_plot[y_pred ==-1,1], marker='.', label='Pred -1', alpha=0.6)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.tight_layout() 
    plt.show()





#create main function to call other functions
def main():
    
    print("=================dataset 1==================")
    #read in data and get train test splits
    X, y, X_train, X_test, y_test, y_train = read_in_data('week4',1)

    #train and evaluate logistic regression model
    best_q, best_c, acc_results , std_dev_results, q_vals, c_vals = log_reg_model_train_eval(X_train, y_train)

    #plot best performers
    plot_best_log_regs(acc_results, std_dev_results, q_vals, c_vals, best_num=3)

    #train and evaluate kNN model
    best_k = knn_model_train_eval(X_train, X_test, y_train, y_test,X,y)
    
    #call functions to train best models for both log and knn
    y_pred_log, logModel, x_test_poly = train_best_params_log_reg(X_train,X_test,y_train,y_test,best_q,best_c)
    y_pred_knn, knnModel = train_best_params_knn(X_train,X_test,y_train,y_test,best_k)
    
    #plot y_pred with original data
    # Overlay predictions on original data (Dataset 1)
    plot_true_vs_pred(X, y, X_test, y_pred_log, title="Dataset 1 — Logistic Regression: True Data and Predicted Points")
    plot_true_vs_pred(X, y, X_test, y_pred_knn, title="Dataset 1 — kNN: True Data and Predicted Points")

    
    #get confusion matrices for both
    cm_log, cm_knn = confusion_matrices(y_test, y_pred_log, y_pred_knn)
    print(f"Confusion matrix for Logistic Regression model:\n{cm_log}")
    print(f"Confusion matrix for kNN model:\n{cm_knn}")
    
    
    #get confusion matrices for baseline tests
    cm_most_common, cm_random = baseline_model_conf_mat(y_train,y_test)
    
    print(f"confusion matrix for most frequenct class: \n{cm_most_common}")
    print(f"confusion matrix for random choice: \n{cm_random}")
    
    #get class probabilty
    y_score_log = logModel.predict_proba(x_test_poly)[:, 1]   
    y_score_knn = knnModel.predict_proba(X_test)[:, 1]
    
    # === plot ===
    plt.figure()
    plot_roc(y_test, y_score=y_score_log, label='Logistic Regression')
    plot_roc(y_test, y_score=y_score_knn, label='kNN')
    plot_roc(y_test, conf_matrix=cm_most_common, label='Baseline (Majority)')
    plot_roc(y_test, conf_matrix=cm_random, label='Baseline (Random)')
    plt.plot([0,1],[0,1],'k--') 
    plt.xlabel('FPR') 
    plt.ylabel('TPR') 
    plt.title('ROC curve - Dataset 1') 
    plt.legend(loc='lower right') 
    plt.grid()
    plt.show()
    
    #================dataset 2==================
    print("=================dataset 2==================")
    X, y, X_train, X_test, y_test, y_train = read_in_data('week4_dataset2',2)
    best_q, best_c, acc_results , std_dev_results, q_vals, c_vals = log_reg_model_train_eval(X_train, y_train)
    
    plot_best_log_regs(acc_results, std_dev_results, q_vals, c_vals, best_num=3)
    
    best_k=knn_model_train_eval(X_train, X_test, y_train, y_test,X,y)
    
    #call functions to train best models for both log and knn
    y_pred_log, logModel, x_test_poly = train_best_params_log_reg(X_train,X_test,y_train,y_test,best_q,best_c)
    y_pred_knn, knnModel = train_best_params_knn(X_train,X_test,y_train,y_test,best_k)
    
    #plot y_pred with original data
    # Overlay predictions on original data (Dataset 1)
    plot_true_vs_pred(X, y, X_test, y_pred_log, title="Dataset 2 — Logistic Regression: True Data and Predicted Points")
    plot_true_vs_pred(X, y, X_test, y_pred_knn, title="Dataset 2 — kNN: True Data and Predicted Points")
    
    #get confusion matrices for both
    cm_log, cm_knn = confusion_matrices(y_test, y_pred_log, y_pred_knn)
    print(f"Confusion matrix for Logistic Regression model:\n{cm_log}")
    print(f"Confusion matrix for kNN model:\n{cm_knn}")
    
    cm_most_common, cm_random = baseline_model_conf_mat(y_train,y_test)
    
    print(f"confusion matrix for most frequenct class: \n{cm_most_common}")
    print(f"confusion matrix for random choice: \n{cm_random}")
    
    #get class probabilty
    y_score_log = logModel.predict_proba(x_test_poly)[:, 1]   
    y_score_knn = knnModel.predict_proba(X_test)[:, 1]
    
    # === plot ===
    plt.figure()
    plot_roc(y_test, y_score=y_score_log, label='Logistic Regression')
    plot_roc(y_test, y_score=y_score_knn, label='kNN')
    plot_roc(y_test, conf_matrix=cm_most_common, label='Baseline (Majority)')
    plot_roc(y_test, conf_matrix=cm_random, label='Baseline (Random)')
    plt.plot([0,1],[0,1],'k--') 
    plt.xlabel('FPR') 
    plt.ylabel('TPR')
    plt.title('ROC curve - Dataset 2') 
    plt.legend(loc='lower right') 
    plt.grid() 
    plt.show()
if __name__ == "__main__":
    main()

    







