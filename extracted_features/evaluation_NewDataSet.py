####################################################################################################################################################

# Built in 2/9/2018

# Last modification 2/9/2021

#  Developed by: Fathi Mahdi Elsiddig Haroun


####################################################################################################################################################

#### NOTE!! #########################################################################################################################################

# This code is for evaluation ML algorithms response under consumer procssor

# Please make sure that  the CSV file had been modified before run the code

#####################################################################################################################################################

# import LIB'S

from sklearn import datasets
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils.fixes import loguniform
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



def train_vege():
    ################################################################################################################################################################################################################################
    # Import CSV DATA SET

    #realDataFeatures = pd.read_csv('shuffled_FeatureVector_allFeatures.csv') # CSV file path
    realDataFeatures = pd.read_csv('all_FeatureVector_selectedFeature.csv') # CSV file path
    #realDataFeatures = pd.read_csv('shuffled_FeatureVector.csv') # CSV file path ('shuffled data')
    #correlation = realDataFeatures.corr()
    X = realDataFeatures.iloc[:,1:-1].values
    Y = realDataFeatures['class'].values
    X = pd.DataFrame(X)
    #print('data before shuffling: ',X) # for debug only
    #realDataFeatures = pd.DataFrame(realDataFeatures)
    #realDataFeatures =  realDataFeatures.reindex(np.random.permutation(realDataFeatures.index))
    #rows_head = ['mean_r','mean_g','mean_b','mean_h','mean_s','mean_v','std_r','std_g','std_b','std_h','std_s','std_v','var_r','var_g','var_b','var_h','var_s','var_v','energy','ASM','homogeneity','class'] # define the colounm names
    #dat = pd.DataFrame(realDataFeatures) # store the head of the csv file and the data 
    #realDataFeatures.to_csv('shuffled_FeatureVector.csv')  # rename the csv file
    #print('data after shuffling: ',X) # for debug only
    #A = [  'Predicted Class','Actual Class'] # legend
    Y = pd.DataFrame(Y)
    sc = StandardScaler()
    Label_Encoder = LabelEncoder()
    print('\n The dataSet : ', realDataFeatures)
    print('\n DataSet features : ', X)
    print('\n DataSet labels : ', Y)
    Y = Label_Encoder.fit_transform(Y)
    print('\n Labels encoder : ', Y)
    X = sc.fit_transform(X)
    print('\n Standard scalar of features table : ', X)

    ####################################################################################################################################################

    # Split the dataSet

    trn_fet, tst_fet, trn_leb, tst_leb = tts(X, Y, test_size=0.2)

    ###################################################################################################################################################

    # Call classifiers

    svm_clf = svm.SVC(kernel='RBF')
    #print('Model params',svm_clf.get_params)
    #tree_clf = tree.DecisionTreeClassifier()
    #randomForest_clf = RandomForestClassifier()
    knn_clf = KNeighborsClassifier()
    #nb_clf = naive_bayes.GaussianNB()
    #mlp_clf = MLPClassifier(10, activation = 'tanh', solver='sgd')

    ##################################################################################################################################################

    #train classifiers
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']} # for debug only gridsearch
    #param_dist = {'C': loguniform(1e0, 1e3),'gamma': loguniform(1e-4, 1e-3),'kernel': ['rbf'],'class_weight':['balanced', None]} # for random search 
    #n_iter_search = 20 # number of iteration for debug only
    svm_clf = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
    #svm_clf = RandomizedSearchCV(svm_clf, param_distributions=param_dist,n_iter=n_iter_search) # random rearch
    #print('best params before: ', svm_clf.get_params)
    svm_clf.fit(trn_fet, trn_leb)
    #print('best params after: ', svm_clf.get_params)
    print('Best params: ', svm_clf.best_params_)
    #print("RandomizedSearchCV took %.2f seconds for %d candidates" " parameter settings." % ((time() - start), n_iter_search)) report(random_search.cv_results_)
    #saved the trained model
    #Save to file in the current working directory
    joblib_file = "svm_model_test.pkl"
    joblib.dump(svm_clf, joblib_file)
    #tree_clf.fit(trn_fet, trn_leb)
    #randomForest_clf.fit(trn_fet, trn_leb)
    joblib_file_knn = "knn_model.pkl"
    knn_clf.fit(trn_fet, trn_leb)
    joblib.dump(knn_clf, joblib_file_knn)
    #nb_clf.fit(trn_fet, trn_leb)
    #mlp_clf.fit(trn_fet,trn_leb)


    ###############################################################################################################################################################################

    #predictions and evaluation
    joblib_model = joblib.load(joblib_file) # load the model
    start1 = time.time()
    #svm_prediction = svm_clf.predict(tst_fet)
    svm_prediction = joblib_model.predict(tst_fet)
    print('The prediction value is: ',svm_prediction)
    end1 = time.time()
    print('\n SVM prediction response time = ', float(end1-start1))
    print('\n Test Labels = ', tst_leb , ' Prediction = ', svm_prediction)
    print('\n accuracy of SVM is: ',accuracy_score(tst_leb,svm_prediction))
    print(classification_report(tst_leb,svm_prediction))
    print(confusion_matrix(tst_leb,svm_prediction))
    #plt.matshow(confusion_matrix(tst_leb,svm_prediction),fignum='SVM confusion matrix' ,cmap=plt.cm.gray)
    plot_confusion_matrix(svm_clf, tst_fet, tst_leb)
    plt.matshow(confusion_matrix(tst_leb,svm_prediction),fignum='SVM confusion matrix')
    plt.show()
    ## test the saved model
    score = joblib_model.score(tst_fet, tst_leb)
    print("Test score: {0:.2f} %".format(100 * score))
    ## test the prediction twice
    loaded_model = joblib_model.predict(tst_fet)
    print('The prediction of the loaded model is: ',loaded_model) # for debug only
    #start2 = time.time()
    #tree_prediction = tree_clf.predict(tst_fet)
    #end2 = time.time()
    #print('\n Tree prediction response time = ', float(end2-start2))
    #print('\n Test Labels = ', tst_leb , ' Prediction = ', tree_prediction)
    #print('\n accuracy of Tree Classifier is: ',accuracy_score(tst_leb,tree_prediction))
    #print(classification_report(tst_leb,tree_prediction))
    #print(confusion_matrix(tst_leb,tree_prediction))
    #start3 = time.time()
    #randomForest_prediction = randomForest_clf.predict(tst_fet)
    #end3 = time.time()
    #print('\n Random Forest prediction response time = ', float(end3-start3))
    #print('\n Test Labels = ', tst_leb , ' Prediction = ', randomForest_prediction)
    #print('\n accuracy of Forest Classifier is: ',accuracy_score(tst_leb,randomForest_prediction))
    #print(classification_report(tst_leb,randomForest_prediction))
    #print(confusion_matrix(tst_leb,randomForest_prediction))
    start4 = time.time()
    knn_prediction = knn_clf.predict(tst_fet)
    end4 = time.time()
    print('\n KNN prediction response time = ', float(end4-start4))
    print('\n Test Labels = ', tst_leb , ' Prediction = ',  knn_prediction)
    print('\n accuracy of KNN is: ', accuracy_score(tst_leb,knn_prediction))
    print(classification_report(tst_leb,knn_prediction))
    print(confusion_matrix(tst_leb,knn_prediction))
    #start5 = time.time()
    #nb_prediction = nb_clf.predict(tst_fet)
    #end5 = time.time()
    #print('\n Naive bayes prediction response time = ', float(end5-start5))
    #print('\n Test Labels = ', tst_leb , ' Prediction = ', nb_prediction)
    #print('\n accuracy of Naive bayes is: ', accuracy_score(tst_leb,nb_prediction))
    #print(classification_report(tst_leb,nb_prediction))
    #print(confusion_matrix(tst_leb,nb_prediction))
    #start6 = time.time()
    #mlp_prediction = mlp_clf.predict(tst_fet)
    #end6 = time.time()
    #print('\n MLP prediction response time = ', float(end6-start6))
    #print('\n Test Labels = ', tst_leb , ' Prediction = ', mlp_prediction)
    #print('\n accuracy of MLP is: ',accuracy_score(tst_leb,mlp_prediction))
    #print(classification_report(tst_leb,mlp_prediction))
    #print(confusion_matrix(tst_leb,mlp_prediction))
    #plt.matshow( confusion_matrix(tst_leb,mlp_prediction), fignum='MLP confusion matrix',cmap=plt.cm.gray)
    ###########################################################################################################################################################################
    ## ploting and demonstraion ####
    # plt.figure('Histogram of Oriented gradient feature')
    # plt.clf()
    # plt.suptitle('Histogram of Oriented gradient feature')
    # plt.hist(HOG)


    #plt.figure('Data Set Plot')
    #plt.clf()
    #plt.xlabel('Samples')
    #plt.ylabel('Values')
    #plt.suptitle('DataSet')
    #plt.plot(X,'o')
    #plt.plot(Y)
    #plt.legend(L)
    #plt.grid()
    #
    #plt.figure('HOG data')
    #plt.clf()
    #plt.xlabel('Samples')
    #plt.ylabel('Values')
    #plt.suptitle('HOG')
    #plt.plot(HOG,'o')
    #plt.plot(Y*max(HOG))
    #plt.grid()
    #
    #plt.figure('STD BLUE')
    #plt.clf()
    #plt.xlabel('Samples')
    #plt.ylabel('Values')
    #plt.suptitle('STD BLUE')
    #plt.plot(BLUE,'o')
    #plt.plot(Y*max(BLUE))
    #plt.grid()

    #plt.figure('HOG Correlation')
    #plt.clf()
    #plt.xlabel('Samples')
    #plt.ylabel('Values')
    #plt.suptitle('HOG Correlation')
    #plt.plot(C,'o')
    #plt.grid()

    #plt.figure('Colors  Correlation')
    #plt.clf()
    #plt.xlabel('Samples')
    #plt.ylabel('Values')
    #plt.suptitle('Colors Correlation')
    #plt.plot(correlation['STDR'].sort_values(ascending=False),'o')
    #plt.grid()

    # plt.figure('Histogram of the Data Set')
    # plt.clf()
    # plt.suptitle('Histogram of the Data Set')
    # plt.hist(X)
    # plt.legend(L)
    # plt.grid()

    # plt.figure('Training Features Plot')
    # plt.clf()
    # plt.xlabel('Samples')
    # plt.ylabel('Values')
    # plt.suptitle('Training Features ')
    # plt.plot(trn_fet,'o')
    # plt.grid()

    # plt.figure('Testing Features Plot')
    # plt.clf()
    # plt.suptitle('Testing Features')
    # plt.plot(tst_fet,'o')
    # plt.grid()

    # plt.figure('Tree Classification Plot')
    # plt.clf()
    # plt.suptitle('Tree Classification Accuracy')
    # plt.plot(tree_prediction,'o')
    # plt.plot(tst_leb,'x')
    # plt.legend(A)
    # plt.grid()
    # plt.xlabel('LABELS')
    # plt.ylabel('CLASSES')

    # plt.figure('Random Forest Classification Plot')
    # plt.clf()
    # plt.suptitle('Random Forest Classification Accuracy')
    # plt.plot(randomForest_prediction,'o')
    # plt.plot(tst_leb,'x')
    # plt.legend(A)
    # plt.grid()
    # plt.xlabel('LABELS')
    # plt.ylabel('CLASSES')
    #
    #plt.figure('SVM Classification Plot')
    #plt.clf()
    #plt.suptitle('SVM Classification Accuracy')
    #plt.plot(svm_prediction,'o')
    #plt.plot(tst_leb,'x')
    #plt.legend(A)
    #plt.grid()
    #plt.xlabel('LABELS')
    #plt.ylabel('CLASSES')
    #
    # plt.figure('KNN Classification Plot')
    # plt.clf()
    # plt.suptitle('KNN Classification Accuracy')
    # plt.plot(knn_prediction,'o')
    # plt.plot(tst_leb,'x')
    # plt.legend(A)
    # plt.xlabel('LABELS')
    # plt.ylabel('CLASSES')
    # plt.grid()

    # plt.figure('Naive Bayes Classification Plot')
    # plt.clf()
    # plt.suptitle('Naive Bayes Classification Accuracy')
    # plt.plot(nb_prediction,'o')
    # plt.plot(tst_leb,'x')
    # plt.legend(A)
    # plt.grid()
    # plt.xlabel('LABELS')
    # plt.ylabel('CLASSES')

    #plt.figure('MLB Classification Plot')
    #plt.clf()
    #plt.suptitle('MLP Classification Accuracy')
    #plt.plot(mlp_prediction,'o')
    #plt.plot(tst_leb,'x')
    #plt.legend(A)
    #plt.xlabel('LABELS')
    #plt.ylabel('CLASSES')
    #plt.grid()
    # testting
    #extracted_feature = pd.read_csv('data\\sample_FeatureVector_1211.csv') # CSV file path
    #data = extracted_feature.iloc[:,1:-1].values # extract the feature vector from the csv file
    #data = pd.DataFrame(data) #convert the feature vector into padas dataframe
    #print('\n DataSet features : ', data)
    #sc = StandardScaler()
    #data = sc.fit_transform(data)
    #print('\n Standard scalar of features table : ', data)
    #svm_prediction = svm_clf.predict(data)
    #print('The testing value is: ',svm_prediction)
    #plt.show()
    return svm_clf
    #################################################################################################################################################################################


### main
train_vege()

                                                                                      
