# imports
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# pre process function that takes a file
def preprocess_data(file_name):
    bank_data = pd.read_csv(file_name) # read clean data file (training/testing)

    # drop all features that are not being extracted from web based UI (generally clinical values or not important)
    bank_data = bank_data.drop(['num'], axis=1)  
    bank_data = bank_data.drop(['pID'], axis=1)
    bank_data = bank_data.drop(['file_2'], axis=1)
    bank_data = bank_data.drop(['file_1'], axis=1)
    bank_data = bank_data.drop(['dataset'], axis=1)
    bank_data = bank_data.drop(['afTap'], axis=1)
    bank_data = bank_data.drop(['sTap'], axis=1)
    bank_data = bank_data.drop(['updrs108'], axis=1)
    bank_data = bank_data.drop(['nqScore'], axis=1)

    # encode output variable
    bank_data['gt'] = bank_data['gt'].replace({'FALSE': 0, 'TRUE': 1})  

    # fill any empty data with the median numeric values 
    bank_data.fillna(bank_data.median(numeric_only=True), inplace=True)
    
    return bank_data

# feature selection function that takes data
def feature_selection(bank_data):
    # set y label to the y column, x data to everything else
    y_label = bank_data['gt']
    X_data = bank_data.drop(['gt'], axis=1)

    # determine the most important 4 features and save them as new data in x_data_selected
    # Fit the selector first
    selector = SelectKBest(score_func=f_classif, k=4)
    selector.fit(X_data, y_label)

    # Print the feature scores in order
    print("Feature scores in order:", sorted(zip(X_data.columns, selector.scores_), key=lambda x: x[1], reverse=True))

    # Transform the data
    X_data_selected = selector.transform(X_data)

    joblib.dump(selector, 'feature_selector.joblib')  # save selector here

    # return the 5 most important features and the y column
    return X_data_selected, y_label

# model building and eval function that takes x data and y label
def model_building_and_evaluation(X_data_selected, y_label):

    # call train test split function to split up data into the 4 different parts (x train, x test, y train, y test)
    X_train, X_test, y_train, y_test = train_test_split(X_data_selected, y_label, test_size=0.25, random_state=0, stratify=y_label)

    # setup scaler and X training and testing data splits
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # build models
    best_svm = svm.SVC()
    best_svm.fit(X_train, y_train)

    best_knn = KNeighborsClassifier()
    best_knn.fit(X_train, y_train)

    best_logreg = LogisticRegression()
    best_logreg.fit(X_train, y_train)

    best_rf = RandomForestClassifier()
    best_rf.fit(X_train, y_train)

    best_xgb = XGBClassifier()
    best_xgb.fit(X_train, y_train)

    best_nb = GaussianNB()
    best_nb.fit(X_train, y_train)

    # save the best svm model and scaler
    joblib.dump(best_logreg, 'best_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    return best_svm, best_knn, best_logreg, best_rf, best_xgb, best_nb, X_train, X_test, y_train, y_test, scaler

# model eval function that takes the 6 best results and uses x and y testing data to compare
def model_evaluation(best_svm, best_knn, best_logreg, best_rf, best_xgb, best_nb, X_test, y_test):

    # predict using best models of each type and x test data
    y_pred_svm = best_svm.predict(X_test)
    y_pred_knn = best_knn.predict(X_test)
    y_pred_logreg = best_logreg.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)
    y_pred_xgb = best_xgb.predict(X_test)
    y_pred_nb = best_nb.predict(X_test)

    # print details about each model
    models = {'SVM': y_pred_svm, 'KNN': y_pred_knn, 'Logistic Regression': y_pred_logreg,
              'Random Forest': y_pred_rf, 'XGBoost': y_pred_xgb, 'Naive Bayes': y_pred_nb}

    for model_name, y_pred in models.items():
        print(f"\n{model_name} Accuracy:", accuracy_score(y_test, y_pred))
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
        print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# prediction function for new data
def predict_new_client(predict_file):
    # preprocess the prediction data file 
    predict_data = preprocess_data(predict_file)

    # fill missing values in the predict data using the median
    predict_data.fillna(predict_data.median(numeric_only=True), inplace=True)

    # load the selector and use it to select the x data from the predict data file
    selector = joblib.load('feature_selector.joblib')
    X_data_selected = selector.transform(predict_data.drop(['gt'], axis=1))

    # load the scaler and use it to transform the selected x data 
    scaler = joblib.load('scaler.joblib')
    X_data_selected = scaler.transform(X_data_selected)

    # load the best model and predict based on the x data
    loaded_model = joblib.load('best_model.joblib')
    prediction = loaded_model.predict(X_data_selected)

    # print the prediction result 
    print("\nPrediction for new client:")
    if prediction[0] == 0:
        print("Most likely you do not have PD")
        return "Most likely you do not have PD"
    else:
        print("There is a possibility you have PD")
        return "There is a possibility you have PD"

# main function to call all of the other functions
def main():
    # training data and predict files 
    train_file = "clean.csv"
    predict_file = "new_data.csv"

    # preprocess training data and do feature selection using it and save results
    bank_data = preprocess_data(train_file)
    X_data_selected, y_label = feature_selection(bank_data)

    # call the model building and eval function and save results
    best_svm, best_knn, best_logreg, best_rf, best_xgb, best_nb, X_train, X_test, y_train, y_test, scaler = model_building_and_evaluation(X_data_selected, y_label)

    # call the model evaluation function using the best models and testing data
    model_evaluation(best_svm, best_knn, best_logreg, best_rf, best_xgb, best_nb, X_test, y_test)

    # call the prediction function using the predict file
    predict_new_client(predict_file)

# run the main function
if __name__ == "__main__":
    main()
