
import preprocess
from base_model import  BaseModel

from datetime import datetime
now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
print('Date time ', date_time)

#data_name = 'winequality-white' #'winequality-red'

data_names = ['Frogs_MFCCs_proper', 'winequality-white']

for data_name in data_names:
    dataset_name = data_name+'.csv'
    #X_train, y_train, X_test, y_test = preprocess.load_data('datasets/winequality-red.csv')
    X_train, y_train, X_test, y_test = preprocess.load_data('datasets/'+dataset_name)

    #dt = DecisionTree()
    #dt.train_model(X_train, y_train)
    #dt.test_model(X_test, y_test)

    base_model = BaseModel(data_name=data_name)
    #"#""
    #############################################

    base_model.train_model_dt(X_train, y_train)
    print('Testing.. Best decision tree ..')
    base_model.test_model(X_test, y_test)
    #"#""
    #############################################
    base_model.train_model_nn(X_train, y_train)
    print('Testing.. Best NN model..')
    base_model.test_model(X_test, y_test)

    #############################################
    base_model.train_model_svm(X_train, y_train)
    print('Testing.. Best SVM..')
    base_model.test_model(X_test, y_test)
    #""#"
    #############################################
    base_model.train_model_boost(X_train, y_train)
    print('Testing.. Best Boosting..')
    base_model.test_model(X_test, y_test)

    #"#""
    #############################################
    base_model.train_model_knn(X_train, y_train)
    print('Testing.. Best KNN..')
    base_model.test_model(X_test, y_test)
    
    #"#""