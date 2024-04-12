import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_data(file, split=True):
    df = pd.read_csv(file)

    #print(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y_vals = y.unique()
    y_nums = pd.factorize(y_vals)[0]
    y.replace(to_replace=y_vals, value=y_nums, inplace=True)
    #df.loc
    #df.iloc
    print('X y', X, y)

    if not split:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #print('X_train, X_test, y_train, y_test ', X_train, X_test, y_train, y_test)
    #print('to numpy X_train, X_test, y_train, y_test ', X_train.to_numpy(), X_test.to_numpy(), y_train, y_test)

    return X_train,  y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


def split(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train,  y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()



def encode_data(X_train):
    enc = OneHotEncoder(handle_unknown='ignore',max_categories=10,min_frequency=3,drop='first')
    encoderX = enc.fit(X_train)
    #self.encoder = encoderX
    enc_xTrain = encoderX.transform(X_train)
    return enc_xTrain

def load_encode_data():
    filename = 'datasets/house-prices-advanced-regression-techniques/train.csv'
    df = pd.read_csv(filename)

    categorical_columns = df.select_dtypes(include=['object']).columns
    pd.get_dummies(df, prefix=['col1', 'col2'])
    pd.get_dummies(df, drop_first=True, sparse=False, dummy_na=True)


if __name__ == '__main__':
    filename = 'datasets/house-prices-advanced-regression-techniques/train.csv'
    df = pd.read_csv(filename)
    X_train = df.iloc[:, :-1]
    y_train = df.iloc[:, -1]
    enc_xTrain = encode_data(X_train)
    enc_xTrain_df=pd.DataFrame(enc_xTrain)
    enc_xTrain_df.to_csv('datasets/house-prices-advanced-regression-techniques/enc_XTrain.csv')
