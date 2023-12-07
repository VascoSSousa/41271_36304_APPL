from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df=pd.read_csv("AAPL.csv")

def modelSelection():
    # definir as features que vou utilizar no modelo
    df_features = df.iloc[:, 0:5]
    # definir a variavel target
    df_target = df['Price']

    X_train, X_test, Y_train, Y_test = train_test_split(df_features.values, df_target.values, test_size=0.3)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    print(lr.score(X_test, Y_test))

def modelEvaluation():
    # definir as features que vou utilizar no modelo
    df_features = df.iloc[:, 0:5]
    # definir a variavel target
    df_target = df['Price']

    X_train, X_test, Y_train, Y_test = train_test_split(df_features.values, df_target.values, test_size=0.3)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    print(lr.score(X_test, Y_test))

    y_test_predict = lr.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2 = r2_score(Y_test, y_test_predict)
    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))


'''
    previsao = ['79550', '5.68', '8', '4.09', '23087']
    previsao = np.array([[79550, 5.68, 8, 4.09, 23087]])
    house = pd.DataFrame(previsao)
    pred = lr.predict(house)
    print(pred)

'''

#modelSelection()