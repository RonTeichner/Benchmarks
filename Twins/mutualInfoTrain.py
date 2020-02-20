import csv
from mutualInfoTrain_func import *

X, T, Y, y0, y1 = read_csv_data()
scaler, X_train_scaled, X_test_scaled, X_train, T_train, Y_train, T_test, Y_test = create_sets(X, T, Y)
mu_ATT_sLearner = calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, scaler, X, T)
x=3