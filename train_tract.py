import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import h5py
import os



'''
We go through the first 6 files to get our histogram
We are using f_tract because both f_tract and s_tract are highly correlational are
invariably duplication
'''
def show_hist(file):
    df = h5py.File('ESC-50-Processed/tracts/hdf5/2019-10-31/'+file,'r')
    ft = df['f_tract']
    plt.hist(ft, rwidth=0.95)
    plt.show()
    
count = 1
for file in os.listdir('ESC-50-Processed/tracts/hdf5/2019-10-31/'):
    if count == 7: break
    show_hist(file)
    count += 1



def trainer(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return (X_train, X_test, y_train, y_test)


# Visualizing the Training set results
def trained_visualizer(X_train, y_train):
    viz_train = plt
    viz_train.scatter(X_train, y_train, color='red')
    viz_train.plot(X_train, regressor.predict(X_train), color='blue')
    viz_train.title('Energy Vs f_tract, s_tract')
    viz_train.xlabel('f_tract, s_tract')
    viz_train.ylabel('Energy')
    viz_train.show()

# Visualizing the Test set results
def test_visualizer(X_test, y_test):
    viz_test = plt
    viz_test.scatter(X_test, y_test, color='red')
    viz_test.plot(X_train, regressor.predict(X_train), color='blue')
    viz_test.title('Energy Vs f_tract, s_tract')
    viz_test.xlabel('f_tract, s_tract')
    viz_test.ylabel('Energy')
    viz_test.show()

#predicting sound
def predict(df_item):
    return regressor.predict(df_item)

    

#def model_builder(data, final):
#   X_train, X_test, y_train, y_test = train_test_split(data, final, random_state=2)
#  pipeline = make_pipe
