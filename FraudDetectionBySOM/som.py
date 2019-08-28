"""
Fraud Dectection by SOM

Dataset: Contain the data that customers had provide when filling the application form to apply
credit card. All attribute names and values have been changed to meaningless symbols. 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

#Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Training the SOM
#Initialize 10*10 grid, input_len = features number
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#Visualizing the results
bone() #initialize the window
#distance_map: return all Mean Interneuron Distances in one matrix
pcolor(som.distance_map().T)
colorbar() #add legend
#From the output we can find the outlier: white block

markers = ['o', 's'] #o: circle, s: square
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, #x coordinates, 0.5 put in center
         w[1] + 0.5, #y coordinates
         markers[y[i]], #y[i]: label
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2
         )
show()

#Finding the potential fraud
mappings = som.win_map(X)
#Key: Winning node, Value: a list of customer associated with this winning node

frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis=0) #all the potential fraud
frauds = sc.inverse_transform(frauds) #transfer back to original value