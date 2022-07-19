import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

#untuk dataset
cov = sklearn.datasets.make_spd_matrix(25,random_state=123)
ran = np.random.normal(0,0.1,25)
x = np.random.multivariate_normal(ran,cov,1000)

x_train, x_test = train_test_split(x,test_size=0.5,random_state=123)
print(x_test[0])
