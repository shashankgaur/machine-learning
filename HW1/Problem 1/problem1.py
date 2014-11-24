#!/usr/bin/python
#author: Shashank Gaur, FEUP, UP201309443
#Course: Machine Learning HomeWork1
#The file loads the training data, calculates the mean, convolution and then decides where to plot other data.
import scipy.io
#from sklearn import svm
import numpy
import math
import cmath
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
var = loadmat('trainingset.mat')

X1 = var['X1']
X2 = var['X2']
X3 = var['X3']
XX = var['XX']

meanX1 = numpy.asmatrix(numpy.mean(X1, axis=0))
meanX3 = numpy.asmatrix(numpy.mean(X3, axis=0))
meanX2 = numpy.asmatrix(numpy.mean(X2, axis=0))
covX1 = numpy.cov(X1.T)
covX2 = numpy.cov(X2.T)
covX3 = numpy.cov(X3.T)
print meanX1, meanX2, meanX3, covX1, covX2, covX3
detcovX1 = numpy.linalg.det(covX1)
detcovX2 = numpy.linalg.det(covX2)
detcovX3 = numpy.linalg.det(covX3)
RowXX = XX.shape[0]
ColXX = XX.shape[1]
#plt.plot(X1[:,0], X1[:,1], 'ro',X2[:,0], X2[:,1], 'bo', X3[:,0], X3[:,1], 'yo')
#Plot of decision boundaries
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, .02),
                     numpy.arange(y_min, y_max, .02))

meu1 = 0.4922*(xx**2)+0.4924*(yy**2)-0.0131*xx*yy-4.8463*xx-5.8193*yy+29.5086
#meu12 = 0.0012*(xx**2)-0.0041*(yy**2)-0.0088*xx*yy-2.9078*xx-7.7995*yy+25.6031


#meu1 = meu1.reshape(xx.shape)

plt.contourf(xx, yy, meu1)#, cmap=plt.cm.Paired)
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, .02),
                     numpy.arange(y_min, y_max, .02))

meu2 = 0.4910*(xx**2)+0.4965*(yy**2)-0.0043*xx*yy-1.9385*xx-1.9802*yy+3.9055

#meu2 = meu2.reshape(xx.shape)
plt.contourf(xx, yy, meu2)#, cmap=plt.cm.Paired)
x_min, x_max = X3[:, 0].min() - 1, X3[:, 0].max() + 1
y_min, y_max = X3[:, 1].min() - 1, X3[:, 1].max() + 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, .02),
                     numpy.arange(y_min, y_max, .02))

meu3 = 0.5259*(xx**2)+1.0234*(yy**2)-1.0433*xx*yy-6.2616*xx-12.3001*yy+36.9585
#meu3 = meu3.reshape(xx.shape)
plt.contourf(xx, yy, meu3)#, cmap=plt.cm.Paired )

#Plot of the X1,X2,X3
plt.scatter(X1[:,0], X1[:,1], c='red', marker='o')
plt.scatter(X2[:,0], X2[:,1], c='blue', marker='o')
plt.scatter(X3[:,0], X3[:,1], c='yellow', marker='o')

#Taking decision for XX observations

for x in range(0, RowXX):
    R1 = math.log(1/math.sqrt(detcovX1))-0.5*((XX[x,:]-meanX1)*numpy.linalg.inv(covX1)*((XX[x,:]-meanX1).transpose()))
    R2 = math.log(1/math.sqrt(detcovX2))-0.5*((XX[x,:]-meanX2)*numpy.linalg.inv(covX2)*((XX[x,:]-meanX2).transpose()))
    R3 = math.log(1/math.sqrt(detcovX3))-0.5*((XX[x,:]-meanX3)*numpy.linalg.inv(covX3)*((XX[x,:]-meanX3).transpose()))

    if (R1>R2 and R1>R3):
        plt.plot(XX[x,0], XX[x,1], 'ks', mew=2, ms=6)
    elif (R2>R3 and R2>R1):
        plt.plot(XX[x,0], XX[x,1], 'k^', mew=2, ms=6)
    elif (R3>R2 and R3>R1):
        plt.plot(XX[x,0], XX[x,1], 'k+', mew=2, ms=10)



plt.show()

