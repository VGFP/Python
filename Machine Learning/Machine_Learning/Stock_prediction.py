import csv
import numpy
import re
import sklearn.svm
import matplotlib.pyplot
from itertools import islice

dates = []
prices=[]

def get_data(stock_data_file):
    with open(stock_data_file, 'r') as csvfile:
        csvFileReader=csv.reader(csvfile)
        next(csvFileReader)
        #Load only 30 days of data
        for row in islice(csvFileReader, 2195, 2216):
            stock_data=row[0]
            splited_data=re.split(r'\t+', stock_data.rstrip('\t'))
            
            dates.append(int(splited_data[2].split('-')[2]))
            
            #Make sure to load the correct column as prices
            prices.append(float(splited_data[6]))
            
    return

def stock_value_prodiction(dates, prices,z):
    #reshape
    dates = numpy.reshape(dates,(len(dates),1))

    # 3 diffrent Support Vector Reggresion models:
    #Linear
    Linear = sklearn.svm.SVR(kernel='linear',C=1e2)
    Linear.fit(dates,prices)
    #
    Poly = sklearn.svm.SVR(kernel='poly',C=1e2,degree=2)
    Poly.fit(dates,prices)
    #
    rbf=sklearn.svm.SVR(kernel='rbf',C=1e2,gamma=0.1)
    rbf.fit(dates,prices)

    plt= matplotlib.pyplot
    plt.scatter(dates,prices,color='black',label='Data')
    plt.plot(dates, rbf.predict(dates), color='brown',label='RBF model')
    plt.plot(dates, Linear.predict(dates), color='red',label='Linear model')
    plt.plot(dates, Poly.predict(dates), color='cyan',label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return rbf.predict(z)[0], Linear.predict(z)[0], Poly.predict(z)[0]
#
get_data('data/aapl.csv')

predict_price=stock_value_prodiction(dates,prices,[[31]])

print(predict_price)
