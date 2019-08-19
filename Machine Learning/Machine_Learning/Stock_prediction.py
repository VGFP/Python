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
        #print and input added for error checking
        for row in islice(csvFileReader, 2195, 2216):
            stock_data=row[0]
            splited_data=re.split(r'\t+', stock_data.rstrip('\t'))
            #print(splited_data)
            dates.append(int(splited_data[2].split('-')[2]))
            #print(dates)
            #input("sdaa")
            prices.append(float(splited_data[6]))
            #print(prices)
            #input("dsadsadas")
    return

def stock_value_prodiction(dates, prices,z):
    dates = numpy.reshape(dates,(len(dates),1))

    # 3 diffrent Support Vector Machines models:

    svr_Linear = sklearn.svm.SVR(kernel='linear',C=1e2)
    svr_Linear.fit(dates,prices)
    svr_Poly = sklearn.svm.SVR(kernel='poly',C=1e2,degree=2)
    svr_Poly.fit(dates,prices)
    svr_rbf=sklearn.svm.SVR(kernel='rbf',C=1e2,gamma=0.1)
    svr_rbf.fit(dates,prices)

    plt= matplotlib.pyplot
    plt.scatter(dates,prices,color='black',label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red',label='RBF model')
    plt.plot(dates, svr_Linear.predict(dates), color='green',label='Linear model')
    plt.plot(dates, svr_Poly.predict(dates), color='blue',label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    print("sdas")
    return svr_rbf.predict(z)[0], svr_Linear.predict(z)[0], svr_Poly.predict(z)[0]

get_data('data/aapl.csv')

predict_price=stock_value_prodiction(dates,prices,[[29]])

print(predict_price)
