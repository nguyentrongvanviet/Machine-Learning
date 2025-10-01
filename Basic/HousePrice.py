import pandas 
import numpy 
from sklearn import linear_model 
numpy.random.seed(42)

fileInput =  pandas.read_csv('train.csv') 

y = numpy.array([fileInput['SalePrice']]).T

a = ['LotArea','OverallQual','ExterQual','Neighborhood','GrLivArea','GarageArea','BsmtQual','YearBuilt','KitchenQual','TotalBsmtSF'] 
X = numpy.array([fileInput[a[0]],fileInput[a[1]],fileInput[a[2]],fileInput[a[3]],fileInput[a[4]],fileInput[a[5]],fileInput[a[6]],fileInput[a[7]],fileInput[a[8]],fileInput[a[9]]]).T
# X = numpy.array([fileInput[a[i]] for i in range(len(a))]).T 


# X = numpy.array([fileInput[fileInput.columns[i]] for i in range(len(fileInput.columns)-1) ]).T
def change(X) : 
    for col in range(X.shape[1]): 
        for D in range(X.shape[0]) : 
            if pandas.isna(X[D][col]) : 
                X[D][col] = 0 
        for D in range(X.shape[0]) : 
            if type(X[D][col]) == str : 
                s = list(set(X[:,col]))
                for row in range(len(X)) : 
                    X[row][col ] = s.index(X[row][col]) 
                break 
    return X 
X = change(X) 
one = numpy.ones((X.shape[0], 1))
Xbar = numpy.concatenate((one, X), axis = 1)

def check(X) : 
    for i in range(X.shape[0]): 
        for j in range(X.shape[1]) :
            X[i][j] = float(X[i][j])

regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

pandas.DataFrame(regr.coef_).to_csv('coef.csv', index=False, header=False)
# print(regr.coef_)
test = pandas.read_csv('test.csv')  

para = numpy.array([test[a[0]],test[a[1]],test[a[2]],test[a[3]],test[a[4]],test[a[5]],test[a[6]],test[a[7]],test[a[8]],test[a[9]]]).T
para = numpy.array([test[test.columns[i]] for i in range(len(test.columns))]).T
para = change(para)
one = numpy.ones((para.shape[0], 1))
para = numpy.concatenate((one, para), axis = 1)
result =  regr.predict(para) 

ID = numpy.array([test['Id']]).T 
result = numpy.concatenate((ID, result), axis = 1); 
result[:, 0] = result[:, 0].astype(int)
pandas.DataFrame(result, columns=['Id', 'SalePrice']).to_csv('result.csv', index=False)