from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import bintrees as bt
#x = [8,12,56,4]
ibt = bt.BinTree()
ibt.insert(8)
ibt.insert(12)
ibt.insert(56)
ibt.insert(4)
print(ibt.root.v)
print(ibt.root.l.v)
print(ibt.root.r.v)
print(ibt.root.r.r.v)

cl_X = np.arange(0, 9).reshape(9,1)
cl_y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
#print('Classification input:\n', X, '\tShape:', cl_X.shape)
#print('Classification labels:\n', y, '\tShape:', cl_y.shape)


# Regression data
r_X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
r_y = np.array([1, 2, 3, 4, 5])
# print('Regression input:\n', r_X, '\tShape:', r_X.shape)
# print('Regression labels:\n', r_y, '\tShape:', r_y.shape)


# Use the following list to test your code for classification
cl_X_test = np.array([[2.1], [5.2], [7.2]])

r_X_test = np.array([[2.1, 5.1], [2.6, 6.2]])

regr = KNeighborsRegressor()
regr.fit(r_X,r_y)
print(regr.predict([[2.1,5.1]]))
print(regr.predict([[2.6,6.2]]))




#difhuaere', 'qzaemueode', 'podugade', 'xda' and 'lqea'
sbt = bt.BinTree()
sbt.insert("difhuaere") #
sbt.insert("qzaemueode")#
sbt.insert("podugade") #
sbt.insert("xda") #
sbt.insert("lqea")
print(sbt.root.v)
print(sbt.root.l.v)
print(sbt.root.r.v)
print(sbt.root.l.l.v)
print(sbt.root.l.l.r.v)