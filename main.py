import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import svm


iris=datasets.load_iris()


X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
#print(X.shape)
print("USING SUPPORT VECTOR CLASSIFICATION")

#print(iris.target_names)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf=svm.SVC()

clf.fit(X_train,y_train)

y_predict=clf.predict(X_test)


print(metrics.accuracy_score(y_test, y_predict))
