from sklearn import svm
from sklearn import datasets
from joblib import dump, load

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))

# saving and loading models
dump(clf, 'filename.joblib') 
clf = load('filename.joblib') 