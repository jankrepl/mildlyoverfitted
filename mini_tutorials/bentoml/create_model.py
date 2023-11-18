import bentoml

from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC(gamma="scale")
clf.fit(X, y)

saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(saved_model)
