from sklearn import tree
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vino_features, vino_target, test_size=0.20, random_state=21)
decisionTree = tree.DecisionTreeClassifier()
decisionTree.fit(X_train, y_train)

decisionTree.score(X_test, y_test)