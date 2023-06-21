# import data_processing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(vino_features, vino_target, test_size=0.20, random_state=21)

print((len(X_train), len(y_train)))
print((len(X_test), len(y_test)))


logisticRegression = LogisticRegression(solver='liblinear')
logisticRegression.fit(X_train, y_train)

logisticRegression.score(X_test, y_test)