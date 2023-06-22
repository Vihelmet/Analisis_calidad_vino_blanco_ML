# import variables as v
# import data_processing as dp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(vino_features, vino_target, test_size=0.20, random_state=21)

print((len(X_train), len(y_train)))
print((len(X_test), len(y_test)))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logisticRegression = LogisticRegression(solver='liblinear')
logisticRegression.fit(X_train, y_train)

prediction = logisticRegression.predict(vino_features)
print(prediction)

logisticRegression.score(vino_features, vino_target)