from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, naive_bayes, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

data = pd.read_csv('data.csv')

data = data.drop(columns = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

data.isnull().any()

data.describe()


categories = []
for i in data.columns:
    categories.append(i)
    #print(f"{i} : {data[i].unique()}")
    #print("-----------------------------")



encode = LabelEncoder()
for col in categories:
    if data[col].dtype == 'object':
        data[col] = encode.fit_transform(data[col])
        
        
for co in categories:
    if data[co].dtype != 'object':
        sns.countplot(x = co, hue = 'Attrition', data = data)
        plt.show()



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1)
ax = data.corr().loc["Attrition"].drop("Attrition").sort_values().plot(kind="barh", figsize=(10, 12), ax=ax)
ax.set_title("Attrititon Corelation")
plt.tight_layout()
plt.show()



target = data['Attrition']
features = data.drop(columns = ['Attrition'])



def accuracy(clf, model, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print(f"{model}\n-------------------\n")
        print("Train Result:\n")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train == False:
        pred = clf.predict(X_test)
        print("Test Result:\n")        
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n===========================================\n")


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 42)


clf = svm.SVC(kernel = 'linear', gamma = 2)
clf.fit(X_train,y_train)


accuracy(clf,'SVM:', X_train, y_train, X_test, y_test, train = True)
accuracy(clf,'SVM', X_train, y_train, X_test, y_test, train = False)


clf2 = tree.DecisionTreeClassifier()
clf2.fit(X_train, y_train)

accuracy(clf2,'Decision Tree:', X_train, y_train, X_test, y_test, train = True)
accuracy(clf2,'Decision Tree', X_train, y_train, X_test, y_test, train = False)


clf3 = naive_bayes.GaussianNB()
clf3.fit(X_train, y_train)

accuracy(clf3, 'Naive-bayes:', X_train, y_train, X_test, y_test, train = True)
accuracy(clf3,'Naive-bayes', X_train, y_train, X_test, y_test, train = False)


clf4 = ensemble.RandomForestClassifier(n_estimators = 30)
clf4.fit(X_train, y_train)

accuracy(clf4,'Random Forest:', X_train, y_train, X_test, y_test, train = True)
accuracy(clf4,'Random Forest', X_train, y_train, X_test, y_test, train = False)


clf5 = ensemble.GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1, max_depth = 3, random_state = 0)
clf5.fit(X_train, y_train)


accuracy(clf5,'Gradient Boosting:', X_train, y_train, X_test, y_test, train = True)
accuracy(clf5,'Gradient Boosting', X_train, y_train, X_test, y_test, train = False)


result = []
p3 = []

for t in range(len(X_test)):
    pred1 = clf.predict(X_test.iloc[t,:].values.reshape(1,-1))
    pred2 = clf2.predict(X_test.iloc[t,:].values.reshape(1,-1))
    pred3 = clf3.predict(X_test.iloc[t,:].values.reshape(1,-1))
    pred4 = clf4.predict(X_test.iloc[t,:].values.reshape(1,-1))
    pred5 = clf5.predict(X_test.iloc[t,:].values.reshape(1,-1))

    pred6 = pred1.tolist()+pred2.tolist()+pred3.tolist()+pred4.tolist()+pred5.tolist()
    a = statistics.mode(pred6)
    
    if pred3 == 1 & pred2 == 1:
        pred3.tolist()
        b = statistics.mode(pred3)
        result.append(b)
    else:
        result.append(a)
print(f"Total Accuracy Score: {accuracy_score(y_test, result)}")
print(f"\nConfusion Matrix: \n {confusion_matrix(y_test, result)}")

























