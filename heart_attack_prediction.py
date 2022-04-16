#%% LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn import tree

#%% Readint the dataset
data = pd.read_csv("heart_data.csv")

#%% EDA - 1

print(data.shape) # (303, 14)

describe = data.describe()
print(describe)

print(data.isna().sum())
"""
age         0
sex         0
cp          0
trtbps      0
chol        0
fbs         0
restecg     0
thalachh    0
exng        0
oldpeak     0
slp         0
caa         0
thall       0
output      0
dtype: int64
"""

#%% EDA - 2

sns.pairplot(data,diag_kind="kde",markers="+")

# Correlation Matrix
corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.title("Corrleation Matrix")
plt.show()

#%% To Get X and Y

y = data.output.values
x = data.drop(["output"],axis = 1)


#%% Confusion Matrix Function

def DrawConfusionMatrix(predicted,ML_algo):
    y_pred = predicted
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)

    f,ax = plt.subplots(figsize = (5,5))
    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax =ax)
    plt.xlabel("y_pred")
    plt.title("Confusion Matrix for {0}".format(ML_algo))
    plt.ylabel("y_true")
    plt.show()


#%% Normalization
x =(x - np.min(x)) / (np.max(x) - np.min(x))

#%% Train - Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

#%% Decision Tree
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_predicted_dt = dt.predict(x_test)

print("Accuracy of the Decision Tree => {0}".format(dt.score(x_test,y_test)*100)) # Accuracy of the Decision Tree => 74.72527472527473

# Confusion Matrix of Decision Tree
DrawConfusionMatrix(y_predicted_dt,"Decision Tree Classifier")


#%% Visualize of the Decision Tree

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt,feature_names=data.columns,filled=True)

# Another way +
plt.figure(figsize=(10,10))
plot_tree(dt,feature_names=data.columns,class_names=["0","1"],filled = True)


#%% KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_predicted_knn = knn.predict(x_test)
print("Accuracy of the KNN => {0}".format(knn.score(x_test,y_test)*100)) # Accuracy of the KNN => 75.82417582417582

best_k = math.sqrt(303)
print(best_k)


knn2 = KNeighborsClassifier(n_neighbors=17)
knn2.fit(x_train,y_train)
y_predicteded_knn2 = knn.predict(x_test)
print("Accuracy of the KNN => {0}".format(knn.score(x_test,y_test)*100)) # Accuracy of the KNN => 75.82417582417582




score_list = []

for each in range(1,50):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))


plt.plot(range(1,50),score_list)
plt.title("K-Value & Accuracy")
plt.xlabel("K-Value")
plt.ylabel("Accuracy")
plt.show()

# the higher accuracy at k = 9 for KNN

knn3 = KNeighborsClassifier(n_neighbors=9)
knn3.fit(x_train,y_train)
y_predicteded_knn3 = knn.predict(x_test)
print("Accuracy of the KNN => {0} for k = 9".format(knn3.score(x_test,y_test)*100)) # Accuracy of the KNN => 81.31868131868131 for k = 9


DrawConfusionMatrix(y_predicteded_knn3,"KNN Classifier")
















