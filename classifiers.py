import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

def     data_frame(pair_name):
    folder_path=f"C:\\Users\\manch\\OneDrive\\Documents\\dev\\Trading\\Synthetics\\Step_index\\data_files\\labeled"
    file        =   folder_path+f"\\{pair_name}"
    df          =   pd.read_csv(file,sep=' ')
    return df
df=data_frame(pair_name="dfUltimate_Moving Average Trends_STEP_INDEX_M1_20_20.0_99969.csv")
y=df["Peak"]
df.pop("Peak")
df.pop("Time Stamps")
df.pop("Instance")
X=df
X_train, X_test, y_train, y_test = X[:80000], X[80000:], y[:80000], y[80000:]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train.astype("float64"))


forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train,y_train)
y_pred=forest_clf.predict(X_test)
print("randomforest",classification_report(y_test,y_pred))
"""
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train, y_train)
y_pred=ovr_clf.predict(X_test)
print("ovr",classification_report(y_test,y_pred))
"""
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
y_pred=knn_clf.predict(X_test)
print("KNN",classification_report(y_test,y_pred))

clf=GridSearchCV(KNeighborsClassifier(),{
                                            'n_neighbors':[3,37,69,11],
                                            'weights':['distance'],
                                            'algorithm':['ball_tree', 'brute']
                                         },cv=3,return_train_score=False)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("knnnn",classification_report(y_test,y_pred),clf.best_params_,clf.cv_results_)