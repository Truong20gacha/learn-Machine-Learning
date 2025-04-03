
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv("diabetes.csv")
#code nay dung de xem thong ke tren web tu file data
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("report.html")
# split data
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# chi dung params cho randomforest
params = {
    "n_estimators": [50, 100, 200, 500],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 5],
    "min_samples_split": [2, 5, 10]
}
#chon mo hinh huan luyen
# model = GridSearchCV(RandomForestClassifier( random_state=100), param_grid=params, scoring="precision", cv=6, verbose=2, n_jobs=6)
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)
cls =LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
models,predictions = cls.fit(x_train, x_test, y_train, y_test)

# print(classification_report(y_test, y_predict))
# cm = np.array(confusion_matrix(y_test, y_predict))
# confusion = pd.DataFrame(cm, index=["is_diabetes", "is_healthy"], columns=["is_diabetes", "is_healthy"])
# sns.heatmap(confusion, annot=True)
# plt.show()



