import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ydata_profiling import ProfileReport
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="bao cao du lieu phan tich", explorative=True)
# profile.to_file("bao_cao_phan_tich.html")

# split data
target = ["Outcome"]
x = data.drop(target, axis=1)
y = data[target].values.ravel()  # Chuyển y về mảng 1 chiều nên ghi nhớ


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# build model
# model = SVC()
# model = LogisticRegression()
param = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"]
}
model = GridSearchCV(RandomForestClassifier(random_state=100), param_grid=param, scoring="recall", cv=6, verbose=2)
model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)

y_predict = model.predict(x_test)
# for i,j in zip(y_predict, y_test):
#     print("Predict: {}. Actual: {}".format(i,j))

# đánh giá hiệu xuất mô hình
# print(classification_report(y_test, y_predict))
"""
 dây là kết quả của svc
                 precision    recall  f1-score   support

            0       0.77      0.83      0.80        99
            1       0.65      0.56      0.60        55

     accuracy                           0.73       154
    macro avg       0.71      0.70      0.70       154
 weighted avg       0.73      0.73      0.73       154

 weighted avg = 0.73 = ( 0.77 * 99 + 0.65 * 55)/154
 macro avg : trung bình không trọng số
 weighted avg : trung bình có trọng số
 khi đánh giá mô hình thì nhìn vào recall , recall canàng thấp thì mô hình càng giảm vd:0.56
cm = np.array(confusion_matrix(y_test, y_predict))

 confusion matrix : là ma trận nhầm lẫm
                   Predicted Positive	Predicted Negative
 Actual Positive	True Positive (TP)	False Negative (FN)
 Actual Negative	False Positive (FP)	True Negative (TN)

[[82 17] : hàng là giá trị thực tế , cột là giá trị dự đoán . ví dụ trong con số này trong 99 người bị bệnh thì có 82 ngươời được dự đoán đúng là không bị bệnh
 [24 31]] 
"""
# confusion = pd.DataFrame(cm, index=["is_healthy", "is_diabetes"], columns=["is_diabetes", "is_healthy"])
# sns.heatmap(confusion, annot=True)
# plt.show()
"""
daay la ket qua cua mô hình phi tuyến tính LogisticRegression
                precision    recall  f1-score   support

            0       0.81      0.80      0.81        99
            1       0.65      0.67      0.66        55

     accuracy                           0.75       154
    macro avg       0.73      0.74      0.73       154
 weighted avg       0.76      0.75      0.75       154
"""