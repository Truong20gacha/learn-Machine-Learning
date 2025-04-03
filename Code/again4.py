import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('csgo.csv')
# profile = ProfileReport(data, title="pandas profiling Report", explorative=True)
# profile.to_file("csgo.html")
# list of un-used columns
unused_columns = ['date', 'month', 'year', 'wait_time_s']

# drop un-used columns
df = df.drop(columns=unused_columns)
# get categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
"""
được sủ dụng để khi làm việc với các cột phân loại trong dữ liệu , chẳng hạn như mã hoá (encoding), xử lí các giá trị thiếu , hoặc đơn giản là để xác định chúng trước khi áp dụng các kĩ thuật khác trong phân tích dữ liệu
"""
# unique valúe in categorical columns
for col in ['map', 'result']:
    print(col, df[col].unique())
# map ['Mirage' 'Dust II' 'Cache' 'Overpass' 'Cobblestone' 'Inferno' 'Austria'
#  'Canals' 'Nuke' 'Italy']
# result ['Win' 'Lost' 'Tie']
"""
bước này nhằm mục đích liệt kê các giá trị duy nhất(unique values) trong các cột phân loại là map và result
"""
# df['result'] = df['result'].map({'Win': 1, 'Tie': 0, 'Lost': -1})
"""
câu lệnh map này được sử dụng để ánh xạ (mapping) các giá trị cột 'result' sang các giá trị số
"""
df = pd.get_dummies(df, columns=['map'])
"""
câu lệnh get_dumies() được thực hiện để mã hoá one-hot encoding trên cột 'map' , theo các hình dung thì nó sẽ lấy dữ liêu trong cột map rồi tạo ra các cột riêng dựa trên dữ liệu của map , và trong dữ liệu riêng đó sẽ chứa true và false(trong hoàn cảnh này nó được hiểu là có được đấu trên map đó không)
"""
# plt.figure(figsize=(15,15))
# sns.heatmap(df.corr(), annot=True, fmt=".2f")
# plt.show()
"""
tạo ra biểu đồ ma trận nhiệt để nhận ra mối quan hệ giữa các biến trong df. màu sắc dậm thì bểu thị mức độ tương quan cao , còn nhạt hoặc trung tính thì biểu thị mức độ tương quan thấp hoặc không có tương quan
"""
# train data
# print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1133 entries, 0 to 1132
# Data columns (total 22 columns):
 #   Column           Non-Null Count  Dtype
# ---  ------           --------------  -----
#  0   day              1133 non-null   float64
#  1   match_time_s     1133 non-null   float64
#  2   team_a_rounds    1133 non-null   float64
#  3   team_b_rounds    1133 non-null   float64
#  4   ping             1133 non-null   float64
#  5   kills            1133 non-null   float64
#  6   assists          1133 non-null   float64
#  7   deaths           1133 non-null   float64
#  8   mvps             1133 non-null   float64
#  9   hs_percent       1133 non-null   float64
#  10  points           1133 non-null   float64
#  11  result           1133 non-null   int64
#  12  map_Austria      1133 non-null   bool
#  13  map_Cache        1133 non-null   bool
#  14  map_Canals       1133 non-null   bool
#  15  map_Cobblestone  1133 non-null   bool
#  16  map_Dust II      1133 non-null   bool
#  17  map_Inferno      1133 non-null   bool
#  18  map_Italy        1133 non-null   bool
#  19  map_Mirage       1133 non-null   bool
#  20  map_Nuke         1133 non-null   bool
#  21  map_Overpass     1133 non-null   bool
# dtypes: bool(10), float64(11), int64(1)
# memory usage: 117.4 KB

# Statistics Data
target = 'result'
x = df.drop(columns=[target])
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (906, 21) (227, 21) (906,) (227,)

# Data preprocessing
scaler = StandardScaler()
"""
Nó sẽ chuẩn hóa các đặc trưng sao cho mỗi đặc trưng có trung bình là 0 và độ lệch chuẩn là 1.
"""
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
"""
nhận biết khi nào nên dùng Std và MMS:
StandardScaler: Dùng khi dữ liệu cần có phân phối chuẩn.
MinMaxScaler: Dùng khi dữ liệu cần nằm trong một khoảng giá trị cụ thể.
"""
# build Model
# Logistic Regresion
# params = {
#     "penalty": ['l1', 'l2'],
#     "C": [0.1, 0.5, 1, 5, 10],
#     "solver": ['liblinear'],
#     "max_iter": [100, 200, 300],
#     "class_weight": ['balanced', None],
#     "random_state": [100]
# }
"""
params : là từ điển chứa các siêu tham số cần tối ưu hoá
penalty : đây là loại hình phạt sẽ được áp dụng để tránh overfitting l1 là lasso (thêm giá trị tuyệt đối của các hệ số) và l2 là Ridge(thêm bình phương của các hệ số)
C : siêu tham số điều chỉnh độ mạnh của hình phạt , giá trị càng nhỏ thì hình phạt càng mạnh
solver : chỉ định thuật toán sẽ sử dụng để tối ưu hoá hàm mất mát . liblinear là một solver phù hợp cho các tập dữ liệu nhỏ hoặc cho mô hình l1 hoặc l2.
max_iter: Số lần lặp tối đa để thuật toán hội tụ.
class_weight: Điều chỉnh trọng số của các lớp, có thể là balanced (cân bằng) hoặc không (None).
random_state: Sử dụng để đặt một hạt giống cho sự ngẫu nhiên, giúp kết quả có thể tái lập.

GridSearchCV : là phương pháp tiềm kiếm tren lưới (grid search) được sử dụng để thử nghiệm tất cả các kết hợp có thể có trong các giá trị của param để tìm ra tổ hợp tốt nhất cho mô hình
LogisticRegression(random_state=100): Mô hình hồi quy logistic mà bạn muốn tối ưu hóa.
param_grid=params: Các siêu tham số cần tìm kiếm và tối ưu hóa, được định nghĩa trước đó.
scoring='accuracy': Mức độ chính xác (accuracy) sẽ được sử dụng làm tiêu chí để đánh giá mô hình.
cv=6: Số lượng folds để thực hiện cross-validation (6-fold cross-validation).
verbose=2: Điều chỉnh mức độ chi tiết khi GridSearchCV in ra quá trình thực hiện.
n_jobs=-1: Sử dụng tất cả các lõi CPU khả dụng để tăng tốc độ tìm kiếm.

fit(x_train, y_train): Huấn luyện mô hình bằng cách thử nghiệm tất cả các kết hợp siêu tham số trên tập huấn luyện x_train và y_train. Sau khi hoàn thành, mô hình sẽ chọn ra tổ hợp siêu tham số tốt nhất.
predict(x_test): Sử dụng mô hình đã huấn luyện để dự đoán nhãn cho tập kiểm tra x_test.
"""
# model = GridSearchCV(LogisticRegression(random_state=100), param_grid=params, scoring='accuracy', cv=6, verbose=2, n_jobs= -1)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred))
#                precision    recall  f1-score   support
#
#           -1       0.70      0.74      0.72       105
#            0       0.50      0.10      0.16        21
#            1       0.72      0.80      0.76       101
#
#     accuracy                           0.71       227
#    macro avg       0.64      0.55      0.55       227
# weighted avg       0.69      0.71      0.69       227

# SVM
# model = SVC()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred))

#                 precision    recall  f1-score   support
#
#         Lost       0.70      0.74      0.72       105
#          Tie       0.00      0.00      0.00        21
#          Win       0.72      0.82      0.77       101
#
#     accuracy                           0.71       227
#    macro avg       0.47      0.52      0.50       227
# weighted avg       0.64      0.71      0.67       227

# Random Forest
params = {
    "n_estimators": [50, 100, 200, 300, 400, 500],
    "criterion": ['gini', 'entropy'],
    "max_depth": [10, 20, 30, 40, 50],
    "random_state": [100]
}
"""
n_estimators: Số lượng cây (decision trees) trong rừng (Random Forest). Bạn đang thử nghiệm với các giá trị 50, 100, 200, 300, 400 và 500.
criterion: Tiêu chí để đo độ “thuần khiết” (purity) của các nút (nodes) trong cây. gini và entropy là hai tiêu chí phổ biến, với gini là độ bất bình đẳng Gini và entropy là độ đo thông tin.
max_depth: Chiều sâu tối đa của các cây. Các giá trị được thử nghiệm bao gồm 10, 20, 30, 40, và 50.
random_state: Sử dụng để đặt một hạt giống cho sự ngẫu nhiên, giúp đảm bảo tính tái lập của kết quả.
"""

# model = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring='accuracy', cv=6, verbose=2, n_jobs= -1)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred))
#                 precision    recall  f1-score   support
#
#         Lost       0.80      0.82      0.81       105
#          Tie       1.00      1.00      1.00        21
#          Win       0.81      0.78      0.79       101
#
#     accuracy                           0.82       227
#    macro avg       0.87      0.87      0.87       227
# weighted avg       0.82      0.82      0.82       227

# print(model.best_params_, model.best_score_)
"""
Bước này giúp bạn biết được những giá trị siêu tham số nào mang lại hiệu suất tốt nhất cho mô hình và điểm số tương ứng của nó
model.best_params_: Đây là từ điển (dictionary) chứa các siêu tham số tốt nhất mà GridSearchCV đã tìm ra sau khi thử nghiệm với tất cả các kết hợp của các siêu tham số trong param_grid. Những giá trị này là tổ hợp mà mô hình đạt được hiệu suất cao nhất
model.best_score_: Đây là điểm số tốt nhất mà mô hình đạt được trong quá trình cross-validation khi sử dụng các siêu tham số tốt nhất được lưu trong best_params_. Điểm số này đại diện cho hiệu suất của mô hình khi áp dụng các siêu tham số tối ưu.
ở dưới đây là kết quả in ra
"""
# {'criterion': 'entropy', 'max_depth': 30, 'n_estimators': 200, 'random_state': 100} 0.8123620309050773
# print(confusion_matrix(y_test, y_pred))
"""
bước này để in ra ma trận nhầm lẫn(confusion matrix) gồm y_ thực tế và y_ dự đoán 
ma trận nhầm lẫn thì vố là 2x2 mà ở đaya thì nó là 3x3 do là dữ liệu của cột y là cột result có 3 nhãn là win , tie, lose
ta có thể dùng unique() để kiểm tra dữ liệu trong lớp y
unique_classes = np.unique(y)
print(unique_classes)
print("Number of classes:", len(unique_classes)) và đó là lí do ma trận nhầm lẫn có 3x3
"""
# [[86  0 19]
#  [ 0 21  0]
#  [22  0 79]]


