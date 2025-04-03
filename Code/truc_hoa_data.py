import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import re

from sklearn.svm import SVC

# Đọc dữ liệu từ các tệp CSV
data2022 = pd.read_csv("weather_data_[2022].csv", sep="\t")
data2023 = pd.read_csv("weather_data_[2023].csv", sep="\t")
data2024 = pd.read_csv("weather_data_[2024].csv", sep="\t")
data = pd.concat([data2022, data2023, data2024], ignore_index=True)

# Các danh sách loại thời tiết
list_sunny = ['Clear', 'Sunny', 'Partly cloudy']
list_rain = ['Light drizzle', 'Light rain', 'Light rain shower', 'Patchy light drizzle', 'Patchy light rain',
             'Patchy light rain with thunder', 'Patchy rain possible', 'Heavy rain', 'Heavy rain at times',
             'Moderate or heavy rain shower', 'Moderate rain', 'Moderate rain at times', 'Overcast',
             'Torrential rain shower']
list_no_cloud = ['Clear', 'Mist']
list_heavy_rain = ['Heavy rain', 'Torrential rain shower', 'Moderate or heavy rain shower']

# Danh sách lưu kết quả cho cột y
y = []
for w in data.weather:
    if w in list_sunny:
        y.append('sunny')
    elif w in list_rain:
        y.append('rain')
    elif w in list_no_cloud:
        y.append('no_cloud')
    elif w in list_heavy_rain:
        y.append('heavy_rain')
    else:
        y.append('unknown')

# Phân chia dữ liệu
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, stratify=y, random_state=0)
x_train_v, x_val, y_train_v, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=0)


# Lớp tiền xử lý
class ColStd(BaseEstimator, TransformerMixin):
    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df, y=None):
        df = X_df.copy()

        # Xử lý cột 'time'
        df['time'] = df['time'].apply(
            lambda x: 'sang' if 6 <= int(re.findall(r'(\d{1,2}):\d{2}', x)[0]) <= 15 else 'toi')

        # Xử lý cột 'month'
        df['month'] = df['month'].apply(lambda x: 'mua' if x in [5, 6, 7, 8, 9, 10, 11] else 'kho')

        # Xử lý cột 'direction'
        df['direction'] = df['direction'].apply(lambda x: x[1:] if len(x) == 3 else x)

        # Phân loại thời tiết
        df['sunny'] = df['weather'].apply(lambda w: 1 if w in list_sunny else 0)
        df['rain'] = df['weather'].apply(lambda w: 1 if w in list_rain else 0)
        df['nocloud'] = df['weather'].apply(lambda w: 1 if w in list_no_cloud else 0)
        df['heavy_rain'] = df['weather'].apply(lambda w: 1 if w in list_heavy_rain else 0)

        # Bỏ cột 'weather' không cần thiết
        return df.drop(columns=['weather'])


# Pipeline tiền xử lý
nume_cols = ['temperature', 'feelslike', 'wind', 'gust', 'cloud', 'humidity', 'pressure']
cate_cols = ['time', 'month', 'direction']

nume_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cate_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(dtype=int, handle_unknown='ignore'))
])

cols_trans = ColumnTransformer(transformers=[
    ('nume', nume_trans, nume_cols),
    ('cate', cate_trans, cate_cols)
])

preprocess_pipeline = Pipeline(steps=[
    ('colstd', ColStd()),
    ('coltrans', cols_trans)
])

# Tiền xử lý tập train và tập val
preprocessed_train_X = preprocess_pipeline.fit_transform(x_train_v)
preprocessed_val_X = preprocess_pipeline.transform(x_val)

# Khảo sát số lượng từng nhãn xem có cân bằng hay không
sunny = 0
rain = 0
nocloud = 0
heavy_rain = 0

# Duyệt qua từng hàng của dữ liệu và đếm số lượng mỗi nhãn
for y in y_train:
    if y == 'sunny':
        sunny += 1
    elif y == 'rain':
        rain += 1
    elif y == 'nocloud':
        nocloud += 1
    elif y == 'heavy_rain':
        heavy_rain += 1

# In ra kết quả
# print(f"sunny: {sunny} rain: {rain} nocloud: {nocloud} heavy_rain: {heavy_rain}")
# sunny: 4608 rain: 1673 nocloud: 0 heavy_rain: 0

# SỬ DỤNG MÔ HÌNH DỰ ĐOÁN

# Khởi tạo pipeline của mô hình
# mlpclf_model = Pipeline(steps=[
#     ('pre', preprocess_pipeline),  # Thay thế preprocess_pipeline bằng pipeline tiền xử lý của bạn
#     ('mlpclf', MLPClassifier(hidden_layer_sizes=(20),activation='tanh', solver='lbfgs', random_state=0, max_iter=500))
# ])
#
# # Các giá trị alpha và hidden_layer_sizes cần dò tìm
# # Danh sách lưu lỗi huấn luyện và kiểm thử
# train_errs = []
# val_errs = []
# iter = []
#
# # Các giá trị alpha và kích thước lớp ẩn cần dò tìm
alphas = [0.001, 0.01, 0.1, 1, 10, 20, 100]
hidden_layer_sizes = [(10), (20), (50), (100)]

# Khởi tạo biến để lưu lỗi kiểm thử tốt nhất và các tham số tốt nhất
best_val_err = float('inf')
best_alpha = None
best_hls = None
#
# # Vòng lặp dò tìm các tham số tốt nhất
for alpha in alphas:
    for hls in hidden_layer_sizes:
        # Lưu tham số hiện tại
        iter.append(f"alpha = {alpha}, hls = {hls}")
        print(iter[-1])

        # Đặt tham số cho mô hình
        mlpclf_model.set_params(mlpclf__alpha=alpha, mlpclf__hidden_layer_sizes=(hls,))

        # Huấn luyện và đánh giá mô hình
        mlpclf_model.fit(x_train_v, y_train_v)
        train_err = 100 * (1 - mlpclf_model.score(x_train_v, y_train_v))
        val_err = 100 * (1 - mlpclf_model.score(x_val, y_val))

        # Lưu lỗi huấn luyện và kiểm thử
        train_errs.append(train_err)
        val_errs.append(val_err)

        # Kiểm tra nếu lỗi kiểm thử hiện tại là tốt nhất
        if val_err < best_val_err:
            best_val_err = val_err
            best_alpha = alpha
            best_hls = hls

# In kết quả tốt nhất
print(f"\nTham số tốt nhất: alpha = {best_alpha}, hidden_layer_sizes = {best_hls}, lỗi kiểm thử = {best_val_err:.2f}%")
# #
# # Tham số tốt nhất: alpha = 10, hidden_layer_sizes = 100, lỗi kiểm thử = 18.14%
#
# # Đổi tên iter thành iter_values để tránh xung đột với hàm tích hợp 'iter'
# for i in range(len(train_errs)):
#     print(f"{iter[i]} \n train_err :{train_errs[i]}\t val_err:{val_errs[i]}")
#     print()
# print(f"best alpha: {best_alpha}")
# print(f"best hidden layer size: {best_hls}")
# best alpha: 10
# best hidden layer size: 100

# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'mlpclf__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
#     'mlpclf__hidden_layer_sizes': [(50,), (100,), (150,), (200,)]
# }
#
# grid_search = GridSearchCV(mlpclf_model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(x_train_v, y_train_v)
#
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best cross-validation score: {grid_search.best_score_}")
# Best parameters: {'mlpclf__alpha': 10, 'mlpclf__hidden_layer_sizes': (150,)}
# Best cross-validation score: 0.8132937380476998

# for i in range(len(train_errs)):
#     if ((i + 1) % 6 == 0 and i != 0):  # Sử dụng 'and' thay vì '&'
#         print(f"{round(train_errs[i], 7)}", end="\n")
#     else:
#         print(f"{round(train_errs[i], 7)}", end="\t")
#
# print()
#
# for i in range(len(val_errs)):
#     if ((i + 1) % 6 == 0 and i != 0):  # Sử dụng 'and' thay vì '&'
#         print(f"{round(val_errs[i], 7)}", end="\n")
#     else:
#         print(f"{round(val_errs[i], 7)}", end="\t")
#train_errs:
#16.3759868	14.1729392	8.0044061	1.1933174	16.2658344	14.0811456
# 8.077841	2.3499174	16.0088122	13.9526345	8.7754727	3.1760602
# 16.4494217	14.5768313	10.2441711	7.2333395	17.3857169	16.8533138
# 16.9634661	16.7798788	18.2669359	18.2852947	18.1567836	18.1384248
# 21.9753993	22.0488342	22.0855517	22.1039104
#val_errs:
# 18.5022026	18.5022026	23.7151248	25.6240822	18.7958884	18.5022026
# 24.1556535	23.8619677	18.722467	19.3098385	22.246696	22.1732746
# 18.2085169	18.2819383	20.5580029	20.7782673	19.1629956	18.4287812
# 18.6490455	18.1350954	19.6769457	19.6035242	19.8237885	19.6035242
# 22.0998532	22.246696	22.246696	22.3201175

# Lấy mô hình với tham số tốt nhất từ GridSearchCV
# best_mlp_model = grid_search.best_estimator_
#
# # Huấn luyện lại mô hình trên tập huấn luyện
# best_mlp_model.fit(x_train_v, y_train_v)
#
# # Dự đoán trên tập kiểm định
# y_pred = best_mlp_model.predict(x_val)

#                 precision    recall  f1-score   support
#
#     no_cloud       0.00      0.00      0.00         3
#         rain       0.74      0.58      0.65       335
#        sunny       0.85      0.96      0.90       922
#      unknown       0.67      0.37      0.48       102
#
#     accuracy                           0.82      1362
#    macro avg       0.56      0.48      0.51      1362
# weighted avg       0.81      0.82      0.81      1362


# In kết quả classification report
# from sklearn.metrics import classification_report
# print(classification_report(y_val, y_pred))

# mlpclf_model.set_params(mlpclf__alpha=best_alpha,mlpclf__hidden_layer_sizes=best_hls)
# mlpclf_model.fit(x_train_v, y_train_v)
# y_pred = mlpclf_model.predict(x_val)
# print(classification_report(y_val, y_pred))

#                 precision    recall  f1-score   support
#
#     no_cloud       0.00      0.00      0.00         3
#         rain       0.74      0.59      0.65       335
#        sunny       0.85      0.96      0.90       922
#      unknown       0.62      0.35      0.45       102
#
#     accuracy                           0.82      1362
#    macro avg       0.55      0.47      0.50      1362
# weighted avg       0.80      0.82      0.80      1362

# MÔ HÌNH SVMCLF

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Khởi tạo mô hình SVM trong pipeline với kernel 'rbf'
svmclf_model = Pipeline(steps=[
    ('pre', preprocess_pipeline),  # Giai đoạn tiền xử lý
    ('svm', SVC(kernel='rbf'))     # SVM với kernel RBF
])

# Danh sách giá trị cho C và gamma để thử nghiệm
c_values = [0.1, 1, 10, 20, 50, 100]
gamma_values = [0.0001, 0.001, 0.01, 0.1, 1]

# Biến lưu lỗi trên tập huấn luyện và kiểm thử
train_errs1 = []
val_errs1 = []
iterations1 = []

# Khởi tạo biến để lưu giá trị lỗi thấp nhất và tham số tốt nhất
best_val_err = float('inf')
best_C = None
best_gamma = None

# Vòng lặp qua các giá trị của C và gamma
for C in c_values:
    for gamma in gamma_values:
        iteration_label = f"C={C}, gamma={gamma}"
        iterations1.append(iteration_label)
        print(f"Đang thử: {iteration_label}")

        # Cập nhật tham số mô hình SVM trong pipeline
        svmclf_model.set_params(svm__C=C, svm__gamma=gamma)

        # Huấn luyện mô hình và tính lỗi trên tập huấn luyện và kiểm thử
        svmclf_model.fit(x_train_v, y_train_v)
        train_err = 100 * (1 - svmclf_model.score(x_train_v, y_train_v))
        val_err = 100 * (1 - svmclf_model.score(x_val, y_val))

        # Ghi lại lỗi cho từng tổ hợp tham số
        train_errs1.append(train_err)
        val_errs1.append(val_err)

        # Cập nhật tham số tốt nhất nếu lỗi kiểm thử hiện tại thấp hơn lỗi tốt nhất trước đó
        if val_err < best_val_err:
            best_val_err = val_err
            best_C = C
            best_gamma = gamma

# # In ra các lỗi cho từng tổ hợp và tham số tốt nhất
# for i in range(len(train_errs1)):
#     print(f"{iterations1[i]} \n train_err: {train_errs1[i]:.4f}\t val_err: {val_errs1[i]:.4f}")
#     print()
# print(f"Best C: {best_C}")
# print(f"Best gamma: {best_gamma}")
# print(f"Lowest validation error: {best_val_err:.4f}%")
# Best C: 50
# Best gamma: 0.1
# Lowest validation error: 18.3554%

svmclf_model.set_params(svm__C=best_C,svm__gamma=best_gamma)
svmclf_model.fit(x_train_v,y_train_v)
y_pred = svmclf_model.predict(x_val)

# print(classification_report(y_val, y_pred))
# #
#               precision    recall  f1-score   support
#
#     no_cloud       0.50      0.33      0.40         3
#         rain       0.74      0.59      0.65       335
#        sunny       0.85      0.95      0.90       922
#      unknown       0.57      0.40      0.47       102
#
#     accuracy                           0.82      1362
#    macro avg       0.67      0.57      0.61      1362
# weighted avg       0.80      0.82      0.81      1362

## 2 mô hình ngang nhau nhưng bên svm nhỉnh hơn nên sẽ chọn bên svm

# Cập nhật siêu tham số tốt nhất cho SVM
svmclf_model.set_params(svm__C=best_C, svm__gamma=best_gamma)

# Huấn luyện lại trên toàn bộ tập train
svmclf_model.fit(x_train, y_train)

# Dự đoán trên tập train
y_pred = svmclf_model.predict(x_train)

# In báo cáo đánh giá
from sklearn.metrics import classification_report
# print(classification_report(y_train, y_pred))
#                precision    recall  f1-score   support

#     no_cloud       0.92      0.69      0.79        16
#         rain       0.84      0.70      0.77      1673
#        sunny       0.89      0.97      0.93      4608
#      unknown       0.86      0.57      0.69       512
#
#     accuracy                           0.88      6809
#    macro avg       0.88      0.73      0.79      6809
# weighted avg       0.87      0.88      0.87      6809

# dự đoán trên tập test
y_pred = svmclf_model.predict(x_test)
# print(classification_report(y_test, y_pred))

#                 precision    recall  f1-score   support
#
#     no_cloud       0.33      0.25      0.29         4
#         rain       0.74      0.59      0.66       419
#        sunny       0.85      0.94      0.89      1152
#      unknown       0.48      0.34      0.40       128
#
#     accuracy                           0.81      1703
#    macro avg       0.60      0.53      0.56      1703
# weighted avg       0.79      0.81      0.80      1703

# Nhận xét
# Độ chính xác của mô mình đạt được mong đợi (>80%)
# Mô hình dự đoán tốt đối với nhãn sunny  (f1 : 0.89)
# Mô hình dự đoán nhãn mưa không tốt lắm, nhưng cũng không hề tệ tuy nhiên độ hiệu quả cũng tương đối -Đưa ra được 65% trường hợp mưa với độ chính xác 78%
# Kết luận
# Dự đoán là khi áp dụng vào thực tế, mô hình hoạt động sẽ không tốt, nhưng không quá tệ
# Có thể sử dụng làm nguồn tham khảo nếu không thể xem dự báo thời tiết vì một lý do nào đó


