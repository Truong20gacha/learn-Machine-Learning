import  pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("StudentScore.xls")

# profile = ProfileReport(data, title = "bao cao du lieu", explorative=True)
# profile.to_file("bao_cao_hoc_sinh.html")
"""
bộ dữ liệu này có mối tương quan cao giữa math score, reading score and writing score 
"""
target = "math score"
# bước này chỉ dùng để kiểm tra dữ liệu khi chưa có ydata_profiling
# corr = data[[target, "writing score", "reading score"]].describe()
"""
lệnh corr() dùng để tính ma trận tương quan ngay khi coi dữ liệu trong ydata_profiling
 hệ số tương quan gồm:
 1 : tương quan dương hoàn toàn (thường thì các feature di chuyển cùng chiều)
 0 : không có tương quan (thường thì không liên quan tới nhau)
-1 : tương quan âm hoàn toàn (thường thì các feature di chuyển ngược chiều)
Giá trị gần 0: Biểu thị mối tương quan yếu
Giá trị gần -1 hoặc 1: Biểu thị mối tương quan rất mạnh (có sự liên hệ chặt chẽ với nhau)
Nếu hệ số tương quan giữa hai biến là 0.9, điều đó có nghĩa là khi biến này tăng, biến kia cũng có xu hướng tăng với mối liên hệ mạnh.
Nếu hệ số tương quan là -0.8, điều đó có nghĩa là khi biến này tăng, biến kia có xu hướng giảm với mối liên hệ mạnh.

lệnh describe() dùng  để tính toán các thống kê cơ bản của các cột số liệu trong một DataFrame. Lệnh này giúp bạn có một cái nhìn tổng quan về dữ liệu của mình.
các cột trong describe() gồm:
- Count: Cho biết số lượng giá trị không bị thiếu.
- Mean: Trung bình của các giá trị trong cột.
- Std: Độ lệch chuẩn, phản ánh mức độ phân tán của dữ liệu.
- Min: Giá trị nhỏ nhất trong cột.
- 25%, 50%, 75%: Các phân vị tương ứng, cho thấy phân bố dữ liệu.
- Max: Giá trị lớn nhất trong cột.
"""

x = data.drop(target, axis=1)
"""
Axis=0 thường được hiểu là thực hiện phép toán theo chiều dọc
Axis=1 thường được hiểu là thực hiện phép toán theo chiều ngang
"""
y = data[target]
#split data
x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#preprogressing
# chi dung buoc nay khi co du lieu bi khuyet
# imputer = SimpleImputer(strategy='median')
# x_train = imputer.fit_transform(x_train)
# x_test = imputer.transform(x_test)
"""
đôi khi nên mã hoá 1 số cột vì nó liên quan đêns lợi ích của khách hàng

dropna() dùng để bỏ các ô bị khuyết
Khi nào không nên sử dụng dropna()?
Khi dữ liệu bị thiếu quá nhiều: Nếu bạn loại bỏ quá nhiều hàng hoặc cột, bạn có thể mất đi những thông tin quan trọng.
Khi bạn có giải pháp thay thế: Có thể thay thế giá trị thiếu bằng các kỹ thuật như fillna() (điền giá trị thay thế) thay vì loại bỏ chúng.

SimpleImputer ĐƯỢC sử dụng để xử lý dữ liệu bị thiếu bằng cách điền các giá trị thay thế (imputation)
"""
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
"""
strategy dungf để điền (impute) các giá trị bị thiếu trong dữ liệu tuỳ theo hoàn cảnh của bộ data thì sẽ chia ra gồm :
strategy='mean' (Giá trị trung bình) Sử dụng khi: Dữ liệu có phân phối tương đối đều, không bị lệch về phía nào (ví dụ: phân phối chuẩn).
strategy='median' (Giá trị trung vị) Sử dụng khi: Dữ liệu có phân phối không đối xứng hoặc bị lệch (skewed distribution ? )
strategy='most_frequent' (Giá trị thường gặp nhất) Sử dụng khi: Cột chứa các giá trị dạng danh mục (categorical) hoặc dữ liệu mà giá trị phổ biến nhất có ý nghĩa quan trọng.
strategy='constant' (Giá trị cố định) Sử dụng khi: Bạn muốn thay thế các giá trị bị thiếu bằng một giá trị cố định, chẳng hạn như 0, "missing", hoặc bất kỳ giá trị nào khác có ý nghĩa cụ thể cho dữ liệu của bạn.
imputer = SimpleImputer(strategy='constant', fill_value=0)

"""
# BƯỚC DÙNG pipeline này dùng để ghi "tắt" khi thay vì fit_transform và transform nhiều lần
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
gender = ["male", "female"]
lunch = x_train["lunch"].unique()
test_prep = x_train["test preparation course"].unique()

"""
lệnh unique() sẽ tìm và trả về một mảng chứa các giá trị duy nhất trong cột này
"""
ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_values, gender, lunch, test_prep]))
])
"""
Pipeline là một công cụ mạnh mẽ trong scikit-learn giúp bạn kết hợp một loạt các bước xử lý dữ liệu và mô hình hóa thành một quy trình duy nhất
không thể dùng 'mean' và 'median' cho dữ liệu dạng số 
có thể dùng most_frequent cho dữ liệu dạng ordinal
categories=[education_values] nếu không ghi cái này trong ordinal encoder thì sẽ mặc định thứ tự theo A -> Z
"""
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=True)),
])
"""
OneHotEncoder là một công cụ biến đổi các biến phân loại (categorical variables) thành các biến nhị phân (binary variables) sử dụng mã hóa one-hot.
Tham số sparse_output trong OneHotEncoder kiểm soát định dạng của ma trận đầu ra.
"""
# ddaay la buoc xu li du lieu dang so o 2 cot reading score va writing score numerical feature
# processed_data = ord_transformer.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])
# for i, j in zip(x_train[["parental level of education", "gender", "lunch", "test preparation course"]].values, processed_data):
#     print("Before {}. After {}".format(i, j))

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"])
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=100))
])

params = {
    "model__n_estimators":[50,100,200,500],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 5],
    "min_samples_split": [2, 5, 10]
}


model = GridSearchCV(reg, param_grid=params, scoring="precision", cv=6, verbose=2, n_jobs=6)

model.fit(x_train, y_train)

# reg.fit(x_train, y_train)
# y_predict = reg.predict(x_test)
# # for i, j in zip(y_test, y_predict):
# #     print("Actual: {}. Predict: {}".format(i, j))
#
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))
"""
MAE và MSE cho thấy mức độ sai lệch giữa giá trị dự đoán và giá trị thực tế.
R² cho biết tỷ lệ phần trăm sự biến thiên trong dữ liệu mà mô hình có thể giải thích.

tại sao khi đánh giá mô hình thì người ta lại lấy y_test , y_predict: thì đơn giản là vì y_test là  các giá trị thực tế mà bạn muốn dự đoán và y_predict là các giá trị mà mô hình dự đoán được còn x thì nó là "đặc trưng" không phải là biến cụ thể nên không lấy
"""
# KẾT QUẢ CỦA LINEAR
# MAE: 4.181966418321513
# MSE: 28.821056563832897
# R2: 0.8815597679452446

# MAE: 4.677958333333334
# MSE: 37.275797902777775
# R2: 0.846814979046555


