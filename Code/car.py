import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

data = pd.read_csv('car.csv',)
# coi bieu do histogram
#visualizition data
# check = data.hist(figsize=(15,8))
# plt.show()
# coi bieu do phan tan
# check = plt.scatter(data['cylinders'], data['displacement'])
# x = plt.xlabel('cylinders')
# y = plt.ylabel('displacement')
# title = plt.title('Scater plot between x and y')
# plt.show()
# coi biểu đồ phân tán với đường hồi quy tuyến tính bằng seaborn
# sns.lmplot(x='cylinders', y='displacement', data=data)
# plt.title('Scatter plot with Linear Regression line')
# plt.show()
# ma trận tương quan
# correlation_matrix = data.corr(numeric_only=True)
# print(correlation_matrix)
#
# check = plt.figure(figsize=(15, 8))
# map = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# title = plt.title('Correlation Matrix')
# plt.show()
#statistics data
# profile = ProfileReport(data, title="bao_cao_xe", explorative=True)
# profile.to_file("mau_xe.html")

data['brandname'] = data['car name'].str.split('').str.get(0)
"""
bước này là để tạo cột mới chứa tên thương hiệu xe 
str.split('') : chia các chuỗi trong cột 'car name' thành danh sách các từ dựa trên dấu cách ('') vd: Từ danh sách ['ford', 'pinto'], nó sẽ lấy 'ford'.
str.get(0) : lấy chuỗi đầu tiên trong list 
"""

unused_columns = ['car name']
data = data.drop(unused_columns, axis=1)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
"""
buoc này là chuyêển kiểu dữ liệu của horsepower từ text sang numeric cụ thể là float64
"""

# train split

target = "mpg"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

num_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
nom_columns = ['brandname']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, num_columns),
    ("nom_features", nom_transformer, nom_columns)
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model",LinearRegression())
])
"""
Pipeline cho phép ta xâu chuỗi nhiều bước xử lý(như tiền xử lí dữ liệu, biến đổi dữ liệu, và mô hình hoá) thành một quy trình liên tục Mỗi bước được định nghĩa như một cặp tên và đối tượng (ví dụ: ("scaler", StandardScaler())).
có thể fit_transform ở mô hình vì chỉ cần fit 1 lần thì nó sẽ tự động áp dụng lên các câu lệnh có pipeline
"""

# reg.fit(x_train, y_train)
# y_predict = reg.predict(x_test)
# for i, j in zip(y_test, y_predict):
#     print("Actual: {}. Predict: {}".format(i, j))
# print(reg.score(x_test, y_test))
# print("R^2 Score:", r2_score(y_test, y_predict))
# print(mean_squared_error(y_test, y_predict))
# print(mean_absolute_error(y_test,y_predict))