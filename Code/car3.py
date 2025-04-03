import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

data = pd.read_csv('car.csv')
# kham phá dữ liệu
# profile = ProfileReport(data, title="", explorative=True)
# profile.to_file(".html")

data['brand name'] = data['car name'].str.split(' ').str.get(0)
unused_column = ['car name']
data = data.drop(unused_column, axis=1)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# data visualization
# biểu đồ histogram
# check = data.hist(figsize=(15,8))
# plt.show()
# coi biểu đồ phân tán
# check = plt.scatter(data['weight'], data['displacement'])
# x = plt.xlabel('cylinders')
# y = plt.ylabel('displacement')
# title = plt.title('scatter plot between x and y')
# plt.show()
# coi biểu đồ phân tán với đường hồi quy tuyến tính bằng seaborn
# sns.lmplot(x = 'displacement', y = 'weight', data=data)
# plt.title('scatter plot with linear regression line')
# plt.show()
# ma trận tương quan
# correlation_matrix = data.corr(numeric_only=True)
# check = plt.figure(figsize=(15, 8))
# map = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# title = plt.title('correlation matrix')
# plt.show()

# train data
target = "mpg"
x = data.drop(target, axis=1)
y = data[target]

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2 , random_state=42)

num_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
nom_columns = ['brand name']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

nom_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, num_columns),
    ("nom_features", nom_transformer, nom_columns)
])

# build model

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model",LinearRegression())
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
for i, j in zip(y_test, y_predict):
    print("Actual: {}. Predict: {}".format(i, j))
print(reg.score(x_test, y_test))
print("R^2 Score:", r2_score(y_test, y_predict))
