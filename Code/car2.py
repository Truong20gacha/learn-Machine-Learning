import pandas as pd
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

data = pd.read_csv('car.csv')
# luot so du lieu
# profile = ProfileReport(data, title="", explorative=True)
# profile.to_file(".html")
# print(data.info())
# RangeIndex: 398 entries, 0 to 397
# Data columns (total 9 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   mpg           398 non-null    float64
#  1   cylinders     398 non-null    int64
#  2   displacement  398 non-null    float64
#  3   horsepower    398 non-null    object
#  4   weight        398 non-null    int64
#  5   acceleration  398 non-null    float64
#  6   model year    398 non-null    int64
#  7   origin        398 non-null    int64
#  8   car name      398 non-null    object
# dtypes: float64(3), int64(4), object(2)
# memory usage: 28.1+ KB

# static data
data['brand name'] = data['car name'].str.split(' ').str.get(0)
# print(data['brand name'])
unused_column = ['car name']
data = data.drop(unused_column, axis=1)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

# train split data
target = 'mpg'

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train.shape)
# print(x_test.shape)

# preprocessing data
num_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
nom_columns = ['brand name']

num_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nom_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, num_columns),
    ("nom_feature", nom_transformer, nom_columns)
])

# x_train_processed = preprocessor.fit_transform(x_train)
# x_test_processed = preprocessor.transform(x_test)

# # Kiểm tra dữ liệu sau khi xử lý
# print(pd.DataFrame(x_train_processed).isna().sum())  # Kiểm tra NaN trong x_train đã qua xử lý
# print(pd.DataFrame(x_test_processed).isna().sum())  # Kiểm tra NaN trong x_test đã qua xử lý

# build model
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("model", LinearRegression())
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
for i, j in zip(y_test, y_predict):
    print("Actual: {}. Predict: {}".format(i, j))

print(reg.score(x_test, y_test))



