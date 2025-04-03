# import  Modules
import inline
import pandas as pd
import numpy as np
import seaborn as sns
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

#loading the dataset

train = pd.read_csv('train_titanic.csv')
test = pd.read_csv('test_titanic.csv')
# print(train.head())
# profile = ProfileReport(train, title="bao_cao", explorative=True)
# profile.to_file("bao_cao_titanic.html")
# print(train.describe())
# print(train.info())

# exploratory Data Analysis
## categorical attributes
# coi data cot survived
# sns.countplot(x='Survived', data=train)  # Sử dụng tham số 'x' để chỉ định cột
# plt.show()
# coi data cot Pclass
# sns.countplot(x='Pclass', data=train)  # Sử dụng tham số 'x' để chỉ định cột
# plt.show()
# bỏ cột ticket vì nó có dữ liệu số lẫn chữ
# chungs ta se lay cac cot gom Survived , Pclass, Sex, SibSp , Parch , Embarked

## numerical attributes
# sns.displot(x='Age', data=train, kde=True)
# plt.show()
# sns.displot(x='Fare', data=train, kde=True)
# plt.show()
# class_fare = train.pivot_table(index='Pclass', values='Fare')
# class_fare.plot(kind='bar')
# plt.xlabel('Pclass')
# plt.ylabel('AVG. Fare')
# plt.xticks(rotation=0)
# plt.show()
# class_fare = train.pivot_table(index='Pclass', values='Fare', aggfunc=np.sum) tổng danh thu thì cứ dùng aggfunc=np.sum
# class_fare.plot(kind='bar')print(df.head)
# # print(train_len)
# plt.xlabel('Pclass')
# plt.ylabel('total Fare')
# plt.xticks(rotation=0)
# plt.show()

## DATA PREPROCESSING
train_len = len(train)
# combine two df
df = pd.concat([train, test], axis=0)
df = df.reset_index(drop=True)
#

# find the null value
# print(df.isnull().sum())
# index             0
# PassengerId       0
# Survived        418
# Pclass            0
# Name              0
# Sex               0
# Age             263
# SibSp             0
# Parch             0
# Ticket            0
# Fare              1
# Cabin          1014
# Embarked          2

# drop or delete the column
df = df.drop(columns=['Cabin'], axis=1)
# fill missing values using mean of that column
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
# fill missing values using mode of the categorical column
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

## Log transformation for uniform data distribution
# sns.displot(df['Fare'])
# plt.show()

df['Fare'] = np.log(df['Fare']+1)
"""
np.log() là hàm logarit tự nhiên (logarithm cơ số e) từ thư viện NumPy.
Việc thêm +1 đảm bảo rằng không có giá trị nào bằng 0 hoặc nhỏ hơn, vì logarit của 0 hoặc số âm không xác định. Thêm +1 giúp tránh lỗi và vẫn giữ nguyên tính chất của dữ liệu.
Phép biến đổi log này giúp giảm độ lệch của dữ liệu, đặc biệt là với các giá trị lớn trong cột Fare. Điều này làm cho phân phối của dữ liệu trở nên cân đối hơn, có lợi cho các mô hình học máy, đặc biệt là những mô hình nhạy cảm với phân phối dữ liệu, như hồi quy tuyến tính.
"""
# sns.displot(df['Fare'])
# plt.show()

## CORRELATION MATRIX
# corr = df.corr(numeric_only=True)
# plt.figure(figsize=(15,8))
# sns.heatmap(corr, annot=True, cmap='coolwarm',)
# plt.show()

## DROP UNNECESSARY COLUMNS
df = df.drop(columns=['Name', 'Ticket'], axis=1)
# print(df.head())
# PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch      Fare Embarked
# 0            1       0.0       3    male  22.0      1      0  2.110213        S
# 1            2       1.0       1  female  38.0      1      0  4.280593        C
# 2            3       1.0       3  female  26.0      0      0  2.188856        S
# 3            4       1.0       1  female  35.0      1      0  3.990834        S
# 4            5       0.0       3    male  35.0      0      0  2.202765        S

## LABEL ENCODER
from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()

# train slpit data
train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]
target = 'Survived'
x = train.drop(columns=[target, 'PassengerId'], axis=1)
y = train[target]

## build model
from sklearn.model_selection import train_test_split, cross_validate
# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))

    score = cross_validate(model, x, y, cv=5)
    print(score)
    print('CV Score:', np.mean(score['test_score']))

from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# classify(model)
# Accuracy: 0.8071748878923767

from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# classify(model)
# Accuracy: 0.7219730941704036
# CV Score: 0.7654761157491683

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# model = RandomForestClassifier()
# classify(model)
# Accuracy: 0.7892376681614349
# CV Score: 0.8080911430544221

# model = ExtraTreesClassifier()
# classify(model)
# Accuracy: 0.8026905829596412
# CV Score: 0.7957629778419435

from xgboost import XGBClassifier
# model = XGBClassifier()
# classify(model)
# Accuracy: 0.7847533632286996
# CV Score: 0.8148327160881301

from lightgbm import LGBMClassifier
# model = LGBMClassifier(force_col_wise=True)
# classify(model)

from catboost import CatBoostClassifier
# model = CatBoostClassifier()
# classify(model)
# CV Score: 0.8226790534178645

## COMPLÊT MODEL TRAINING WITH FULL DATA

model = LGBMClassifier()
model.fit(x, y)

x_test = train.drop(columns=['PassengerId', 'Survived'], axis=1)

pred = model.predict(x_test)
# print(pred)

# Test Submission

sub = pd.read_csv('gender_submission.csv')

# sub['Survived'] = pred
print("Predictions Length:", len(pred))  # Kiểm tra chiều dài của dự đoán
print("Submission Data Length:", len(sub))  # Kiểm tra chiều dài của DataFrame sub
