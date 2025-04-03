import pandas as pd
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('StudentScore.xls')
# profile = ProfileReport(df, title="", explorative=True)
# profile.to_file(".html")

# data visualazation
# correlation_matrix = df.corr(numeric_only=True)
#
# plt.figure(figsize=(15,8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', color="red")
# plt.title('correlation Matrix')
# plt.show()

# train split data
from sklearn.model_selection import train_test_split
target = 'math score'
x = df.drop(target, axis=1)
y = df[target]

x_train ,x_test ,y_train , y_test =  train_test_split(x,y , test_size=0.2, random_state=42)

# preprocessing data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
                    "master's degree"]
gender = ["male", "female"]
lunch = x_train["lunch"].unique()
test_prep = x_train["test preparation course"].unique()


num_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])
ord_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy="most_frequent")),
    ('encoder', OrdinalEncoder(categories=[education_values, gender , lunch, test_prep]))
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
])

# BUILD MODEL
from sklearn.linear_model import LinearRegression
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
for i, j in zip(y_test, y_pred):
    print("Actual: {}. Predict: {}".format(i, j))

print(reg.score(x_test, y_test))



