import pandas as pd
import preprocessor
from sklearn.ensemble import RandomForestRegressor
from ydata_profiling import ProfileReport
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("StudentScore.xls")
target = "math score"
# profile = ProfileReport(data, title="Student Score Report", explorative=True)
# profile.to_file("student.html")


x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# processed data : tien xu ly du lieu
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree",
                    "master's degree"]
gender = ["male", "female"]
lunch = x_train["lunch"].unique()
test_prep = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_values, gender, lunch, test_prep])),
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# params = {
#     "model__n_estimators": [50, 100, 200],
#     "model__criterion": ["squared_error", "absolute_error", "poisson"],
#     "model__max_depth": [None, 2, 5],
#     "model__min_samples_split": [2, 5, 10],
#     "preprocessor__num_features__imputer__strategy": ["median", "mean"]
# }
# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models,predictions = reg.fit(x_train, x_test, y_train, y_test)
# print(models)
# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print(models)

# model = RandomizedSearchCV(reg, param_distributions=params, scoring="r2", cv=6, verbose=2, n_jobs=6, n_iter=30)
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)
# reg.fit(x_train, y_train)
# y_predict = reg.predict(x_test)
# for i,j in zip(y_test, y_predict):
#     print("Actual: {}. Predict: {}". format(i,j))
# print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
# print("R2: {}".format(r2_score(y_test, y_predict)))