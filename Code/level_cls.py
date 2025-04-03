import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from imblearn.over_sampling import RandomOverSampler, SMOTEN
import re
def fileter_location(loc):
    result = re.findall(r",\s[A-Z]{2}$", loc)
    if len(result):
        return result[0][2:]
    else:
        return loc

data = pd.read_excel("final_project.ods", dtype= str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(fileter_location)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={
    "managing_director_small_medium_company": 500,
    "specialist": 500,
    "director_business_unit_leader": 500,
    "bereichsleiter": 1000

})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(y_train.value_counts())
print("---------------")
x_train, y_train = ros.fit_resample(x_train, y_train)
print(y_train.value_counts())
exit(0)

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95 ), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(), "industry")
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])
# processed_data = cls.fit_transform(x_train, y_train)
# print(processed_data.shape) # Uni gram + bi gram : (6458, 850562)
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))

#                                           precision    recall  f1-score   support
#
#                         bereichsleiter       0.34      0.35      0.34       185
#          director_business_unit_leader       0.30      0.19      0.23        16
#                    manager_team_leader       0.54      0.56      0.55       518
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.81      0.79      0.80       889
#                             specialist       0.33      0.17      0.22         6
#
#                               accuracy                           0.66      1615
#                              macro avg       0.39      0.34      0.36      1615
#                           weighted avg       0.66      0.66      0.66      1615













# processed_data = preprocessor.fit_transform(x_train)
# print(processed_data.shape)


# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_train.shape)

# vectorizer = TfidfVectorizer(stop_words="english")
# preproced_data = vectorizer.fit_transform(x_train["function"])
# preproced_data = pd.DataFrame(preproced_data.todense())
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(preproced_data.shape)

# print(len(x_train["industry"].unique()))
# encoder = OneHotEncoder()
# processed_data = encoder.fit_transform(x_train[["location"]])
# print(processed_data.shape)