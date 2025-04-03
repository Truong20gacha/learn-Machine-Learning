import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LarsCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('StudentScore.xls')
# print(data.head())
A = data['parental level of education'].unique()
# input(A)
# ["bachelor's degree" 'some college' "master's degree" "associate's degree"
#  'high school' 'some high school']
numeric_data = data.select_dtypes(include=[int, float])
"""
Kết quả sẽ là một DataFrame chỉ chứa các cột có kiểu dữ liệu số. (ở đây là gọi dữ liệu co dạng là int và float)
"""
# print(data[numeric_data.columns].corr())
"""
gọi lệnh tính toán ma trận hệ số tương quan giữa các cột trong biến numeric_data
               math score	  reading score	   writing score
math score	      1.00	           0.82	            0.80
reading score	  0.82             1.00	            0.95
writing score	  0.80             0.95	            1.00
"""

# train split data
target = "math score"
"""
lí do vì sao chọn cột math làm target thì đơn giản là vì người thực hiện muốn dự đoán điểm số toán dựa trên các điểm số khác như điểm đọc và điểm viết
và thêm 1 cái nữa là nó là cái mình dự đoán , ví dụ về bộ data này là do mình muốn dự đoán về số điểm toán nên target mới à math score , có liên quan tới hệ số đặc trưng
"""
x = data.drop(target, axis=1)
y = data[target]

x_train , x_test , y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

education_values = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
gender_values = ['male','female']
lunch_values = x_train['lunch'].unique()
test_preparation_course_values = x_train['test preparation course'].unique()

ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler'), OrdinalEncoder(categories=[education_values,gender_values,lunch_values,test_preparation_course_values])
])

nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder())
])
preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("nom_features", nom_transformer,["race/ethnicity"]),
    ("ord_features", ord_transformer,["parental level of education","gender","lunch","test preparation course"])]
    )
"""
các bước có pipeline ở trên thì nó sẽ làm "tiền xử lí cho ColumnTransformer " vì trong data có đa dạng feature có kiểu dữ liệu là ordinal , nominal , numerical vậy nên khi làm các bước có pipeline thì nên để đi cùng với ColumnTransformer để xác định cột nào trong data sẽ sử dụng nó
một số mô hình như cây quyết định (Decision Trees) hoặc Random Forest có thể không yêu cầu chuẩn hóa dữ liệu đầu vào.
"""
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LarsCV())
])
processed_data = reg.fit_transform(x_train)
pd.DataFrame(processed_data)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
# for i, j in zip(y_pred, y_test):
#     print("Predicted: ", i, "Actual: ", j)

params = {
    "model__max_n_alphas": [100, 200, 300, 400, 500],
    "model__n_jobs": [1, 2, 3, 4, 5],
    "model__precompute": [True, False],
    "model__max_iter": [100, 200, 300],
    "model__cv": [3, 4, 5, 6, 7, 8]
}
"""
mỗi params cho mỗi mô hinnh đều là khác nhau nên nhớ coi lại kĩ 
model__max_n_alphas": [100, 200, 300, 400, 500]:
Tham số này xác định số lượng giá trị alpha (tham số điều chỉnh) mà mô hình LarsCV (Least Angle Regression Cross-Validation) sẽ thử. LarsCV sẽ chọn ra giá trị alpha tốt nhất từ các giá trị này.

model__n_jobs": [1, 2, 3, 4, 5]:
n_jobs kiểm soát số lượng lõi CPU sẽ được sử dụng cho xử lý song song. Thiết lập số cao hơn có thể giúp tăng tốc độ tính toán.

model__precompute": [True, False]:
Nếu True, mô hình sẽ sử dụng ma trận Gram tính sẵn để tăng tốc độ tính toán, đặc biệt hữu ích khi có nhiều đặc trưng (features). Nếu False, ma trận sẽ được tính trong quá trình chạy.

odel__max_iter": [100, 200, 300]:
Số lần lặp tối đa được cho phép để mô hình hội tụ. Tham số này giúp tránh trường hợp mô hình chạy vô hạn nếu không hội tụ được.

model__cv": [3, 4, 5, 6, 7, 8]:
Số lượng fold (gấp) trong cross-validation. Cross-validation giúp ước tính hiệu suất của mô hình trên dữ liệu chưa thấy qua việc chia dữ liệu thành nhiều phần và kiểm tra trên từng phần.
"""

model = RandomizedSearchCV(reg, param_distributions=params, cv=5, n_jobs=-1,verbose=2, scoring='r2', n_iter=9, random_state=42)


model.fit(x_train, y_train)
y_pred = model.predict(x_test)
