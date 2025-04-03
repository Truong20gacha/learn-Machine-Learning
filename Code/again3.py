import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
# print(data.head())
# print(data.info())
# print(data)
# profile = ProfileReport(data, title="phan_tich_bc", explorative=True)
# profile.to_file("bao_cao_taolao.html")

# static data
# train_spilt_data

target = ["logS"]
x = data.drop(target, axis=1)
# print(x)
y = data[target].values.ravel()
# print(y)
#       logS
# 0    -2.180
# 1    -2.000
# 2    -1.740
# 3    -1.480
# 4    -3.040
# ...     ...
# 1139  1.144
# 1140 -4.925
# 1141 -3.893
# 1142 -3.790
# 1143 -2.581

x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
# print(x_train)
# 80% của bộ train trong data
"""
MolLogP    MolWt  NumRotatableBonds  AromaticProportion
107  3.14280  112.216                5.0            0.000000
378 -2.07850  142.070                0.0            0.000000
529 -0.47730  168.152                0.0            0.000000
546 -0.86740  154.125                0.0            0.000000
320  1.62150  100.161                2.0            0.000000
..       ...      ...                ...                 ...
802  3.00254  250.301                1.0            0.842105
53   2.13860   82.146                3.0            0.000000
350  5.76304  256.348                0.0            0.900000
79   3.89960  186.339               10.0            0.000000
792  2.52334  310.297                3.0            0.300000
"""
# 20% của bộ test
# print(x_test)
"""
MolLogP    MolWt  NumRotatableBonds  AromaticProportion
822   2.91000  172.268                7.0            0.000000
118   7.27400  360.882                1.0            0.666667
347   1.94040  145.161                0.0            0.909091
1123  1.98640  119.378                0.0            0.000000
924   1.70062  108.140                0.0            0.750000
...       ...      ...                ...                 ...
1114  1.76210  478.513                4.0            0.000000
427   6.32820  276.338                0.0            1.000000
711   0.04430  218.205                5.0            0.000000
4     2.91890  187.375                1.0            0.000000
948   3.56010  318.328                2.0            0.750000
"""

# Model building ??
lr = LinearRegression()
lr.fit(x_train, y_train)

# Applying the model to make a prediction
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# evaluate model performance
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
# 1.0075362951093687
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
# 0.7645051774663391
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
# 1.0206953660861033
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
# 0.7891616188563282

# print("LR MSE (Train):", lr_train_mse)
# print("LR R2 (Train):", lr_train_r2)
# print("LR MSE (Test):", lr_test_mse)
# print("LR R2 (Test):", lr_test_r2)
# LR MSE (Train): 1.0075362951093687
# LR R2 (Train): 0.7645051774663391
# LR MSE (Test): 1.0206953660861033
# LR R2 (Test): 0.7891616188563282

# hoặc ta có thể dùng dataframe để trình bày dữ liệu cho đpẹ mắt

ls_result = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
ls_result.columns = ['method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(ls_result)
#                method Training MSE Training R2  Test MSE   Test R2
# 0  Linear regression     1.007536    0.764505  1.020695  0.789162

# Random Forest
# Training the model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)
# Applying the model to make a prediction
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)
# evaluate model performance
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)
rf_result = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_result.columns = ['method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(rf_result)
#             method Training MSE Training R2  Test MSE   Test R2
# 0  Random Forest     1.028228    0.759669  1.407688  0.709223

#Model comparison

df_models = pd.concat([ls_result, rf_result], axis=0)
print(df_models)
#                method Training MSE Training R2  Test MSE   Test R2
# 0  Linear regression     1.007536    0.764505  1.020695  0.789162
# 0      Random Forest     1.028228    0.759669  1.407688  0.709223

#Data visualization of prediction results
plt.figure(figsize=(5,5))
"""
Dòng này tạo một hình mới với kich thước được chỉ định là 5x5
"""
plt.scatter(x=y_train, y=y_lr_train_pred,c="#7CAE00", alpha=0.3)
"""
Dòng này tạo biểu đồ phân tán (scatter plot) với 'y_train'(Logs thực nghiệm) trên trục x và y_lr_train_pred (LogS dự đoán) trên trục y. Các điểm được tô màu với mã màu hex #7CAE00 và có mức độ trong suốt là 0.3. 
"""
z = np.polyfit(y_train, y_lr_train_pred, 1)
"""
Dòng này thực hiện khớp một đa thức bậc nhất (một đường thẳng) với dữ liệu. Hàm np.polyfit trả về các hệ số của phương trình đường thẳng (hệ số góc và điểm giao). Số 1 chỉ ra rằng một đa thức bậc nhất (đường thẳng) đang được khớp.
"""
p = np.poly1d(z)
"""
òng này tạo một hàm đa thức p từ các hệ số z thu được từ np.polyfit. Hàm này có thể được sử dụng để tính các giá trị y của đường thẳng vừa khớp.
"""
plt.plot(y_train, p(y_train), "#F8766D")
"""
 Dòng này vẽ đường thẳng vừa khớp lên biểu đồ phân tán. Các giá trị x là y_train, và các giá trị y được tính bằng cách áp dụng hàm đa thức p lên y_train. Đường thẳng này được tô màu với mã màu hex #F8766D.
"""
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
plt.show()
