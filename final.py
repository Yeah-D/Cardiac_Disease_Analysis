import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

df = pd.read_csv("heart_2020_cleaned.csv")
df.shape

col_names = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic',
             'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma','KidneyDisease', 'SkinCancer']

df.columns = col_names

df2= df

#________숫자형 데이터 범주형으로 바꿔주기________

#BMI
ranges = [0, 18.5, 24.9, 29.9, 39.9, 95]
df2['BMI'] = pd.cut(df2['BMI'], ranges, labels =['Under', 'Normal', 'Pre-Obesity', 'Obesity', 'High-Obesity'])

#Sleep Time
ranges2 = [0, 3, 6, 24 ]
df2['SleepTime'] = pd.cut(df2['SleepTime'], ranges2, labels = ['lack', 'Semi-lack', 'enough'])

#PhysicalHealth
ranges3 = [0, 1, 10, 20, 29 , 30]
df2['PhysicalHealth'] = pd.cut(df2['PhysicalHealth'], ranges3, labels = ['0,','1s','10s','20s','30'])

#MentalHealth
ranges4 = [0, 1, 10, 20, 29 , 30]
df2['MentalHealth'] = pd.cut(df2['MentalHealth'], ranges4, labels = ['0,','1s','10s','20s','30'])

age_map = {'18-24': '20s', '25-29': '20s', '30-34': '30s', '35-39': '30s', '40-44': '40s', '45-49': '40s', '50-54': '50s', '55-59': '50s', '60-64': '60s', '65-69': '60s', '70-74': '70s', '75-79': '70s', '80 or older': '80s'}

df2['AgeCategory'] = df2['AgeCategory'].apply(lambda x: age_map[x] if x in age_map else x)


#열 제거
df2 = df.drop(['AlcoholDrinking', 'Race', 'PhysicalActivity','GenHealth', 'Smoking', 'SleepTime', 'SkinCancer', 'MentalHealth', 'KidneyDisease'], axis=1)

print(df2.columns)
age_category_factors = df2['AgeCategory'].unique()

# 범주형 데이터 값들의 개수와 비율 계산
for col in df2.columns:
    if col in df2:
        print('')
        print('### Column: {} ###'.format(col))
        print(df2[col].value_counts(normalize=True))

le = LabelEncoder()
import itertools  # itertools 모듈 추가

results = []

# 샘플링 작업 10번 반복
for i in range(10):

    # 요소 리스트 생성
    age_categories = df2['AgeCategory'].unique()  # 연령대 요소 추가
    genders = df2['Sex'].unique()
    heart_diseases = df2['HeartDisease'].unique()

    # 가능한 모든 조합 생성
    combinations = list(itertools.product(age_categories, genders, heart_diseases))

    min_counts = []

    # 각 조합별로 최소 데이터 수 계산
    for combination in combinations:
        matched_rows = df2[(df2['AgeCategory'] == combination[0]) & (df2['Sex'] == combination[1]) & (
                    df2['HeartDisease'] == combination[2])]
        if not matched_rows.empty:
            min_counts.append(matched_rows.shape[0])

    min_count = min(min_counts)

    # 동일한 비율로 조합된 샘플을 저장할 빈 데이터 프레임
    equal_ratio_samples = pd.DataFrame(columns=df2.columns)

    # 각 조합에서 동일한 비율의 샘플을 추출하고 결과 데이터 프레임에 추가
    for combination in combinations:
        matched_rows = df2[(df2['AgeCategory'] == combination[0]) & (df2['Sex'] == combination[1]) & (
                    df2['HeartDisease'] == combination[2])]
        if not matched_rows.empty:
            samples = matched_rows.sample(n=min_count)
            equal_ratio_samples = equal_ratio_samples.append(samples, ignore_index=True)

    # 데이터 프레임을 무작위로 셔플
    equal_ratio_samples = equal_ratio_samples.sample(frac=1).reset_index(drop=True)

    # 결과를 리스트에 추가
    results.append(equal_ratio_samples)

# 결과 예측값을 저장할 빈 리스트
bagging_predicted = []

for i in range(10):
    # 결과 데이터 프레임 준비
    df3 = results[i]

    # 범주형 데이터 인코딩
    columns_to_encode = df2.columns
    for column in columns_to_encode:
        le = LabelEncoder()
        df3[column] = le.fit_transform(df3[column])

    # 특성과 타겟 변수 준비
    X = df3.drop('HeartDisease', axis=1)
    y = df3['HeartDisease']

    # stratify 옵션을 사용하여 class 비율을 동일하 유지한 채로 train/test 분리 수행
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 방법1-GridSearchCV 사용
# 하이퍼파라미터 조정을 위한 매개변수 그리드
param_grid = {
    'n_neighbors': range(1, 40),
    'weights': ['uniform', 'distance'],  # Add weights parameter
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev' ]  # Add metric parameter
}

# KNN 분류기 생성
knn = KNeighborsClassifier()

# 하이퍼파라미터 조정을 위해 GridSearchCV 사용
grid_search = GridSearchCV(knn, param_grid, cv=5)

# fit the randomized search object to the data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# 피쳐 크기 조정
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 최상의 하이퍼 파라미터로 KNN 모델 초기화 및 적합
clf_knn = KNeighborsClassifier(**best_params)
clf_knn.fit(X_train_scaled, y_train)

# 모형 예측 및 평가
pred = clf_knn.predict(X_test_scaled)

# evaluate the model
accuracy = accuracy_score(y_test, pred)
report = classification_report(y_test, pred)

print("Best hyperparameters: ", best_params)
print(f"Accuracy score: {accuracy:.4f}")
print("Classification report:")
print(report)
print("-------------------------------------------------")

# 방법2-RandomizedSearchCV 사용
# 하이퍼파라미터 조정을 위한 매개변수 그리드
param_dist = {'n_neighbors': range(1, 50),
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

# KNN 분류기 생성
knn = KNeighborsClassifier()

# 하이퍼파라미터 조정을 위해 RandomizedSearchCV 사용
rscv = RandomizedSearchCV(knn, param_distributions=param_dist, cv=5, n_iter=20, random_state=42)

# fit the randomized search object to the data
rscv.fit(X_train_scaled, y_train)

# get best hyperparameters
best_params = rscv.best_params_

# 피쳐 크기 조정
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 최상의 하이퍼 파라미터로 KNN 모델 초기화 및 적합
knn = KNeighborsClassifier(**best_params)
knn.fit(X_train_scaled, y_train)

# 모형 예측 및 평가
y_pred = knn.predict(X_test_scaled)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best hyperparameters:", best_params)
print(f"Accuracy score: {accuracy:.4f}")
print("Classification report:")
print(report)

# 최종
# 범주형 데이터 인코딩
le = LabelEncoder()
df_encoded = df2.apply(le.fit_transform)

# 언더샘플링 수행 및 KNN model 조정
for i in range(5):
    # 언더샘플링을 수행하여 클래스의 불균형 조정
    min_count = df_encoded['HeartDisease'].value_counts().min()
    undersampled_df = pd.concat([df_encoded[df_encoded['HeartDisease'] == 0].sample(min_count),
                                 df_encoded[df_encoded['HeartDisease'] == 1].sample(min_count)])
    X = undersampled_df.drop('HeartDisease', axis=1)
    y = undersampled_df['HeartDisease']

    # 훈련 및 테스트 세트로 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # KNN model 학습
    clf_knn = KNeighborsClassifier(n_neighbors=25)
    clf_knn.fit(X_train, y_train)
    
    # model 평가
    y_pred = clf_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy score: {accuracy:.4f}')
    print("Classification report-\n")
    print(classification_report(y_test, y_pred))
    
    # bagging에 예측 값 저장
    bagging_predicted.append(y_pred)

# bagging 예측에서 다수결 계산 - 앙상블 기법
bagging_predictions = pd.DataFrame(bagging_predicted).T.mode(axis=1).squeeze()
print("Final result")
print("Bagging predictions:\n")
print(bagging_predictions)

# bagging 예측의 정확도 계산
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
print(f'Bagging accuracy score: {bagging_accuracy:.4f}')
    
# bagging 예측에 대한 분류 결과
print("Classification report for bagging predictions:\n")
print(classification_report(y_test, bagging_predictions))