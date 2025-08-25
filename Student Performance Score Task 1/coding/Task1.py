import pandas as pd       
import numpy as np         
import matplotlib.pyplot as plt  
import seaborn as sns      
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression    
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import time

warnings.filterwarnings('ignore')
#read data
students_data = pd.read_csv("C:/Users/DELL/Desktop/ML learning internship/Student Performance Score Task 1/csv file/archive/StudentPerformanceFactors.csv")
#explore data
print(students_data.head())
print(students_data.info())
print(students_data.duplicated().sum())
print(f'The Shape of The DataSet:{students_data.shape}')
print(students_data.describe())
students_data1=students_data.copy()
students_data1.dropna(axis=0,inplace=True)
students_data1['Motivation_Level'] = students_data1['Motivation_Level'].str.strip().str.title()
motivation_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3
}

students_data1['Motivation_Level'] = students_data1['Motivation_Level'].map(motivation_mapping)
print(students_data1['Motivation_Level'].unique())


y=students_data1.Exam_Score
print(y)
#features
features_input = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Motivation_Level',
    'Internet_Access', 'Teacher_Quality', 'Tutoring_Sessions',
    'Learning_Disabilities', 'Access_to_Resources',
    'Extracurricular_Activities', 'Peer_Influence'
]


#visualization
plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
sns.scatterplot(data=students_data1, x='Hours_Studied', y='Exam_Score', hue='Gender')
plt.title('Hours Studied vs Exam Score')

plt.subplot(1, 4, 2)
sns.scatterplot(data=students_data1, x='Attendance', y='Exam_Score', hue='Gender')
plt.title('Attendance vs Exam Score')

plt.subplot(1, 4, 3)
sns.scatterplot(data=students_data1, x='Sleep_Hours', y='Exam_Score', hue='Gender')
plt.title('Sleep Hours vs Exam Score')

plt.subplot(1, 4, 4)
sns.boxplot(x='Motivation_Level', y='Exam_Score', data=students_data1)
plt.title('Motivation Level vs Exam Score')

plt.tight_layout()
plt.show(block=False)  

#heatmap
plt.figure(figsize=(6,5))
sns.heatmap(students_data1.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show(block=False)  
time.sleep(10)

input("Press Enter to close all figures...")


X=students_data1[features_input]
#target
Y=students_data1['Exam_Score']
print(X.describe())
print(X.head())

X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(X_test.shape,X_train.shape)

#building model
model=LinearRegression(fit_intercept=True,n_jobs=1)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("predictions: ",y_pred)

intercept=model.intercept_
print("intercepts: ",intercept)

coeff=model.coef_
print("coefficents: ",coeff)
#evaluation
mean_square=mean_squared_error(y_test, y_pred) 
print("mean square: ",mean_square)
r2=r2_score(y_test, y_pred)   
print("r2 score: ",r2)
train_score=model.score(X_train, y_train) 
print("train score: ",train_score)
test_score=model.score(X_test, y_test)  
print("test score: ",test_score)

feature_names = X.columns  

for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef}")









