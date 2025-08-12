Student Exam Score Prediction
📌 Overview
This project predicts students’ exam scores based on multiple academic, behavioral, and social factors such as study hours, attendance, motivation level, access to resources, and peer influence.
It uses a Linear Regression model to determine how each factor impacts performance, providing actionable insights for improving student outcomes.
 Dataset from Kaggel:
The dataset contains the following features:
Hours_Studied
Attendance
Sleep_Hours
Motivation_Level
Internet_Access
Teacher_Quality
Tutoring_Sessions
Learning_Disabilities
Access_to_Resources
Extracurricular_Activities
Peer_Influence
Exam_Score (Target Variable)

Technologies Used:
Python 3
Pandas – Data handling
NumPy – Numerical computations
Matplotlib & Seaborn – Data visualization
Scikit-learn – Machine learning model building

📊 Steps in the Project:
Data Loading & Cleaning – Handle missing values, encode categorical features.
Exploratory Data Analysis (EDA) – Visualize relationships between features and exam scores.
Feature Encoding – Convert categorical variables to numeric format.
Train-Test Split – Split the dataset into training and testing sets.
Model Training – Build and train a Linear Regression model.
Model Evaluation – Evaluate performance using Mean Squared Error (MSE) and R² score.
Insights – Identify which factors most influence exam scores.

📈 Model Performance:
Mean Squared Error (MSE): 5.50
R² Score (Test Data): 0.646
Train Score: 0.627
