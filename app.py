import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Headings
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')

st.write(df.describe())

# Prepare the data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to get user input data
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Get user data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Random Forest model training
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predictions = rf.predict(x_test)

# Random Forest accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Logistic Regression model training
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predictions = lr.predict(x_test)

# Logistic Regression accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)

# Neural Network model
nn = Sequential([
    Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
nn_loss, nn_accuracy = nn.evaluate(x_test, y_test, verbose=0)

# Visualizations
st.title('Visualised Patient Report')

# Color function based on Random Forest prediction
color_rf = 'red' if rf.predict(user_data)[0] == 1 else 'green'

# Display prediction results
st.subheader('Random Forest Prediction')
st.write(f"Prediction: {'Diabetic' if rf.predict(user_data)[0] == 1 else 'Non-Diabetic'}")
st.write(f"Accuracy: {rf_accuracy:.2f}")

st.subheader('Logistic Regression Prediction')
st.write(f"Prediction: {'Diabetic' if lr.predict(user_data)[0] == 1 else 'Non-Diabetic'}")
st.write(f"Accuracy: {lr_accuracy:.2f}")

st.subheader('Neural Network Prediction')
st.write(f"Prediction: {'Diabetic' if nn.predict(user_data)[0] > 0.5 else 'Non-Diabetic'}")
st.write(f"Accuracy: {nn_accuracy:.2f}")

# Plotting the distribution of Glucose levels
st.subheader('Glucose Level Distribution')
plt.figure(figsize=(10, 5))
sns.histplot(df['Glucose'], kde=True, color='blue')
plt.title('Distribution of Glucose Levels')
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
st.pyplot(plt)

# Plotting a correlation heatmap
st.subheader('Feature Correlation Heatmap')
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
st.pyplot(plt)

# Plotting pairplot
st.subheader('Pairplot of Features')
sns.pairplot(df, hue='Outcome')
st.pyplot(plt)
