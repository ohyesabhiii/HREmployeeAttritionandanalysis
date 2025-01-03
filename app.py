import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud

class DynamicOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.quantiles_ = {
            col: (X[col].quantile(self.lower_quantile), X[col].quantile(self.upper_quantile))
            for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.quantiles_.items():
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Polarity'] = X['ExitStatement'].apply(
            lambda text: TextBlob(text).polarity if pd.notnull(text) else None
        )
        X['Subjectivity'] = X['ExitStatement'].apply(
            lambda text: TextBlob(text).subjectivity if pd.notnull(text) else None
        )
        return X



# Load the trained model
pickle_file_path = r"D:\HREmployeeAttritionandanalysis\hr_model.pkl"
file_path=r"D:\HREmployeeAttritionandanalysis\sentiment_model.pkl"
model = joblib.load(pickle_file_path)
sentiment_model=joblib.load(file_path)
employee_data = pd.read_csv(r'D:\HREmployeeAttritionandanalysis\login.csv')
try:
    detailed_employee_data = pd.read_csv('Modified_HR_Employee_Attrition_Data1.csv')
except FileNotFoundError:
    st.error("The detailed employee data file (Modified_HR_Employee_Attrition_Data1.csv) is missing.")
    detailed_employee_data = pd.DataFrame()  # Create an empty DataFrame as a fallback

# Sample HR credentials (replace with your actual data)
hr_credentials = {
    "username": "hr",
    "password": "hr123"
}

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.employee_number = None
if "creating_password" not in st.session_state:
    st.session_state.creating_password = False
# Function to validate or create passwords for employees
def handle_employee_login(employee_number, password):
    global employee_data

    # Check if employee number exists in the database
    if employee_number not in employee_data["EmployeeNumber"].values:
        st.error("Employee number not found.")
        return False

    # Get the employee's row
    employee_row = employee_data[employee_data["EmployeeNumber"] == employee_number]

    # Check if a password is already set
    if pd.notna(employee_row["Password"].values[0]):
        # Validate the password
        if password == employee_row["Password"].values[0]:
            st.success(f"Welcome back, {employee_row['EmployeeNumber'].values[0]}!")
            return True
        else:
            st.error("Incorrect password.")
            return False
    else:
        st.session_state.creating_password = True
        st.session_state.employee_number = employee_number
        return False

# Function to handle setting a new password
def handle_set_password(new_password, confirm_password):
    global employee_data

    if new_password == confirm_password and new_password:
        # Update the password in the database
        employee_data.loc[employee_data["EmployeeNumber"] == st.session_state.employee_number, "Password"] = new_password
        employee_data.to_csv('login.csv', index=False)
        st.success("Password set successfully! Please log in again.")
        st.session_state.creating_password = False  # Exit password creation mode
    else:
        st.error("Passwords do not match or are empty.")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.employee_number = None

# Login Screen for Employees and HR
if not st.session_state.logged_in:
    st.title("Login")

    # Separate login buttons for Employee and HR
    role = st.radio("Select Role", options=["Employee", "HR"])

    if role == "Employee":
        emp_number = st.number_input("Enter Employee Number:", min_value=1, step=1)
        
        # If creating a new password, show the password creation form
        if st.session_state.creating_password and emp_number == st.session_state.employee_number:
            new_password = st.text_input("Set a new password:", type="password")
            confirm_password = st.text_input("Confirm new password:", type="password")

            if st.button("Set Password"):
                handle_set_password(new_password, confirm_password)
        
        else:
            emp_password = st.text_input("Enter Password:", type="password")
            
            if st.button("Login as Employee"):
                if handle_employee_login(emp_number, emp_password):
                    st.session_state.logged_in = True
                    st.session_state.user_role = "employee"
                    st.session_state.employee_number = emp_number

    elif role == "HR":
        hr_username = st.text_input("HR Username")
        hr_password = st.text_input("HR Password", type="password")

        if hr_username == "hr" and hr_password == "hr123":
            if st.button("Login as HR"):
                st.success(f"Welcome, HR Admin!")
                st.session_state.logged_in = True
                st.session_state.user_role = "hr"
        else:
            if hr_username or hr_password:  # Show error only when user has entered something.
                st.error("Invalid HR credentials.")

else:
    # Logged-in view for employees or HR users
    if st.session_state.user_role == "employee":
        # Employee Dashboard
        emp_row = detailed_employee_data[detailed_employee_data["EmployeeNumber"] == st.session_state.employee_number]
        emp_name = emp_row["EmployeeNumber"].values[0]
        st.sidebar.title(f"Logged in as: {emp_name} (Employee #{st.session_state.employee_number})")
        if not emp_row.empty:
            attrition_status = emp_row["Attrition"].values[0]

            if attrition_status == "Yes":
                # Show message if the employee has left
                st.error("Employee left.")
            else:
                # Display employee details excluding the Attrition column
                emp_details = emp_row.drop(columns=["Attrition"]).iloc[0].to_dict()
                st.title(f"Welcome, Employee #{st.session_state.employee_number}!")
                st.subheader("Your Details:")
                for key, value in emp_details.items():
                    st.write(f"**{key}:** {value}")

                # Ask if the employee is exiting the company
                exit_option = st.radio("Are you exiting the company?", options=["No", "Yes"])
                
                if exit_option == "Yes":
                    # Comment box for Exit Statement
                    exit_statement = st.text_area("Please provide your exit statement:")
                    
                    if st.button("Submit Exit Statement"):
                        # Update ExitStatement in the DataFrame and save to CSV
                        detailed_employee_data.loc[detailed_employee_data["EmployeeNumber"] == st.session_state.employee_number, "ExitStatement"] = exit_statement
                        detailed_employee_data.loc[detailed_employee_data["EmployeeNumber"] == st.session_state.employee_number, "Attrition"] = "Yes"
                        # Save updated DataFrame to CSV
                        detailed_employee_data.to_csv('Modified_HR_Employee_Attrition_Data1.csv', index=False)
                        
                        st.success("Your exit statement has been submitted. Thank you!")
        
        else:
            st.error(f"No details found for Employee #{st.session_state.employee_number}.")

        # Logout button
        if st.sidebar.button("Logout"):
            logout()
    elif st.session_state.user_role == "hr":
        # HR-specific content here
        st.title("HR Analytics Dashboard")
        st.write("""
            This app allows HR professionals to:
            - Upload employee data for analysis.
            - Predict employee turnover probability.
            - Perform clustering and visualization.
        """)

        # File uploader on the main page
        uploaded_file = st.file_uploader("Upload Employee Data (CSV)", type=["csv"])

        if uploaded_file is not None:
            # Read uploaded CSV file
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.write(data.head())

            # Sidebar navigation for features
            st.sidebar.title("Options")
            selected_option = st.sidebar.radio(
                "Choose a feature:",
                ("Predict Turnover Probability", "Clustering", "Sentiment Analysis")
            )

            # Option 1: Predict Turnover Probability
            if selected_option == "Predict Turnover Probability":
                st.subheader("Turnover Probability Prediction")
                if st.button("Predict"):
                    predictions = model.predict(data)
                    data['Turnover_Probability'] = predictions

                    st.write("Predictions:")
                    st.write(data)

                    # Pie Chart of Turnover Categories
                    st.subheader("Pie Chart of Turnover Categories")
                    category_counts = data['Turnover_Probability'].value_counts()
                    fig1, ax1 = plt.subplots(figsize=(6, 6))
                    ax1.pie(
                        category_counts,
                        labels=category_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=['#4CAF50', '#F44336']  # Green for "No", Red for "Yes"
                    )
                    ax1.set_title("Turnover Probability Distribution")
                    st.pyplot(fig1)

                    # Count Plot of Turnover Categories
                    st.subheader("Count Plot of Turnover Categories")
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    sns.countplot(x='Turnover_Probability', data=data, palette=['#4CAF50', '#F44336'], ax=ax2)
                    ax2.set_title("Count of Turnover Categories")
                    ax2.set_xlabel("Turnover Category")
                    ax2.set_ylabel("Count")
                    st.pyplot(fig2)

                    # Downloadable CSV with predictions
                    csv = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

            # Option 2: Clustering
            elif selected_option == "Clustering":
                st.sidebar.subheader("Clustering Parameters")
                selected_features = st.sidebar.multiselect("Select Features for Clustering", options=data.columns)

                # Exclude 'ExitStatement' from clustering features if selected
                if 'ExitStatement' in selected_features:
                    selected_features.remove('ExitStatement')

                if selected_features:
                    # Separate categorical and numerical columns
                    categorical_columns = data[selected_features].select_dtypes(include=['object', 'category']).columns.tolist()
                    numerical_columns = data[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

                    # Preprocess Data: One-Hot Encoding for categorical and scaling for numerical
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numerical_columns),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                        ]
                    )

                    # Fit and transform the data
                    scaled_data = preprocessor.fit_transform(data[selected_features])

                    # Number of Clusters
                    num_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=4)

                    # KMeans Clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    clusters = kmeans.fit_predict(scaled_data)
                    data['Cluster'] = clusters

                    # Display cluster assignments
                    st.subheader("Cluster Assignments")
                    st.write(data['Cluster'].value_counts().sort_index())

                    # Display cluster centroids (only for numerical features)
                    if numerical_columns or categorical_columns:
                        centroids = kmeans.cluster_centers_

                        # Get feature names for categorical columns after one-hot encoding
                        if categorical_columns:
                            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
                            all_feature_names = numerical_columns + list(cat_feature_names)
                        else:
                            all_feature_names = numerical_columns

                        centroid_df = pd.DataFrame(centroids, columns=all_feature_names)
                        st.subheader("Cluster Centroids")
                        st.write(centroid_df)

                    # Visualization of Clusters
                    x_axis = st.sidebar.selectbox("Select X-axis Feature", options=selected_features)
                    y_axis = st.sidebar.selectbox("Select Y-axis Feature", options=selected_features)

                    if x_axis and y_axis:
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(
                            x=data[x_axis],
                            y=data[y_axis],
                            hue=data['Cluster'],
                            palette="viridis",
                            s=100,
                            edgecolor='k'
                        )
                        plt.title(f"KMeans Clustering ({num_clusters} Clusters)")
                        plt.xlabel(x_axis)
                        plt.ylabel(y_axis)
                        st.pyplot(plt.gcf())

            # Option 3: Sentiment Analysis
            elif selected_option == "Sentiment Analysis":
                if 'ExitStatement' in data.columns:
                    # Drop missing values in 'ExitStatement'
                    df_sentiment = data.dropna(subset=['ExitStatement'])

                    # Function to calculate polarity and subjectivity
                    def analyze_sentiment(text):
                        blob = TextBlob(text)
                        return blob.sentiment.polarity, blob.sentiment.subjectivity

                    # Apply sentiment analysis
                    df_sentiment['Polarity'], df_sentiment['Subjectivity'] = zip(*df_sentiment['ExitStatement'].apply(analyze_sentiment))

                    # Scatter Plot: Polarity vs. Subjectivity
                    st.subheader("Scatter Plot of Polarity vs. Subjectivity")
                    fig1, ax1 = plt.subplots(figsize=(6, 6))
                    ax1.scatter(df_sentiment['Polarity'], df_sentiment['Subjectivity'], alpha=0.5)
                    ax1.set_title('Sentiment Analysis of Exit Statements')
                    ax1.set_xlabel('Polarity')
                    ax1.set_ylabel('Subjectivity')
                    ax1.grid(True)
                    ax1.axhline(0.5, color='red', linestyle='--', label='Subjective Threshold')
                    ax1.axvline(0, color='blue', linestyle='--', label='Neutral Threshold')
                    ax1.legend()
                    
                    # Display scatter plot in Streamlit
                    st.pyplot(fig1)

                    # Word Cloud
                    st.subheader("Word Cloud of Exit Statements")
                    text = ' '.join(df_sentiment['ExitStatement'])
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.imshow(wordcloud, interpolation='bilinear')
                    ax2.axis('off')  # Turn off axis labels
                    ax2.set_title('Word Cloud of Exit Statements', fontsize=16)
                    
                    # Display word cloud in Streamlit
                    st.pyplot(fig2)

                else:
                    st.error("The uploaded dataset does not contain an 'ExitStatement' column.")
        if st.sidebar.button("Logout"):
            logout()