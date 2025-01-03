# HR Employee Attrition and Analysis 🚀💼
This project is a feature-rich HR Analytics and Employee Dashboard built using Streamlit, designed to provide secure and actionable insights for both employees and HR professionals. The application includes separate dashboards for employees and HR, along with a robust login system that uses hashed credentials.
---

## **Features**

### **1. Login System** 🔐
- Employees and HR professionals can log in securely using their credentials stored in `login.csv`.
- Passwords are securely hashed for enhanced security.
- Employees can set up or reset their passwords after validating their `EmployeeNumber`.

### **2. Employee Dashboard** 👩‍💻
- Displays personal details like job role, department, monthly income, etc.
- Allows employees to submit exit statements if they are leaving the organization.
- Secure access ensures employees only see their own data.

### **3. HR Dashboard** 🧑‍💼
- HR professionals can:
  - Upload employee data for analysis.
  - Predict employee turnover probability using a pre-trained ML model.
  - Perform clustering and sentiment analysis on employee data.
  - Visualize key HR metrics using charts and graphs.

### **4. Turnover Prediction**
- Predict employee turnover probability using a machine learning model.
- Download predictions as a CSV file for further analysis.

### **5. Clustering**
- Perform KMeans clustering on employee data for segmentation.
- Visualize clusters and review cluster centroids.

### **6. Sentiment Analysis**
- Analyze employee exit statements for sentiment polarity and subjectivity.
- Generate word clouds to visualize frequently used terms in exit statements.

---

## **Technologies Used** 🛠️
- **Programming Language**: Python
- **Framework**: Streamlit
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation and analysis
  - `matplotlib`, `seaborn`: Data visualization
  - `bcrypt`: Password hashing
  - `sklearn`: Machine learning (e.g., clustering, prediction)
  - `wordcloud`, `textblob`: Sentiment analysis
  - `joblib`: Model serialization

---

## **Project Structure** 🗂️
```bash
HR_Analytics_Project/
├── app.py # Main Streamlit application file
├── pipeline.ipynb # Notebook for data preprocessing and model training
├── login.csv # Stores employee login credentials (hashed passwords)
├── Modified_HR_Employee_Attrition_Data1.csv # Employee dataset
├── hr_model.pkl # Pre-trained turnover prediction model
├── sentiment_model.pkl # Pre-trained sentiment analysis model
└── README.md # Project documentation (this file)
```
---

## **Setup Instructions**

### **1. Prerequisites** ⚙️
Ensure you have Python installed (version >= 3.8). Install the required libraries:
```bash
pip install streamlit pandas numpy matplotlib seaborn bcrypt sklearn wordcloud textblob joblib
```
### **2. How to Run the Application**
1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Run the Streamlit app:
```bash
streamlit run app.py
```
4. Open the provided URL in your browser to access the dashboard.

---

## **How It Works**

### **Login System** 🔑
1. The `login.csv` file stores hashed passwords for employees and HR professionals.
2. Employees can log in using their Employee ID, while HR professionals use predefined credentials.
3. If an employee does not have a password set up, they can create one by validating their Employee ID against the dataset (`Modified_HR_Employee_Attrition_Data1.csv`).

### **Employee Dashboard**
- Displays personal details like job role, department, monthly income, etc.
- Allows employees to submit exit statements if they are leaving the organization.
- Ensures secure access to individual data only.

### **HR Dashboard**
Once logged in, HR professionals can:
- Upload new datasets for analysis.
- Predict turnover probabilities using machine learning models.
- Perform clustering to segment employees into groups based on similarities.
- Analyze sentiment from exit statements.


---

## **Dataset Information**

The dataset (`Modified_HR_Employee_Attrition_Data1.csv`) contains detailed information about employees, including:
- Personal details: `EmployeeNumber`, `Gender`, `Age`
- Job-related details: `JobRole`, `Department`, `MonthlyIncome`
- Attrition indicators: `YearsAtCompany`, `TrainingTimesLastYear`, `ExitStatement`

The additional file (`login.csv`) stores login credentials for employees:

---

## **Future Enhancements** 🌟🚀
1. Implement role-based access control (e.g., different permissions for HR vs employees).
2. Integrate a database (e.g., PostgreSQL) for secure storage of login credentials instead of CSV files.
3. Add advanced analytics features like predictive modeling for promotions or salary hikes.

---

## **Acknowledgments** 💡✨🎉
This project was developed to provide HR professionals with actionable insights into employee data while maintaining secure access for employees through a robust login system.

Feel free to contribute or report issues to improve this project!


