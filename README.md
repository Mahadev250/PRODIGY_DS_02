# PRODIGY_DS_02
 Circle with gradient Simple Gradient Half Circle Shape 02  Task-02  Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice, such as the Titanic dataset from Kaggle. Explore the relationships between variables and identify patterns and trends in the data.

CODE:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (assuming train.csv is downloaded from Kaggle)
df = pd.read_csv(r"C:\Users\Navaneet\Downloads\titanic\train.csv")
df.head()

# Basic info
df.info()

# Statistical summary
df.describe()

# Check for missing values
df.isnull().sum()

# Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
df.drop(columns='Cabin', inplace=True)
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Convert 'Sex' and 'Embarked' to numeric using get_dummies
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Survived', hue='Sex_male', data=df)
plt.title('Survival by Gender')
plt.xticks([0, 1], ['Did not survive', 'Survived'])
plt.legend(['Female', 'Male'])
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation')
plt.show()

![image](https://github.com/user-attachments/assets/9abbe79b-d335-4054-b709-99e6f601b2d9)
![image](https://github.com/user-attachments/assets/d3edfe0e-57e2-4ccc-9e0a-14abefb66478)
![image](https://github.com/user-attachments/assets/25e6a0d4-d6fb-4036-93d5-191478f37302)
![image](https://github.com/user-attachments/assets/48aa1bb1-78bf-48f6-977e-30c0550da1a2)
![image](https://github.com/user-attachments/assets/ac3e7e3d-c705-4510-8d39-2a9e06d600c0)
