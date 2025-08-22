<H3>ENTER YOUR NAME Mohamed Hameem Sajith J</H3>
<H3>ENTER YOUR REGISTER NO.  212223240090</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

try:
    dataset = pd.read_csv('Churn_Modelling.csv')
    print("STEP 2: Dataset imported successfully.")
    print("First 5 rows of the dataset:")
    print(dataset.head())
    print("\n")

    
    X = dataset.iloc[:, 3:13].values 
    y = dataset.iloc[:, 13].values  

    print("Independent variables (X) selected:")
    print(X[:5, :]) 
    print("\nDependent variable (y) selected:")
    print(y[:5]) 
    print("\n")

except FileNotFoundError:
    print("Error: 'Churn_Modelling.csv' not found. Please make sure the file is in the correct directory.")
    exit()

print("STEP 3: Checking for missing data...")
missing_values = dataset.isnull().sum().sum()
if missing_values == 0:
    print("No missing data found in the dataset.\n")
else:
    print(f"Found {missing_values} missing values. Consider handling them.\n")

print("STEP 4: Encoding categorical data...")

labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])
print("Gender column encoded (e.g., Female -> 0, Male -> 1).")
print("X after encoding Gender:")
print(X[:5, :])
print("\n")


ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float64)


X = X[:, 1:] 
print("Geography column encoded using OneHotEncoder and one column removed to avoid dummy variable trap.")
print("X after encoding Geography:")
print(X[:5, :])
print("\n")

print("STEP 5: Splitting the dataset into Training set and Test set...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}\n")


print("STEP 6: Normalizing the data using Feature Scaling...")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Training and Test sets have been scaled.")
print("Scaled X_train (first 5 rows):")
print(X_train[:5, :])
print("\nScaled X_test (first 5 rows):")
print(X_test[:5, :])
print("\n")

print("Data preprocessing complete!")

```

## OUTPUT:


<img width="715" height="642" alt="image" src="https://github.com/user-attachments/assets/911dff34-dd94-4ed8-b92d-bb80086651be" />


<img width="413" height="61" alt="image" src="https://github.com/user-attachments/assets/80b5cec9-8e76-402e-8014-252618522218" />


<img width="947" height="536" alt="image" src="https://github.com/user-attachments/assets/ce4d0ef8-2a01-4d5d-9e0f-02f6e8972d04" />


<img width="703" height="132" alt="image" src="https://github.com/user-attachments/assets/aa4a8ea9-bd99-4404-b177-33fea63cd280" />


<img width="825" height="563" alt="image" src="https://github.com/user-attachments/assets/9850952b-e693-42b8-b496-07f168ab3fcc" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


