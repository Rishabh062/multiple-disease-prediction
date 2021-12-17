import numpy as np 
import pickle
import pandas as pd 

# Load the data set
df = pd.read_csv("kaggle_diabetes.csv")

# remane the DiabetesPredictionFunction as DPF for making our work easier
df = df.rename(columns = {'DiabetesPredictionFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN(Not a number)
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean ,median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(),inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(),inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(),inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(),inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# Create Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20)
classifier.fit(X_train,y_train)

# Creating a pickel file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier,open(filename,'wb'))
print("Successfull")

