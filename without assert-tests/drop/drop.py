# imports

# The Libraries are needed to perfornm all the functions
import csv
import json
import numpy as np
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")






# ////////////////////////////////// Step 3.a /////////////////////////////////////////
## loading the dataset
def load_data():
    dataset = pd.read_csv('patient_data.csv', encoding='latin1')
    return dataset
# Test Function
# Load the dataset
dataset=load_data()

# Removing Protected Information
# 4. Remove Protected health information (PHI): Names, Addresses etc.
# ////////////////////////////////// Step 4. /////////////////////////////////////////
def remove_phe(dataset):
    dataset = dataset.drop(["first_name", "lastName", "Email", "Address"], axis=1)
    return dataset



# Remove PHE
# numpy array of data
dataset=remove_phe(dataset)

# this storage is needed due to datatype issue
store=dataset['cancerPresent']
# converting columns to numeric
def numeric_issue(df):
    
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
    return df
dataset=numeric_issue(dataset)




# ////////////////////////////////// Step 3.b /////////////////////////////////////////

# Checking insights of dataset
def dataset_properties(dataset):
    
    print(dataset.isnull().sum())

    # Non-Standard Missing Values
    print(dataset['glucose_mg/dl_t1'].unique())
    print(dataset['glucose_mg/dl_t2'].unique())
    print(dataset['glucose_mg/dl_t3'].unique())
    print(dataset['cancerPresent'].unique())
    print("\n")
    print(dataset.describe())





# Checking insights of dataset
# test function
dataset_properties(dataset)




# ////////////////////////////////// Step 5. /////////////////////////////////////////
# Working on Missing Value
def unexpected_with_null(dataset):
    
    # Here unexpected values are taken and are replaced with null
    unexpeted_values = [np.inf,-np.inf,'null', 0.0, -1., 100000.0,'','100000' , '0','n/a','-1']
    dataset = dataset.replace(unexpeted_values, np.NaN)
    dataset['cancerPresent']=store
    print(dataset.to_string())
    return dataset



# Unit Test 
# Unexpected with Null Values
dataset=unexpected_with_null(dataset)




# If atrophy is not present then replace it with 0
def replace_atrophy(dataset):
    dataset['atrophy_present']=dataset['atrophy_present'].replace(np.NaN, 0)
    return dataset



# Test Function
# Clean Atrophy Present
dataset=replace_atrophy(dataset)



# Dropping the null values
def drop_null(dataset):
    dataset=dataset.dropna()
    return dataset



# Unit Test
# Drop all Null Values
dataset=drop_null(dataset)




# Checking insights of dataset
# test function
dataset_properties(dataset)




## Resetting dataset index and replacing dataset types
def dataset_final_cleaned(df):
    #resetting index
    df=df.reset_index(drop=True)
    df['patient_id']=df["patient_id"].astype("int")
    df["glucose_mg/dl_t1"]=df["glucose_mg/dl_t1"].astype("float64")
    df["glucose_mg/dl_t2"]=df["glucose_mg/dl_t2"].astype("float64")
    df["glucose_mg/dl_t3"]=df["glucose_mg/dl_t3"].astype("float64")
    return df


# Test Function
# Final Clean data
dataset=dataset_final_cleaned(dataset)



# Checking insights of dataset
# test function
dataset_properties(dataset)

# ////////////////////////////////// Step 6. /////////////////////////////////////////



# The function that calculates the avearge of three Glucose mg/dl
def average_calculation(dataset):
    dataset['glucose_mg/dl_average']=np.mean(dataset[['glucose_mg/dl_t1','glucose_mg/dl_t2','glucose_mg/dl_t3']].T)
    dataset['glucose_mg/dl_average']=round(dataset['glucose_mg/dl_average'],2)
    return dataset




# Test Function
# Part 6 Average calculation
dataset=average_calculation(dataset)




# Checking insights of dataset
# test function
dataset_properties(dataset)




# ////////////////////////////////// Step 7. /////////////////////////////////////////




## Replace based on condition
# Values below 140 indicates normal
# Value above 200 inidcates diabetes
# Values in between indicates prediabetes
def diabetes_indication(dataset):
    dataset['diabetes_indication']=np.where((dataset['glucose_mg/dl_average'])<140,'normal',(np.where(dataset['glucose_mg/dl_average']>200,'diabetes','prediabetes')))
    return dataset




# Test Function
# Part 7 Diabetes indication
dataset=diabetes_indication(dataset)




# Checking insights of dataset
# test function
dataset_properties(dataset)




# Final print
print(dataset)



# ////////////////////////////////// Step 8. /////////////////////////////////////////
# We choose CSV file
def write_final_csv(df):
    df.to_csv("drop.csv",index=False)
    print("File Wrriten")



# Writing Final CSV
write_final_csv(dataset)

# The graphs are to show final details about the work for verification


# ////////////////////// Visulizations and Verefication. ///////////////////////////////
# Graph 1 to show Cancer and Diabetes relation
plt.figure(figsize=(10,5))
sns.countplot(data=dataset,x='diabetes_indication',hue='cancerPresent').set(title='Cancer and Diabetes')
plt.show()

# Graph 2 to show relation of the dataset

# Corelation to show how dataset columns depends on each other

plt.figure(figsize=(16,8))
corr = dataset.corr()
sns.heatmap(corr, annot=True)
plt.show()





