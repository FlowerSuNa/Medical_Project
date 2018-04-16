
##2. Draw Simple Graph
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

medical = pd.read_csv("medical_dataset_modify.csv")


## Check Data
print(medical.head(10))
print(medical.columns)


## Scatter plot and histogram
c = ['AGE_GROUP', 'WAIST','BP_HIGH', 'BP_LWST', 'BLDS', 
     'TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE', 'HMG', 
     'OLIG_PROTE_CD', 'CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP', 'BMI']

pd.plotting.scatter_matrix(medical[c], marker='o',
                           figsize=(20,20),
                           s=10, alpha=.1)


## Create function
## 'Boxplot'
def countplot(row, col, data, columns):
    plt.figure()
    plt.rcParams['figure.figsize'] = (20,10 * row)
    
    count = row * col
    
    if len(columns) < count:
        count -= 1
    
    color = ['#BBDEFB','#F8BBD0','#EBDEF0']
    
    for i in range(count):
        plt.subplot(row, col, i+1)
        ax = sns.countplot(columns[i], data=data, palette=color)
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
                    fontsize=12, ha='center', va='bottom')
    plt.show()
    
## 'Histogram'
def histogram(row, col, data, columns, bins):
    plt.figure()
    plt.rcParams['figure.figsize'] = (20,10 * row)
    
    count = row * col
    
    if len(columns) < count:
        count -= 1
    
    for i in range(count):
        plt.subplot(row, col, i+1)
        sns.distplot(data[columns[i]], bins=bins[i], kde=False)
    plt.show()   
    
    
## Barplot : Sex, Age, City Code, BMI State
columns = ['SEX','AGE_GROUP','CITY_CODE','BMI_STATE']
countplot(2,2,medical,columns)


## Histogram : Height, Weight, Waist, BMI
columns = ['HEIGHT', 'WEIGHT', 'WAIST','BMI']
histogram(2,2,medical,columns,[10,20,20,20])


## Histogram : Eyesight
columns = ['SIGHT_LEFT','SIGHT_RIGHT']
histogram(1,2,medical.loc[medical[]],columns,[30,3])


## Barplot : Hearing
columns = ['HEAR_LEFT','HEAR_RIGHT']
countplot(1,2,medical,columns)


## Barplot : Smoke, Drink
columns = ['SMK_STAT_TYPE_CD','DRK_YN']
countplot(1,2,medical,columns)