
## 2. Preprocessing Outliers
## Import library and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

medical = pd.read_csv("medical_dataset_modify.csv")


## Check data
print(medical.columns)
print(medical.describe())


## Copy data
md = medical.copy()


## Create function
## Boxplot
def boxplot(row, col, data, columns):
    plt.figure()
    plt.rcParams["figure.figsize"] = (20, 5*row)
    
    count = row * col
        
    if len(columns) < count:
        count -= 1
    
    color = ['#BBDEFB','#F8BBD0','#EBDEF0']
    
    for i in range(count):
        plt.subplot(row, col, i+1)
        sns.boxplot(columns[i], data=data, palette=[color[i%3]])
    plt.show()
    
## Outliers
def outliers(data):
    Q1 = data.quantile(q=0.25)
    Q3 = data.quantile(q=0.75)
    Q = Q3 - Q1
    
    lower = Q1 - Q * 4
    upper = Q3 + Q * 4
    
    if lower < 0:
        lower = 0
    
    return lower, upper
    
    
## Check outliers : Height, Weight, Waist, BMI
columns = ['HEIGHT', 'WEIGHT', 'WAIST', 'BMI']
boxplot(4,1,md,columns)

print("Weight more than 100kg : %d" % len(md.loc[md['WEIGHT'] >= 100]))
print("Waist more than 90cm : %d" % len(md.loc[md['WAIST'] >= 90]))

print("BMI less than 18.5 : %d" % len(md.loc[md['BMI'] < 18.5]))
print("BMI more than 25 : %d" % len(md.loc[md['BMI'] >= 25]))
print("BMI more than 35 : %d" % len(md.loc[md['BMI'] >=35]))

## Create derived variables
md['BMI_STATE'] = np.nan
md['BMI_STATE'].loc[md['BMI'] < 18.5] = 1 ## lowweight
md['BMI_STATE'].loc[md['BMI'] >= 18.5] = 2 ## normal
md['BMI_STATE'].loc[md['BMI'] >= 23] = 3 ## overweight
md['BMI_STATE'].loc[md['BMI'] >= 25] = 4 ## obesity
md['BMI_STATE'].loc[md['BMI'] >= 35] = 5 ## extreme obesity
print(md['BMI_STATE'].head(10))


## Check outliers : Eyesight
columns = ['SIGHT_LEFT', 'SIGHT_RIGHT']
boxplot(2,1,medical,columns)

print("Lost of left eyesight : %d" % len(md.loc[md['SIGHT_LEFT']==9.9]))
print("Lost of right eyesight : %d" % len(md.loc[md['SIGHT_RIGHT']==9.9]))

s = len(md.loc[(md['SIGHT_LEFT']==9.9) & (md['SIGHT_RIGHT']==9.9)])
print("Lost of both eyesight : %d" % s)

## Change to Zero for blindness
md['SIGHT_LEFT'].loc[md['SIGHT_LEFT']==9.9] = 0
md['SIGHT_RIGHT'].loc[md['SIGHT_RIGHT']==9.9] = 0

columns = ['SIGHT_LEFT', 'SIGHT_RIGHT']
boxplot(2,1,md,columns)


## Check outliers : Blood pressure
columns = ['BP_HIGH', 'BP_LWST']
boxplot(2,1,md,columns)

lower,upper = outliers(md['BP_HIGH'])
print("BP_HIGH lower value : %d" % lower)
print("BP_HIGH less than %d : %d" % (lower, len(md.loc[md['BP_HIGH'] <= lower])))
print("BP_HIGH upper value : %d" % upper)
print("BP_HIGH more than %d : %d" % (upper, len(md.loc[md['BP_HIGH'] >= upper])))

lower,upper = outliers(md['BP_LWST'])
print("BP_LWST lower value : %d" % lower)
print("BP_LWST less than %d : %d" % (lower, len(md.loc[md['BP_LWST'] <= lower])))
print("BP_LWST upper value : %d" % upper)
print("BP_LWST more than %d : %d" % (upper, len(md.loc[md['BP_LWST'] >= upper])))

## Create derived variables
md['BP_STATE'] = np.nan
md['BP_STATE'].loc[md['BP_HIGH'] < 90] = 1 ## hypotension
md['BP_STATE'].loc[md['BP_HIGH'] >= 90] = 2 ## normal 
md['BP_STATE'].loc[(md['BP_HIGH'] >= 120) | (md['BP_LWST'] >= 80)] = 3 ## before hypertension
md['BP_STATE'].loc[(md['BP_HIGH'] >= 140) | (md['BP_LWST'] >= 90)] = 4 ## level1 hypertension
md['BP_STATE'].loc[(md['BP_HIGH'] >= 160) | (md['BP_LWST'] >= 100)] = 5 ## level2 hypertension
md['BP_STATE'].loc[(md['BP_HIGH'] >= 180) | (md['BP_LWST'] >= 110)] = 6 ## hypertension crisis
print(md['BP_STATE'].head(10))



## Check outliers : Cholesterol
columns = ['TOT_CHOLE', 'HDL_CHOLE', 'LDL_CHOLE', 'TRIGLYCERIDE']
boxplot(4,1,md,columns)

lower,upper = outliers(md['TOT_CHOLE'])
print("TOT_CHOLE lower value : %d" % lower)
print("TOT_CHOLE less than %d : %d" % (lower, len(md.loc[md['TOT_CHOLE'] <= lower])))
print("TOT_CHOLE upper value : %d" % upper)
print("TOT_CHOLE more than %d : %d" % (upper, len(md.loc[md['TOT_CHOLE'] >= upper])))

lower,upper = outliers(md['HDL_CHOLE'])
print("HDL_CHOLE lower value : %d" % lower)
print("HDL_CHOLE less than %d : %d" % (lower, len(md.loc[md['HDL_CHOLE'] <= lower])))
print("HDL_CHOLE upper value : %d" % upper)
print("HDL_CHOLE more than %d : %d" % (upper, len(md.loc[md['HDL_CHOLE'] >= upper])))

lower,upper = outliers(md['LDL_CHOLE'])
print("LDL_CHOLE lower value : %d" % lower)
print("LDL_CHOLE less than %d : %d" % (lower, len(md.loc[md['LDL_CHOLE'] <= lower])))
print("LDL_CHOLE upper value : %d" % upper)
print("LDL_CHOLE more than %d : %d" % (upper, len(md.loc[md['LDL_CHOLE'] >= upper])))


print("HDL cholesterol less than 60 : %d" %len(md.loc[md['HDL_CHOLE'] < 60]))
print("HDL cholesterol more than 200 : %d" %len(md.loc[md['HDL_CHOLE'] >= 200]))
print("HDL cholesterol more than 400 : %d" %len(md.loc[md['HDL_CHOLE'] >= 400]))

print("LDL cholesterol more than 160 : %d" %len(md.loc[md['LDL_CHOLE'] >= 160]))
print("LDL cholesterol more than 200 : %d" %len(md.loc[md['LDL_CHOLE'] >= 200]))
print("LDL cholesterol more than 400 : %d" %len(md.loc[md['LDL_CHOLE'] >= 400]))
print("LDL cholesterol more than 1000 : %d" %len(md.loc[md['LDL_CHOLE'] >= 1000]))
print("LDL cholesterol 9999 : %d" %len(md.loc[md['LDL_CHOLE'] == 9999]))

print("triglyceride more than 200 : %d" %len(md.loc[md['TRIGLYCERIDE'] >= 200]))
print("triglyceride more than 400 : %d" %len(md.loc[md['TRIGLYCERIDE'] >= 400]))
print("triglyceride more than 600 : %d" %len(md.loc[md['TRIGLYCERIDE'] >= 600]))
print("triglyceride more than 800 : %d" %len(md.loc[md['TRIGLYCERIDE'] >= 800]))
print("triglyceride 9999 : %d" %len(md.loc[md['TRIGLYCERIDE'] == 9999]))

## Change to Nan for outliers
md['LDL_CHOLE'].loc[md['LDL_CHOLE'] == 9999] = np.nan


lower, upper = outliers(md['TOT_CHOLE'])
upper

## Check outliers : Blood sugar before meals
columns = ['BLDS']
boxplot(1,1,md,columns)

print("BLDS more than 126 : %d" %len(md.loc[md['BLDS'] >= 126]))
print("BLDS more than 200 : %d" %len(md.loc[md['BLDS'] >= 200]))
print("BLDS more than 400 : %d" %len(md.loc[md['BLDS'] >= 400]))

## Create derived variable
md['BLDS_STATE'] = np.nan
md['BLDS_STATE'].loc[md['BLDS'] < 100] = 1 ## normal
md['BLDS_STATE'].loc[md['BLDS'] >= 100] = 2 ## before_diabetes
md['BLDS_STATE'].loc[md['BLDS'] >= 126] = 3 ## diabetes
print(md['BLDS_STATE'].head(10))


## Check ouliers : Hemoglobin
columns = ['HMG']
boxplot(1,1,md,columns)

print("HMG more than 17 : %d" %len(md.loc[md['HMG'] >= 17]))
print("HMG more than 20 : %d" %len(md.loc[md['HMG'] >= 20]))
print("HMG less than 12 : %d" %len(md.loc[md['HMG'] < 12]))
print("HMG less than 8 : %d" %len(md.loc[md['HMG'] < 8]))


## Check outliers : Serum creatinine
columns = ['CREATININE']
boxplot(1,1,md, columns)

print("creatinine more than 1.2 : %d" %len(md.loc[md['CREATININE'] >= 1.2]))
print("creatinine more than 2.0 : %d" %len(md.loc[md['CREATININE'] >= 2]))
print("creatinine more than 20.0 : %d" %len(md.loc[md['CREATININE'] >= 20]))
print("creatinine less than 0.5 : %d" %len(md.loc[md['CREATININE'] < 0.5]))


## Check outliers : AST, ALT
columns = ['SGOT_AST', 'SGPT_ALT']
boxplot(2,1,md, columns)

print("AST 0 : %d" %len(md.loc[md['SGOT_AST'] == 0]))
print("AST more than 40 : %d" %len(md.loc[md['SGOT_AST'] >= 40]))
print("AST more than 100 : %d" %len(md.loc[md['SGOT_AST'] >= 100]))
print("AST more than 200 : %d" %len(md.loc[md['SGOT_AST'] >= 200]))
print("AST 999 : %d" %len(md.loc[md['SGOT_AST'] == 999]))

print("ALT 0 : %d" %len(md.loc[md['SGPT_ALT'] == 0]))
print("ALT more than 40 : %d" %len(md.loc[md['SGPT_ALT'] >= 40]))
print("ALT more than 100 : %d" %len(md.loc[md['SGPT_ALT'] >= 100]))
print("ALT more than 200 : %d" %len(md.loc[md['SGPT_ALT'] >= 200]))
print("ALT 999 : %d" %len(md.loc[md['SGPT_ALT'] == 999]))


## Check outliers : Gamma GTP
columns = ['GAMMA_GTP']
boxplot(1,1,md, columns)

print("Gamma GTP (male) more than 63 : %d" %len(md.loc[(md['GAMMA_GTP'] >= 63) & (md['SEX'] == 1)]))
print("Gamma GTP (female) more than 35 : %d" %len(md.loc[(md['GAMMA_GTP'] >= 35) & (md['SEX'] == 2)]))

print("Gamma GTP more than 100 : %d" %len(md.loc[md['GAMMA_GTP'] >= 100]))
print("Gamma GTP more than 200 : %d" %len(md.loc[md['GAMMA_GTP'] >= 200]))
print("Gamma GTP more than 300 : %d" %len(md.loc[md['GAMMA_GTP'] >= 300]))


## Save dataset
md.to_csv("medical_dataset_outliers.csv", index=False)
md.info()


pd.set_option('display.max_columns', 100)