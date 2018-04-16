
## 1. Data Exporation
## Import library and data
import pandas as pd
import numpy as np

medical = pd.read_csv("NHIS_OPEN_GJ_2014.CSV", encoding="CP949")


## Check Data
print(medical.head(10))
print(medical.columns)


## Change the name of variables
names = ['YEAR', 'ID', 'SEX', 'AGE_GROUP', 'CITY_CODE', 
         'HEIGHT', 'WEIGHT', 'WAIST', 'SIGHT_LEFT', 'SIGHT_RIGHT', 
         'HEAR_LEFT', 'HEAR_RIGHT', 'BP_HIGH', 'BP_LWST', 'BLDS', 
         'TOT_CHOLE', 'TRIGLYCERIDE', 'HDL_CHOLE', 'LDL_CHOLE', 'HMG', 
         'OLIG_PROTE_CD', 'CREATININE', 'SGOT_AST', 'SGPT_ALT', 'GAMMA_GTP', 
         'SMK_STAT_TYPE_CD', 'DRK_YN', 'HCHK_OE_INSPEC_YN', 
         'CRS_YN', 'TTR_YN', 'DATA_STD_DT']
medical.columns = names
print(medical.columns)


## Check data structure
print(medical.shape)
print(medical.info())
print(medical.describe())


## Copy data
md = medical.copy()


## Remove unnecessary variables
del md['YEAR']
del md['DATA_STD_DT']
del md['HCHK_OE_INSPEC_YN']
del md['TTR_YN']
del md['CRS_YN']


## Create derived variables
md['BMI'] = np.nan
md['BMI'] = md['WEIGHT'] / np.square( md['HEIGHT'] * 0.01 )


## Check correlation
print(md.corr(method='pearson'))


## Save dataset
md.to_csv("medical_dataset_modify.csv", index=False)

