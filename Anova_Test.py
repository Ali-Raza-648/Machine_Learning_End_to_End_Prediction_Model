#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from scipy.stats import f_oneway
anova_frame = pd.read_csv('Anova_Data .csv')
model1=list(anova_frame['Linear_Reg'])
model2=list(anova_frame['Rndm_Frst_Reg'])
model3=list(anova_frame['Grd_Bst_Reg'])

#Conduct the one_way Anova

#Linear and Random Forest
print('Linear & Random Forest :\n')

print(f_oneway(model1,model2))

#Linear and Gradient Boost
print("")
print('Linear & Gradient Boost :\n')

print(f_oneway(model1,model3))

#Random Forest and Gradient Boost
print("")
print('Random Forest  & Gradient Boost :\n')

print(f_oneway(model2,model3))



# corr_matrix = anova_frame.corr()

# from pandas.plotting import scatter_matrix

# scatter_matrix(anova_frame,figsize=(10,8),color='purple')

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import statsmodels.api as sm
# import numpy as np

# # Create the first model
# X = np.array(range(1, 11)).reshape((-1, 1))
# y = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 10.0, 11.1])
# model1 = sm.OLS(y, X).fit()

# # Create the second model
# X2 = sm.add_constant(X)
# model2 = sm.OLS(y, X2).fit()
# from statsmodels.stats.anova import anova_lm
# aov_table = anova_lm(model1, model2)
# print(aov_table)


# In[2]:


import matplotlib.pyplot as plt
rmse_rf = [0,3.140039434465755,2.6734974991663614,2.554974950953531,2.943172311655837,2.9200791924528473]
rmse_gb = [0,2.6365355138386866,2.2951649842678656,2.234468878351084,2.595598762989817,2.622463964260096]
x_both= [0,20,40,60,80,100]
plt.figure(1)
plt.plot(x_both,rmse_rf,label='Random Forest',marker='o',color='red',linestyle='--')
plt.plot(x_both,rmse_gb,label='Gradient Boosting',marker='o',color='black',linestyle='--')
plt.title('Random Forest Vs Gardient Boosting')
plt.xlabel('Test Data (%)')
plt.ylabel('RMSE')
plt.grid()
plt.legend()

