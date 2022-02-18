# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:17:14 2022

@author: mka.msc
"""

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import warnings
import os
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import scipy.stats as ss
import torch
from progressbar import ProgressBar as pb
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
# Working directory

os.chdir("C:/Arbeit/Paper/Alfonso_Steffe_ML/Otterbach")


# Load data
lfsat=pd.read_stata('lfsat_V12_2019-02-13c.dta').dropna()

# Clean up

cleanup_nums = {"lfsat": {"1.0": 1, "2.0": 2, "3.0":3, "4.0":4, "5.0":5, "6.0":6, "7.0":7, "8.0":8, "9.0":9, "[0] Completely dissatisfied    0": 0,"[10] Completely satisfied    10": 10 }}
lfsat=lfsat.replace(cleanup_nums)
# Exclude columns

l=["nkids","educ","lfsat","behinderung","age","unempl","nilf","eigenheim",
   "care","married","female","time","mode","ghealth","bhealth","rhhinc","hhincsat_n"]

for i in range(1,19):
    l+=["dagecat" + str(i)]
    
lfsat_pe=lfsat.loc[:,l]
lfsat_pe=lfsat_pe.drop(["bhealth"],axis=1)
lfsat_pe=lfsat_pe.drop(["dagecat1"],axis=1)
lfsat_pe=lfsat_pe.drop(["dagecat2"],axis=1)
lfsat_pe=lfsat_pe[lfsat_pe.age>=20]
lfsat_pe=lfsat_pe[lfsat_pe.age<=70]

lfsat_pe=lfsat_pe[lfsat_pe.behinderung!=300]


# Shuffle data set
lfsat_pe=lfsat_pe.sample(frac=1, random_state=100)

# Split in Training and Test data
random.seed(1)
train, test=train_test_split(lfsat_pe,test_size=0.2, random_state=10)

# Graphical analyis 
# lfsat_pe.groupby("age").lfsat.mean().plot()



x_train=train.drop(["lfsat"],axis=1).to_numpy().astype(np.float64)
y_train=train.lfsat.to_numpy().astype(np.float64)

x_test=test.drop(["lfsat"],axis=1).to_numpy().astype(np.float64)
y_test=test.lfsat.to_numpy().astype(np.float64)

x_train_s=np.array(ss.zscore(x_train)).astype(np.float64)
x_test_s=np.array(ss.zscore(x_test)).astype(np.float64)


# Model tuning

results=[[],[],[],[]]
hyper_params=[[50,60,70,80,90,100,150,200],[2,3,4,5,8,10,15,20]]
it=0
for i in range(0,8,1):
    for j in range(0,8,1):
        regr=RandomForestRegressor(n_estimators=hyper_params[0][i], max_depth=hyper_params[1][j],random_state=1)
        regr.fit(x_train_s,y_train)
        y_hat=regr.predict(x_test_s)
        test_error=np.mean((y_test-y_hat)**2)
        results[0].append(it)
        results[1].append(hyper_params[0][i])
        results[2].append(hyper_params[1][j])
        results[3].append(test_error)
        it+=1

print("The lowest test error is", "{:.2f}".format(min(results[3])))
print("n_estimators=","{:.2f}".format(results[1][results[3].index(min(results[3]))]))
print("max_depth=","{:.2f}".format(results[2][results[3].index(min(results[3]))]))


regr=RandomForestRegressor(n_estimators=200,max_depth=15,random_state=1)
regr.fit(x_train_s,y_train)
y_hat=regr.predict(x_test_s)

final=test
final["pre"]=y_hat
os.chdir("C:/Arbeit/Paper/Alfonso_Steffe_ML/Otterbach/final_plots")

pre=final.groupby("age",as_index=False)["pre","lfsat"].mean()
plt.plot(pre.age,pre.pre, label="predicted values")
plt.plot(pre.age,pre.lfsat, label="actual values")
plt.ylabel("life Satisfaction")
plt.xlabel("age")
plt.legend(("predicted values", "actual values"))
plt.savefig("baseline_RF.png",dpi=1000)
plt.show()



cohorts=[]
for i in range(3,19):
    cohorts+=["dagecat" + str(i)]
    
final["cohort"]=0

k=3
for i in cohorts:
    final.loc[final[i]==1, "cohort"]=k
    k+=1



coh=final.groupby("cohort",as_index=False)["pre","lfsat"].mean()
coh_tick=["1925","1945","1960","1985"]

plt.ylabel("life Satisfaction")
plt.legend(("predicted values", "actual values"))
plt.plot(coh.cohort,coh.pre, label="predicted values")
plt.plot(coh.cohort,coh.lfsat, label="actual values")
plt.legend(("predicted values", "actual values"))
plt.xlabel("cohort")
plt.xticks([4,8,12,16],coh_tick)
plt.savefig("baseline_RF_cohort.png",dpi=1000)
plt.show()

##############################################################################
##############################################################################
##############################################################################
##############################################################################


# All models for comparison
random.seed(1)
train, test=train_test_split(lfsat_pe,test_size=0.2, random_state=10)
regr=RandomForestRegressor(n_estimators=200,max_depth=15,random_state=1)
final=test.copy()
# Full model


x_train=train.drop(["lfsat"],axis=1).to_numpy().astype(np.float64)
y_train=train.lfsat.to_numpy().astype(np.float64)

x_test=test.drop(["lfsat"],axis=1).to_numpy().astype(np.float64)
y_test=test.lfsat.to_numpy().astype(np.float64)

x_train_s=np.array(ss.zscore(x_train)).astype(np.float64)
x_test_s=np.array(ss.zscore(x_test)).astype(np.float64)

regr.fit(x_train_s,y_train)
y_hat=regr.predict(x_test_s)


final["pre"]=y_hat





# No age model
random.seed(1)
train, test=train_test_split(lfsat_pe,test_size=0.2, random_state=10)

x_train=train.drop(["lfsat","age"],axis=1).to_numpy().astype(np.float64)
y_train=train.lfsat.to_numpy().astype(np.float64)

x_test=test.drop(["lfsat","age"],axis=1).to_numpy().astype(np.float64)
y_test=test.lfsat.to_numpy().astype(np.float64)

x_train_s=np.array(ss.zscore(x_train)).astype(np.float64)
x_test_s=np.array(ss.zscore(x_test)).astype(np.float64)

regr.fit(x_train_s,y_train)
y_hat=regr.predict(x_test_s)

final["pre_noage"]=y_hat


# No cohort model
random.seed(1)
train, test=train_test_split(lfsat_pe,test_size=0.2, random_state=10)

dl=l[19:]
dl.append("lfsat")

x_train=train.drop(dl,axis=1).to_numpy().astype(np.float64)
y_train=train.lfsat.to_numpy().astype(np.float64)

x_test=test.drop(dl,axis=1).to_numpy().astype(np.float64)
y_test=test.lfsat.to_numpy().astype(np.float64)

x_train_s=np.array(ss.zscore(x_train)).astype(np.float64)
x_test_s=np.array(ss.zscore(x_test)).astype(np.float64)

regr.fit(x_train_s,y_train)
y_hat=regr.predict(x_test_s)


final["pre_nocohort"]=y_hat

# No age and cohort model
random.seed(1)
train, test=train_test_split(lfsat_pe,test_size=0.2, random_state=10)

dl.append("age")

x_train=train.drop(dl,axis=1).to_numpy().astype(np.float64)
y_train=train.lfsat.to_numpy().astype(np.float64)

x_test=test.drop(dl,axis=1).to_numpy().astype(np.float64)
y_test=test.lfsat.to_numpy().astype(np.float64)

x_train_s=np.array(ss.zscore(x_train)).astype(np.float64)
x_test_s=np.array(ss.zscore(x_test)).astype(np.float64)

regr.fit(x_train_s,y_train)
y_hat=regr.predict(x_test_s)
final["pre_noagenocohort"]=y_hat


# Final figures

final["p_1"]=final.pre-final.pre_noage
final["p_2"]=final.pre-final.pre_nocohort
final["p_3"]=final.pre-final.pre_noagenocohort
aplot=final.groupby("age", as_index=False)["p_1","p_2","p_3"].mean()
plt.plot(aplot.age,aplot.p_1, label="No age")
plt.plot(aplot.age,aplot.p_2, label="No Cohorts")
plt.plot(aplot.age,aplot.p_3, label="No Age and no Cohorts")
plt.ylabel("life Satisfaction")
plt.xlabel("age")
plt.legend(("No age", "No Cohorts", "No Age and no Cohorts"))
plt.savefig("difference_age_RF.png",dpi=1000)
plt.show()