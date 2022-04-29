# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:02:34 2020

Mahsa Mozafarinia

Homework 6

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

data=pd.read_csv("D:/Old_Data/math/Data science toseeh/Files/Spine.csv")
data.dtypes
summary=data.describe(include="all")

np.sum(data.isna())
data.shape
data.agg({"PI":("mean","std"),"GS":("mean","median","std")})


a=data.groupby("Categories").agg(("mean","std"))

len(data[data["Categories"]=="Hernia"])
len(data[data["Categories"]=="Normal"])
len(data[data["Categories"]=="Spondylolisthesis"])
#PI-Hernia
47.6384+1.96*(10.6971/np.sqrt(60))
47.6384-1.96*(10.6971/np.sqrt(60))
#PI-Normal
51.6852+1.96*(12.3682/np.sqrt(100))
51.6852-1.96*(12.3682/np.sqrt(100))
#PI-Spondy
71.5142+1.96*(15.1093/np.sqrt(150))
71.5142-1.96*(15.1093/np.sqrt(150))

a1=np.mean(data["PI"].loc[data["Categories"]=="Normal"])
a2=np.mean(data["PI"].loc[data["Categories"]=="Hernia"].dropna())
a3=np.mean(data["PI"].loc[data["Categories"]=="Spondylolisthesis"])

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"PI",alpha=0.4,bins=20)
plt.legend()
plt.axvline(a1,color='green')
plt.axvline(a2,color='blue')
plt.axvline(a3,color='red')
#########################################################
#PT-Hernia
17.3988+1.96*(7.01671/np.sqrt(60))
17.3988-1.96*(7.01671/np.sqrt(60))
#PI-Normal
12.8214+1.96*(6.7785/np.sqrt(100))
12.8214-1.96*(6.7785/np.sqrt(100))
#PI-Spondy
20.748+1.96*(11.5062/np.sqrt(150))
20.748-1.96*(11.5062/np.sqrt(150))

a1=np.mean(data["PT"].loc[data["Categories"]=="Normal"])
a2=np.mean(data["PT"].loc[data["Categories"]=="Hernia"].dropna())
a3=np.mean(data["PT"].loc[data["Categories"]=="Spondylolisthesis"])

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"PT",alpha=0.4,bins=20)
plt.legend()
plt.axvline(a1,color='green')
plt.axvline(a2,color='blue')
plt.axvline(a3,color='red')
##############################################################
#LL-Hernia
35.46+1.96*(9.76/np.sqrt(60))
35.46-1.96*(9.76/np.sqrt(60))
#LL-Normal
43.54+1.96*(12.36/np.sqrt(100))
43.54-1.96*(12.36/np.sqrt(100))
#LL-Spondy
64.11+1.96*(16.39/np.sqrt(150))
64.11-1.96*(16.39/np.sqrt(150))

a1=np.mean(data["LL"].loc[data["Categories"]=="Normal"])
a2=np.mean(data["LL"].loc[data["Categories"]=="Hernia"].dropna())
a3=np.mean(data["LL"].loc[data["Categories"]=="Spondylolisthesis"])

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"LL",alpha=0.4,bins=20)
plt.legend()
plt.axvline(a1,color='green')
plt.axvline(a2,color='blue')
plt.axvline(a3,color='red')
################################################################
#GS-Hernia
2.48+1.96*(5.53/np.sqrt(60))
2.48-1.96*(5.53/np.sqrt(60))
#Normal
2.18+1.96*(6.30/np.sqrt(100))
2.18-1.96*(6.30/np.sqrt(100))
#Spondy
51.89+1.96*(40.10/np.sqrt(150))
51.89-1.96*(40.10/np.sqrt(150))

a1=np.mean(data["GS"].loc[data["Categories"]=="Normal"])
a2=np.mean(data["GS"].loc[data["Categories"]=="Hernia"].dropna())
a3=np.mean(data["GS"].loc[data["Categories"]=="Spondylolisthesis"])

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"GS",alpha=0.4,bins=20)
plt.legend()
plt.axvline(a1,color='green')
plt.axvline(a2,color='blue')
plt.axvline(a3,color='red')
############################################################################
#SS-Hernia
30.23+1.96*(7.55/np.sqrt(60))
30.23-1.96*(7.55/np.sqrt(60))
#Normal
38.86+1.96*(9.62/np.sqrt(100))
38.86-1.96*(9.62/np.sqrt(100))
#Spondy
50.76+1.96*(12.31/np.sqrt(150))
50.76-1.96*(12.31/np.sqrt(150))

a1=np.mean(data["SS"].loc[data["Categories"]=="Normal"])
a2=np.mean(data["SS"].loc[data["Categories"]=="Hernia"].dropna())
a3=np.mean(data["SS"].loc[data["Categories"]=="Spondylolisthesis"])

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"SS",alpha=0.4,bins=20)
plt.legend()
plt.axvline(a1,color='green')
plt.axvline(a2,color='blue')
plt.axvline(a3,color='red')
####################################################################################
#SS-Hernia
116.47+1.96*(9.35/np.sqrt(60))
116.47-1.96*(9.35/np.sqrt(60))
#Normal
123.89+1.96*(9.01/np.sqrt(100))
123.89-1.96*(9.01/np.sqrt(100))
#Spondy
114.51+1.96*(15.58/np.sqrt(150))
114.51-1.96*(15.58/np.sqrt(150))

a1=np.mean(data["PR"].loc[data["Categories"]=="Normal"])
a2=np.mean(data["PR"].loc[data["Categories"]=="Hernia"].dropna())
a3=np.mean(data["PR"].loc[data["Categories"]=="Spondylolisthesis"])

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"PR",alpha=0.4,bins=20)
plt.legend()
plt.axvline(a1,color='green')
plt.axvline(a2,color='blue')
plt.axvline(a3,color='red')
####################################################################################
grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"PT",alpha=0.4,bins=20)
plt.legend()

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"LL",alpha=0.4,bins=20)
plt.legend()

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"SS",alpha=0.4,bins=20)
plt.legend()

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"PR",alpha=0.4,bins=20)
plt.legend()

grid=sns.FacetGrid(hue="Categories",data=data)
grid.map(plt.hist,"GS",alpha=0.4,bins=20)
plt.legend()




#e
sns.boxplot(x="Categories", y="GS", data=data)

I=[]
for x in np.unique(data["Categories"]):#Repeat for each category
 I.append(data['GS'].loc[data['Categories']==x].dropna())#Add a component to the list l. The component is the vector of Ratings for category x 
plt.boxplot(I,notch=True)#Boxplots for Rreviews associated with each category
plt.xticks(np.arange(1,4),np.unique(data['Categories']))#Add and rotate the text
plt.xlabel('Categories')
plt.ylabel('Gs')

df=data.drop("Categories",axis=1)
df=scale(df)
df.shape
pca=PCA()
pca.fit(df)
w=pca.components_.T
w.shape

pd.DataFrame(w[:,:],index=data.columns[:-1],columns=['W1','W2','W3','W4','W5','W6'])
pd.DataFrame(pca.explained_variance_ratio_,index=range(1,7),columns=['Explained Variability'])

pca.explained_variance_ratio_.cumsum()

plt.figure(2)
plt.bar(range(1,7),pca.explained_variance_,color="blue",edgecolor="Red")


#e
y=pca.transform(df)
y
y.shape

scatter=sns.scatterplot(y[:,0],y[:,1],hue=data["Categories"])
plt.axvline(np.mean(data["GS"]))
x1=df[y[:,0]>8]
x2=data[y[:,0]>8]




b=data[data["GS"]>400]
data.agg("std")
data.agg("mean")

plt.figure()
plt.scatter(data["GS"],data["PR"])