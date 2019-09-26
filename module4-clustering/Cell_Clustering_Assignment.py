
#%%
import matplotlib.pyplot as pyplot
pyplot.rcParams['figure.facecolor'] = '#002B36'
pyplot.rcParams['axes.facecolor'] = 'black'

#%% [markdown]
# # K-Means Clustering
#%% [markdown]
# # 1) Use the "Breast Cancer Wisconsin (Diagnostic) Data Set" from Kaggle to try and cluster types of cancer cells. 
# 
# Here's the original dataset for your reference:
# 
# <https://www.kaggle.com/uciml/breast-cancer-wisconsin-data>
#%% [markdown]
# ## This is a supervised learning dataset
# 
# (Because it has **labels** - The "diagnosis" column.)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # You don't necessarily have to use this
from sklearn.cluster import KMeans # You don't necessarily have to use this
from sklearn.preprocessing import StandardScaler # You don't necessarily have to use this

df = pd.read_csv("https://raw.githubusercontent.com/ryanleeallred/datasets/master/Cancer_Cells.csv")
print(df.shape)
df.head()

#%% [markdown]
# ## Now it's an unsupervised learning dataset
# 
# (Because we've removed the diagnosis label) - Use this version.

#%%
train = df.drop('diagnosis', axis=1).drop('Unnamed: 32', axis=1).drop('id', axis=1)
# Dropping the 'id' column sharply increases the accuracy of the
# non-normalized 2-cluster K-Means result
# It doesn't really change anything else
# But is probably still good practice.
train.head()

#%%
train.isna().sum()

#%%
for col in train.columns:
	print(col, train[col].max())

#%% [markdown]
# ## Let's do it!
# 
# - You might want to do some data exploration to see if you can find specific columns that will help you find distinct clusters of cells
# - You might want to use the elbow method to decide on the number of clusters to use.
# 

#%%
# Perform K-Means Clustering on the Dataset
import numpy
import matplotlib.pyplot as pyplot

variances = []
stddevs = []
kmeans = []

for k in range(1, 11):
	print(f'Running KMeans(n_clusters={k})')
	kmeans.append(KMeans(n_clusters=k))
	kmeans[-1].fit(train)
	variances.append(kmeans[-1].inertia_/k)
	stddevs.append(variances[-1]**.5)

#%%

pyplot.plot(range(1, 11), stddevs)
pyplot.grid()
pyplot.title('Standard deviation by cluster')
pyplot.show()

#%%
kmeans[2].cluster_centers_
stddevs

#%% [markdown]
# ## Check you work: 
# 
# This is something that in a truly unsupervised learning situation **WOULD NOT BE POSSIBLE**. But for educational purposes go back and grab the true dianosis column (label) from the original dataset. Take your cluster labels and compare them to the original diagnosis column. You can make scatterplots for each to see how they compare or you can calculate a percent accuracy score like: 
# \begin{align}
# \frac{\text{Num Correct Labels}}{\text{Num Total Observations}}
# \end{align}

#%%
#kmeans[1].labels_==1

#%%
len(df['diagnosis']=='B')
#%%
len(kmeans[1].labels_)
#%%
import pandas
pandas.Series(kmeans[1].labels_).value_counts()
#%%
df['diagnosis'].value_counts()
#%%
import scipy.stats as stats
result = pandas.DataFrame(numpy.transpose([df['diagnosis']=='B',kmeans[1].labels_==stats.mode(kmeans[1].labels_)[0][0]]), columns=['diagnosis', 'kmeans_2_clusters'])
# (result['diagnosis']==(kmeans[1].labels_==1)).value_counts()
result['correct'] = result['diagnosis']==result['kmeans_2_clusters']
result['correct'].value_counts()

#%%
percent_correct = result['correct'].value_counts()[True]/len(result['correct'])*100
print(f'Got {percent_correct:5.4}% correct.')
naive_percent_correct = (result['diagnosis']==True).value_counts()[True]/len(result['diagnosis'])*100
print(f'Naive estimation (all diagnosis = \'B\') would get {naive_percent_correct:5.4}%')

#%%

# Testing how good the data is if we just normalize it first
from sklearn.preprocessing import StandardScaler

target = df['diagnosis'].replace({'B':1,'M':0})
# target.value_counts()
scaler = StandardScaler()
processed = scaler.fit_transform(train,y=target)

kmeans=KMeans(n_clusters=2)
kmeans.fit(processed)


result['kmeans_normalized_2_clusters'] = kmeans.labels_==stats.mode(kmeans.labels_)[0][0]
result['normalized_correct'] = result['diagnosis']==result['kmeans_normalized_2_clusters']
result['normalized_correct'].value_counts()

#%%
percent_correct = result['normalized_correct'].value_counts()[True]/len(result['normalized_correct'])*100
print(f'Got {percent_correct:5.4}% correct for 2 clusters on normalized data.')

#%% [markdown]
# # 2) Perform PCA on your dataset first and *then* use k-means clustering. 
# 
# - You need to standardize your data before PCA.
# - First try clustering just on PC1 and PC2 so that you can make a scatterplot of your clustering.
# - Then use use a scree plot to decide how many principal components to include in your clustering, and use however many principal components you need in order to retain 90% of the variation of the original dataset
# 
# 

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

target = df['diagnosis'].replace({'B':1,'M':0})
target.value_counts()

#%%

scaler = StandardScaler()
processed = scaler.fit_transform(train,y=target)

pca = PCA()
pca.fit(processed)

print(f'eigenvectors: {pca.components_}')
print(f'eigenvalues: {pca.explained_variance_}')
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
print(pca)

projected = pca.transform(processed)
print(f'projected: {projected}')

#%%
variance_sum = numpy.cumsum(pca.explained_variance_ratio_)
pyplot.plot(range(1,len(variance_sum)+1), variance_sum)
pyplot.plot(range(1,len(variance_sum)+1),  pca.explained_variance_ratio_)
pyplot.xticks(range(1,len(variance_sum)+1,2))
pyplot.grid()
pyplot.axhline(y=0.9,linestyle='--')
pyplot.show()

#%%
for i in range(len(variance_sum)):
	print(f'PC{i+1} explains {variance_sum[i]*100:5.4}% of variation')
	if variance_sum[i] > 0.9:
		print(f'Need PC{i+1}')
		break

#%%
# PC1 clustering
import scipy.stats as stats
processed = scaler.fit_transform(train,y=target)
pc1 = PCA(1)
pc1.fit(processed)
projected_1 = pc1.transform(processed)
kmeans=KMeans(n_clusters=2)
kmeans.fit(projected_1)

#kmeans.labels_==stats.mode(kmeans.labels_)[0][0]

#%%

result['pc1_cluster_1'] = kmeans.labels_==stats.mode(kmeans.labels_)[0][0]
result['pc1_correct'] = result['diagnosis']==result['pc1_cluster_1']
result['pc1_correct'].value_counts()

#%%
percent_correct = result['pc1_correct'].value_counts()[True]/len(result['pc1_correct'])*100
print(f'Got {percent_correct:5.4}% correct for 2 clusters on PC1.')

#%%
# PC2 clustering
processed = scaler.fit_transform(train,y=target)
pc2 = PCA(2)
pc2.fit(processed)
projected_2 = pc2.transform(processed)
kmeans=KMeans(n_clusters=2)
kmeans.fit(projected_2)

#kmeans.labels_==stats.mode(kmeans.labels_)[0][0]

#%%

result['pc2_cluster_1'] = kmeans.labels_==stats.mode(kmeans.labels_)[0][0]
result['pc2_correct'] = result['diagnosis']==result['pc2_cluster_1']
result['pc2_correct'].value_counts()

#%%
percent_correct = result['pc2_correct'].value_counts()[True]/len(result['pc2_correct'])*100
print(f'Got {percent_correct:5.4}% correct for 2 clusters on PC2.')

#%%
kmeans.cluster_centers_

#%%
projected_2.shape

#%%
pyplot.scatter(projected_2[:,0],projected_2[:,1], alpha=0.2)
pyplot.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='r')
pyplot.title('PC2 components with K-Means centroids')
pyplot.legend(['Data','Centroids'])
pyplot.show()

#%%
# Clustering on PC7
processed = scaler.fit_transform(train,y=target)
pc7 = PCA(7)
pc7.fit(processed)
projected_7 = pc7.transform(processed)
kmeans=KMeans(n_clusters=2)
kmeans.fit(projected_7)

#kmeans.labels_==stats.mode(kmeans.labels_)[0][0]

#%%

result['pc7_cluster_1'] = kmeans.labels_==stats.mode(kmeans.labels_)[0][0]
result['pc7_correct'] = result['diagnosis']==result['pc7_cluster_1']
result['pc7_correct'].value_counts()

#%%
percent_correct = result['pc7_correct'].value_counts()[True]/len(result['pc7_correct'])*100
print(f'Got {percent_correct:5.4}% correct for 2 clusters on PC7.')

#%% [markdown]
# ## Check your work: 
# 
# - Compare your PC1, PC2 clustering scatterplot to the clustering scatterplots you made on the raw data
# - Calculate accuracy scores for both the PC1,PC2 Principal component clustering and the 90% of explained variance clustering.
# 
# How do your accuracy scores when preprocessing the data with PCA compare to the accuracy when clustering on the raw data?


#%% [markdown]
# # Stretch Goals:
# 
# - Study for the Sprint Challenge
# - Work on your Data Storytelling Project
# - Practice your two-minute presentation for your Data Storytelling Project

