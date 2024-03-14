#!/usr/bin/env python
# coding: utf-8

# <h5>Customer Analysis<h5>

# In[1]:


# Data manipulation and analysis
import numpy as np
import pandas as pd
import datetime

# Visualization
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

# Preprocessing and machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics

# Miscellaneous
import warnings
import sys


# In[2]:


# Reading the dataset from a CSV file
df = pd.read_csv("marketing_campaign.csv", sep="\t")

# Displaying the first few rows of the DataFrame
df.head()


# In[3]:


# Displaying information about the DataFrame
df.info()


# In[4]:


# Checking for NaN values in each column
df.isnull().sum()


# In[5]:


# Dropping rows with NaN values
df = df.dropna()


# In[9]:


# Convert "Dt_Customer" to datetime format
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format='%d-%m-%Y')

# Extract dates
dates = df["Dt_Customer"].dt.date

# Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in the records:", max(dates))
print("The oldest customer's enrolment date in the records:", min(dates))


# In[11]:


#Created a feature "Customer_Duration"
days = []
d1 = max(dates) #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
df["Customer_Duration"] = days
df["Customer_Duration"] = pd.to_numeric(df["Customer_Duration"], errors="coerce")


# In[12]:


df["Customer_Duration"]


# In[13]:


# Value counts for the "Marital_Status" column
print("Total categories in the feature Marital_Status:\n", df["Marital_Status"].value_counts(), "\n")

# Value counts for the "Education" column
print("Total categories in the feature Education:\n", df["Education"].value_counts())


# In[14]:


#Calculating age:
df["Age"] = 2014 - df["Year_Birth"]

# Total spendings:
df["Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

# Deriving living Situation:
df["Living_With"] = df["Marital_Status"].replace({"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone"})

# Feature for Total childrean:
df["Children"] = df["Kidhome"] + df["Teenhome"]

# Feature for Family Size:
df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner": 2}) + df["Children"]

# Feature for Parenthood:
df["Is_Parent"] = np.where(df.Children > 0, 1, 0)

# Segmenting Education Levels:
df["Education"] = df["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})

# Renaming Columns:
df = df.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

#Dropping Redundant Features:
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID",]
df = df.drop(to_drop, axis=1)


# In[15]:


# Displaying descriptive statistics for numeric columns
df.describe()


# In[16]:


#Represents the number of nanoseconds in a day.
df['Customer_Duration'] = df['Customer_Duration'] / (24 * 60 * 60 * 10**9)


# In[17]:


# Assuming 'df' is your DataFrame
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

# Define custom colors
parent_color = "#0000FF"  # Blue
non_parent_color = "#FF0000"  # Red

# Plotting following features
To_Plot = ["Income", "Recency", "Customer_Duration", "Age", "Spent", "Is_Parent"]

print("Relative Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df[To_Plot], hue="Is_Parent", palette={1: parent_color, 0: non_parent_color}, plot_kws={'facecolor': 'white'})
# Taking hue
plt.show()


# In[18]:


# Dropping rows where Age is greater than or equal to 90
df = df[(df["Age"] < 90)]

# Dropping rows where Income is greater than or equal to 600,000
df = df[(df["Income"] < 600000)]


# In[19]:


# Checking columns with object data type (assumed to be categorical)
s = (df.dtypes == 'object')
object_cols = list(s[s].index)

# Printing the list of categorical variables
print("Categorical variables in the dataset:", object_cols)


# In[20]:


#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    df[i]=df[[i]].apply(LE.fit_transform)

print("All features are now numerical")


# In[21]:


# Compute the correlation matrix for the encoded DataFrame
corrmat= df.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(corrmat,annot=True, cmap='viridis', center=0)


# In[22]:


# Creating a copy of the original DataFrame
ds = df.copy()

# Creating a subset of the DataFrame by dropping certain columns
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)

# Scaling the data using StandardScaler
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)

# Displaying the head of the original DataFrame
ds.head()


# In[23]:


# Initiating PCA to reduce dimensions to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=["col1", "col2", "col3"])

# Displaying descriptive statistics for the transformed DataFrame
PCA_ds.describe().T


# In[24]:


# Extracting col1 and col2 from PCA_ds
x = PCA_ds["col1"]
y = PCA_ds["col2"]

# Scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(x, y, c="blue", marker="o", alpha=0.7)
plt.title("Scatter Plot of Data in the Reduced Dimension")
plt.xlabel("col1")
plt.ylabel("col2")
plt.show()


# <h5>Clustering <h5>

# In[25]:


# Quick examination of elbow method to find the number of clusters
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()


# In[26]:


#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
df["Clusters"]= yhat_AC


# In[27]:


# Assuming x and y are the columns in your PCA_ds dataframe
x = PCA_ds["col1"]
y = PCA_ds["col2"]

# Plotting the clusters in a 2D scatter plot
plt.figure(figsize=(10, 8))

# Use the "Set2" colormap
cmap = plt.cm.get_cmap("Set2")

scatter = plt.scatter(x, y, c=PCA_ds["Clusters"], cmap=cmap, marker='o', s=40)

# Add a colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')

plt.title("The Plot Of The Clusters")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()


# In[28]:


# Assuming df is your DataFrame and "Clusters" is the column with cluster assignments

# Define the "Set2" color palette
pal = sns.color_palette("Set2")

# Plotting countplot of clusters with the "Set2" palette
plt.figure(figsize=(8, 6))
pl = sns.countplot(x=df["Clusters"], palette=pal)
pl.set_title("Distribution Of The Clusters")
plt.show()


# In[29]:


# Assuming df is your DataFrame and "Clusters" is the column with cluster assignments

# Define the "husl" palette with 9 colors
pal = sns.color_palette("Set2")

# Create a scatter plot
plt.figure(figsize=(10, 8))
pl = sns.scatterplot(data=df, x="Spent", y="Income", hue="Clusters", palette=pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# In[30]:


plt.figure()
pl=sns.swarmplot(x=df["Clusters"], y=df["Spent"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=df["Clusters"], y=df["Spent"], palette=pal)
plt.show()


# In[31]:


#Creating a feature to get a sum of accepted promotions 
df["Total_Promos"] = df["AcceptedCmp1"]+ df["AcceptedCmp2"]+ df["AcceptedCmp3"]+ df["AcceptedCmp4"]+ df["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=df["Total_Promos"],hue=df["Clusters"], palette= pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()


# In[32]:


#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=df["NumDealsPurchases"],x=df["Clusters"], palette= pal)
pl.set_title("Number of Deals Purchased")
plt.show()


# In[46]:


Personal = ["Kidhome", "Teenhome", "Children", "Family_Size", "Is_Parent", "Education", "Living_With"]

# Set the palette
pal = sns.color_palette("Set2")

# Assuming df is your DataFrame containing the personal attributes, 'Spent', and 'Clusters' columns

for i in Personal:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=i, y="Spent", hue="Clusters", data=df, palette=pal)
    plt.title(f"Spent vs {i} by Clusters")
    plt.xlabel(i)
    plt.ylabel("Spent")
    plt.legend(title="Clusters")
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




