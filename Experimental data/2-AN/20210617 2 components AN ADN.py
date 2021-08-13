#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
plt.switch_backend('agg')


# In[2]:


paths = ['20210617']
filenames = os.listdir (paths[0])


# In[3]:


dic_samples = {}

low_ind = 1000
high_ind = 3000
n = 1
for path in paths:
    filenames = os.listdir (path)
    fnames = []

    for f in filenames:
        if f[-3:]  == 'CSV':
            fnames.append(f)
    count = 0;
    
    labels = fnames
    
    for label in labels:
        df = pd.read_csv(path+'/'+label,header = None)
        if count == 0:
            df2 = df 
            df2.columns = ['wn','Absorbance']
        else:
            df.columns = ['wn','Absorbance']
            df2 = pd.concat([df2, df['Absorbance']], axis = 1)
        count = count+1
    df2.set_index('wn', inplace = True)
    df2.columns = labels
    df = df2.transpose()
    df = df.loc[:,df.columns > low_ind]
    df = df.loc[:,df.columns < high_ind]
    
    df['Dataset number'] = np.ones(len(df))*n
    dic_samples[path] = df
    n = n+1


# In[4]:


labels = pd.read_csv(paths[0]+'/Labels.csv')


# In[5]:


labels.set_index('Sample',inplace = True)


# In[6]:


df = dic_samples[paths[0]]

df.sort_index(inplace = True)

labels.sort_index(inplace = True)


# In[7]:


df.index == labels.index


# In[8]:


idx = 0
df.insert(loc=idx, column='F AN', value = labels['F AN'])
idx = 1
df.insert(loc=idx, column='F ADN', value = labels['F ADN'])


# In[9]:


df['F AN'] = labels['F AN'].values
df['F ADN'] = labels['F ADN'].values

df.head()


# In[10]:


labels.index


# In[11]:


C_AN = 4.1/100
C_ADN = 4.2/100

df['AN %'] = df['F AN']*C_AN/(df['F AN'] + df['F ADN'])
df['ADN %'] = df['F ADN']*C_ADN/(df['F AN']  + df['F ADN'])

df.reset_index(inplace = True)

df.drop([ 'F AN', 'F ADN'],axis = 1, inplace = True)

df.drop('index', axis = 1, inplace = True)

df.head()


# In[12]:


df.sort_values(by = 'AN %', inplace = True)


# In[13]:


current_palette = sns.color_palette("mako", len(df))
sns.set_palette(current_palette)
#sns.set_style(style='white')

fig = plt.figure()
ax = plt.axes()
j = 0
for j in range(len(df)):
    ax.plot(df.columns[:-4],df.iloc[j,:-4])


#ax.set_title('IPA in water',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(10, 4)

ax.set_ylabel('Absorbance', fontsize=14, fontname = 'Arial')
ax.set_xlabel('wavenumber [cm$^{-1}$]', fontsize=14, fontname = 'Arial')


#plt.legend(np.round(df['AN %']*0.1,3),fontsize = 14, loc = 'right', bbox_to_anchor=(1.7, 0.5))

#ax.set_xlim(3000,900) 
#ax.set_ylim(0,0.25)

#ax.set_xlim(1600,900) 
#ax.set_ylim(0,0.25)

plt.show()


# In[14]:


X = df.iloc[:,:-2]
y = df.iloc[:,-2:]
len(X)


# In[15]:


pca = PCA()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

color1 = 'darkblue'
color2 = 'darkorange'
pca.explained_variance_
fig = plt.figure()
ax = plt.axes()


n_pca = 20
n = np.arange(1,n_pca+1)
plt.bar(n,pca.explained_variance_ratio_[0:n_pca]*100,
            color = color2,linewidth = 1, edgecolor = 'black')
ax.plot(n,np.cumsum(pca.explained_variance_ratio_[0:n_pca]*100),marker = 'o',color = color1)

ax.set_title('Principal Component Analysis', fontsize=20, fontname = 'Arial')

ax.set_xlabel('Number of Principal Components', fontsize=16, fontname = 'Arial')
ax.set_ylabel('Explained Variance [%]', fontsize=16, fontname = 'Arial')
ax.tick_params(length = 4, labelsize= 11, width =2)
#ax.set_ylim([0.8,1.01])
plt.legend(['Cummulative','Per Component'], fontsize = 14)#, loc = 'upper right')
plt.xticks(np.arange(1,n_pca+1
                    ))
fig.set_size_inches(10, 5)


# In[17]:


X = df.iloc[:,:-3]
y = df.iloc[:,-2:]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    pca = PCA(3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    mlp = LinearRegression()
    mlp.fit(X_train, y_train)
    r_scores_LR.append(mlp.score(X_test, y_test))
    y_predict = mlp.predict(X_test)

    errors = (np.abs(y_predict-y_test)/y_test)*100
    errors_abs = (np.abs(y_predict-y_test))
    mean_error_LR.append(np.mean(errors_abs))
    
    
    mlp =MLPRegressor(hidden_layer_sizes=(10,10),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'identity')
    mlp.fit(X_train, y_train)
    r_scores_ANN.append(mlp.score(X_test,y_test))
    
    y_predict = mlp.predict(X_test)

    errors = (np.abs(y_predict-y_test)/y_test)*100
    errors_abs = (np.abs(y_predict-y_test))
    mean_error_ANN.append(np.mean(errors_abs))
    y_test_mat[j] = y_test
    


for j in range(len(r_scores_LR)):
    score = r_scores_LR[j]
    if score <0:
        r_scores_LR[j] = 0
        
for j in range(len(r_scores_ANN)):
    score = r_scores_ANN[j]
    if score <0:
        r_scores_ANN[j] = 0


# In[18]:


error_AN = []
error_ADN = []
error_PN = []
for item in mean_error_LR:
    error_AN.append(item[0])
    error_ADN.append(item[1])
    

print(np.mean(error_AN)*100)
print(np.mean(error_ADN)*100)


# In[19]:


np.mean(r_scores_LR)


# In[20]:


np.mean(r_scores_ANN)


# In[21]:


X = df.iloc[:,:-3]
y = df.iloc[:,-2:]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    pca = PCA(3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    mlp = LinearRegression()
    mlp.fit(X_train, y_train)
    r_scores_LR.append(mlp.score(X_test, y_test))
    y_predict = mlp.predict(X_test)

    errors = (np.abs(y_predict-y_test)/y_test)*100
    
    errors_abs = np.sqrt(mean_squared_error(y_test, y_predict, multioutput = 'raw_values'))
    mean_error_LR.append(errors_abs)
    
    #errors_abs = (np.abs(y_predict-y_test))
    #mean_error_LR.append(np.mean(errors_abs))
    
    
    mlp =MLPRegressor(hidden_layer_sizes=(15,15),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'identity')
    mlp.fit(X_train, y_train)
    r_scores_ANN.append(mlp.score(X_test,y_test))
    
    y_predict = mlp.predict(X_test)

    
    errors = (np.abs(y_predict-y_test)/y_test)*100
    
    errors_abs = np.sqrt(mean_squared_error(y_test, y_predict, multioutput = 'raw_values'))
    mean_error_ANN.append(errors_abs)
    
    #errors_abs = (np.abs(y_predict-y_test))
    #mean_error_ANN.append(np.mean(errors_abs))
    y_test_mat[j] = y_test
    


for j in range(len(r_scores_LR)):
    score = r_scores_LR[j]
    if score <0:
        r_scores_LR[j] = 0
        
for j in range(len(r_scores_ANN)):
    score = r_scores_ANN[j]
    if score <0:
        r_scores_ANN[j] = 0


# In[22]:


error_AN = []
error_ADN = []
error_PN = []
for item in mean_error_LR:
    error_AN.append(item[0])
    error_ADN.append(item[1])
    

print(np.mean(error_AN)*100)
print(np.mean(error_ADN)*100)


# In[23]:


print(np.mean(pd.DataFrame(mean_error_LR))*100)
print(np.mean(pd.DataFrame(mean_error_ANN))*100)


# In[24]:


print(np.std(pd.DataFrame(mean_error_LR))*100)
print(np.std(pd.DataFrame(mean_error_ANN))*100)


# In[25]:


np.mean(r_scores_LR)


# In[26]:


np.mean(r_scores_ANN)


# In[27]:


np.std(r_scores_LR)
np.std(r_scores_ANN)


# In[33]:


X = df.iloc[:,:-3]
y = df.iloc[:,-2:]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    pca = PCA(3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    mlp = LinearRegression()
    mlp.fit(X_train, y_train)
    r_scores_LR.append(mlp.score(X_test, y_test))
    y_predict = mlp.predict(X_test)

    errors = (np.abs(y_predict-y_test)/y_test)*100
    errors_abs = (np.abs(y_predict-y_test))
    mean_error_LR.append(np.mean(errors_abs))
    
    
    mlp =MLPRegressor(hidden_layer_sizes=(10,10),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'identity')
    mlp.fit(X_train, y_train)
    r_scores_ANN.append(mlp.score(X_test,y_test))
    
    y_predict = mlp.predict(X_test)

    errors = (np.abs(y_predict-y_test)/y_test)*100
    errors_abs = (np.abs(y_predict-y_test))
    mean_error_ANN.append(np.mean(errors_abs))
    y_test_mat[j] = y_test
    


for j in range(len(r_scores_LR)):
    score = r_scores_LR[j]
    if score <0:
        r_scores_LR[j] = 0
        
for j in range(len(r_scores_ANN)):
    score = r_scores_ANN[j]
    if score <0:
        r_scores_ANN[j] = 0

error_AN = []
error_ADN = []

for item in mean_error_LR:
    error_AN.append(item[0])
    error_ADN.append(item[1])
    

print(np.mean(error_AN)*100)
print(np.mean(error_ADN)*100)

error_AN = []
error_ADN = []

for item in mean_error_ANN:
    error_AN.append(item[0])
    error_ADN.append(item[1])

print(np.mean(error_AN)*100)
print(np.mean(error_ADN)*100)

print(np.mean(r_scores_LR))

np.mean(r_scores_ANN)


# In[29]:


X = df.iloc[:,:-3]
y = df.iloc[:,-2:]

pca = PCA(3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)



mlp = LinearRegression()
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))


# In[31]:


X = pca.transform(X)
y_predict = mlp.predict(X)
fig = plt.figure()
ax = plt.axes()


s= 9

ax.plot(y['AN %']*100,pd.DataFrame(y_predict)[0]*100,'o',
        markerfacecolor = 'red',markeredgecolor = 'red', markersize = s)
ax.plot(y['ADN %']*100,pd.DataFrame(y_predict)[1]*100,'o', 
        markerfacecolor = 'c',markeredgecolor = 'c', markersize = s)

ax.legend(['AN','ADN'],fontsize = 16.5)


ax.plot([0,4],[0,4],'-', color = 'gray')

ax.plot(y['AN %']*100,pd.DataFrame(y_predict)[0]*100,'o',
        markerfacecolor = 'red',markeredgecolor = 'red', markersize = s)
ax.plot(y['ADN %']*100,pd.DataFrame(y_predict)[1]*100,'o', 
        markerfacecolor = 'c',markeredgecolor = 'c', markersize = s)

ax.tick_params(length = 4, labelsize= 14.5, width =2)
ax.set_ylabel(' $C_{predicted}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
ax.set_xlabel('$C_{real}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
#ax.set_title('LR',fontsize=16,fontname = 'Arial')

fig.set_size_inches(6.5, 6.5)

#ax.set_xlim(0,3.5) 
#ax.set_ylim(0,3.5)

ax.text(0.675, 0.06, "$R^{2}=0.977$", transform=ax.transAxes, fontsize=18.5,
        verticalalignment='bottom',bbox =dict(boxstyle='Square',facecolor = 'White', edgecolor = 'gray'))



plt.show()


# In[ ]:


y.to_csv('y_real.csv')


# In[ ]:


y_predict = pd.DataFrame(y_predict, columns = ['AN %','ADN %'])
y_predict.to_csv('y_predict.csv')


# In[ ]:


mlp.score(X_test, y_test)


# In[ ]:




