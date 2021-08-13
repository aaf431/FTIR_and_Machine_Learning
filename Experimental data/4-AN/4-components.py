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
paths = ['20210614 (1)','20210614 (2)','20210614 (3)']
filenames = os.listdir (paths[0])

fnames = []

for f in filenames:
    if f[-3:]  == 'CSV':
        fnames.append(f)

dic_samples = {}

low_ind = 1000
high_ind = 3000

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
    dic_samples[path] = df


labels1 = pd.read_csv(paths[0]+'/Labels.csv')

labels1.set_index('Sample', inplace = True)

labels2 = pd.read_csv(paths[1]+'/Labels.csv')

labels2.set_index('Sample', inplace = True)

labels3 = pd.read_csv(paths[2]+'/Labels.csv')

labels3.set_index('Sample', inplace = True)


# In[2]:


df = pd.concat([dic_samples[paths[0]],dic_samples[paths[1]]])


# In[3]:


df.sort_index(inplace = True)
labels = pd.concat([labels1,labels2])
labels.sort_index(inplace = True)


# In[4]:


df.index == labels.index


# In[5]:



idx = 0
df.insert(loc=idx, column='F AN', value = labels['F AN'])
idx = 1
df.insert(loc=idx, column='F ADN', value = labels['F ADN'])
idx = 2
df.insert(loc=idx, column='F PN', value = labels['F PN'])
idx = 3
df.insert(loc=idx, column='F glycerol', value = labels['F glycerol'])

df['F AN'] = labels['F AN'].values
df['F ADN'] = labels['F ADN'].values
df['F PN'] = labels['F PN'].values
df['F glycerol'] = labels['F glycerol'].values
df.head()


# In[6]:


C_AN = 4.1/100
C_ADN = 4.2/100
C_PN = 3.8/100
C_glycerol = 5/100

df['AN %'] = df['F AN']*C_AN/(df['F AN'] + df['F ADN']+df['F PN']+df['F glycerol'])
df['ADN %'] = df['F ADN']*C_ADN/(df['F AN']  + df['F ADN']+ df['F PN']+df['F glycerol'])
df['PN %'] = df['F PN']*C_PN/(df['F AN']  + df['F ADN'] + df['F PN']+df['F glycerol'])
df['glycerol %'] = df['F glycerol']*C_glycerol/(df['F AN']  + df['F ADN'] + df['F PN']+df['F glycerol'])


df.reset_index(inplace = True)

df.drop([ 'F AN', 'F ADN', 'F PN', 'F glycerol'],axis = 1, inplace = True)

df.drop('index', axis = 1, inplace = True)

df.head()


# In[7]:


df.sort_values(by = 'glycerol %', inplace = True)


# In[8]:


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


#plt.legend(np.round(df['glycerol %']*0.1,3),fontsize = 14, loc = 'right', bbox_to_anchor=(1.7, 0.5))

#ax.set_xlim(3000,900) 
#ax.set_ylim(0,0.25)

#ax.set_xlim(1600,900) 
#ax.set_ylim(0,0.25)

plt.show()


# In[9]:


X = df.iloc[:,:-4]
y = df.iloc[:,-4]

pca = PCA()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

pca.explained_variance_ratio_


color1 = 'darkblue'
color2 = 'darkorange'
pca.explained_variance_
fig = plt.figure()
ax = plt.axes()


n_pca = 11
n = np.arange(1,n_pca+1)
plt.bar(n,pca.explained_variance_ratio_[0:n_pca]*100,color = color2,linewidth = 1, edgecolor = 'black')

ax.plot(n,np.cumsum(pca.explained_variance_ratio_[0:n_pca]*100),marker = 'o',color = color1)

ax.set_title('Principal Component Analysis', fontsize=20, fontname = 'Arial')

ax.set_xlabel('Number of Principal Components', fontsize=16, fontname = 'Arial')
ax.set_ylabel('Explained Variance [%]', fontsize=16, fontname = 'Arial')
ax.tick_params(length = 4, labelsize= 14, width =2)
#ax.set_ylim([0.8,1.01])
plt.legend(['Cummulative','Per Component'], fontsize = 14)#, loc = 'upper right')
plt.xticks(np.arange(1,n_pca+1
                    ))
fig.set_size_inches(10, 5)


# In[10]:


pca = pd.DataFrame(pca.explained_variance_ratio_[:10])

pca.to_csv('pca.csv')


# In[11]:


X = df.iloc[:,:-4]
y = df.iloc[:,-4:]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


for j in range(n_samples):
    pca = PCA(7)

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
    
    
    mlp =MLPRegressor(hidden_layer_sizes=(10,10),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'identity')
    mlp.fit(X_train, y_train)
    r_scores_ANN.append(mlp.score(X_test,y_test))
    
    y_predict = mlp.predict(X_test)

    errors = (np.abs(y_predict-y_test)/y_test)*100
    
    errors_abs = np.sqrt(mean_squared_error(y_test, y_predict, multioutput = 'raw_values'))
    mean_error_ANN.append(errors_abs)
    #errors_abs = (np.abs(y_predict-y_test))
    #mean_error_ANN.append(np.mean(errors_abs))
    


for j in range(len(r_scores_LR)):
    score = r_scores_LR[j]
    if score <0:
        r_scores_LR[j] = 0
        
for j in range(len(r_scores_ANN)):
    score = r_scores_ANN[j]
    if score <0:
        r_scores_ANN[j] = 0


# In[12]:


print(np.mean(r_scores_LR ))
print(np.mean(r_scores_ANN))


# In[13]:


print(np.std(r_scores_LR ))
print(np.std(r_scores_ANN))


# In[12]:


print(np.mean(mean_error_LR))
print(np.mean(mean_error_ANN))


# In[13]:


print(np.std(pd.DataFrame(mean_error_LR))*100)
print(np.std(pd.DataFrame(mean_error_ANN))*100)


# In[140]:


print(np.mean(pd.DataFrame(mean_error_LR)))
print(np.mean(pd.DataFrame(mean_error_ANN)))


# In[131]:


mean_squared_error(y_test, y_predict, multioutput = 'raw_values')


# In[134]:


mean_squared_error(y_test, y_predict, multioutput = 'raw_values')


# In[113]:


X = df.iloc[:,:-4]
y = df.iloc[:,-4:]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


for j in range(n_samples):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    


for j in range(len(r_scores_LR)):
    score = r_scores_LR[j]
    if score <0:
        r_scores_LR[j] = 0
        
for j in range(len(r_scores_ANN)):
    score = r_scores_ANN[j]
    if score <0:
        r_scores_ANN[j] = 0

print(np.mean(r_scores_LR ))
print(np.mean(r_scores_ANN))


# In[16]:


X = df.iloc[:,:-4]
y = df.iloc[:,-4:]

pca = PCA(7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

mlp = LinearRegression()
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))

X = pca.transform(X)


y_predict = mlp.predict(X)


# In[17]:


fig = plt.figure()
ax = plt.axes()


s= 7

ax.plot(y['AN %']*100,pd.DataFrame(y_predict)[0]*100,'o',
        markerfacecolor = 'red',markeredgecolor = 'red', markersize = s)

ax.plot([0,10],[0,10],'-', color = 'gray')

ax.plot(y['AN %']*100,pd.DataFrame(y_predict)[0]*100,'o',
        markerfacecolor = 'red',markeredgecolor = 'red', markersize = s)

ax.tick_params(length = 4, labelsize= 14.5, width =2)
ax.set_ylabel(' $C_{predicted}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
ax.set_xlabel('$C_{real}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
#ax.set_title('LR',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(6.5, 6.5)

ax.set_xlim(0,3.5) 
ax.set_ylim(0,3.5)

ax.text(0.675, 0.06, "AN", transform=ax.transAxes, fontsize=18.5,
        verticalalignment='bottom',bbox =dict(boxstyle='Square',facecolor = 'White', edgecolor = 'gray'))



plt.show()

fig = plt.figure()
ax = plt.axes()


s= 7

ax.plot(y['ADN %']*100,pd.DataFrame(y_predict)[1]*100,'o',
        markerfacecolor = 'c',markeredgecolor = 'c', markersize = s)

ax.plot([0,10],[0,10],'-', color = 'gray')

ax.plot(y['ADN %']*100,pd.DataFrame(y_predict)[1]*100,'o',
        markerfacecolor = 'c',markeredgecolor = 'c', markersize = s)

ax.tick_params(length = 4, labelsize= 14.5, width =2)
ax.set_ylabel(' $C_{predicted}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
ax.set_xlabel('$C_{real}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
#ax.set_title('LR',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(6.5, 6.5)

ax.set_xlim(0,3.5) 
ax.set_ylim(0,3.5)

ax.text(0.675, 0.06, "ADN", transform=ax.transAxes, fontsize=18.5,
        verticalalignment='bottom',bbox =dict(boxstyle='Square',facecolor = 'White', edgecolor = 'gray'))



plt.show()

fig = plt.figure()
ax = plt.axes()


s= 7

ax.plot(y['PN %']*100,pd.DataFrame(y_predict)[2]*100,'o',
        markerfacecolor = 'darkblue',markeredgecolor = 'darkblue', markersize = s)

ax.plot([0,10],[0,10],'-', color = 'gray')

ax.plot(y['PN %']*100,pd.DataFrame(y_predict)[2]*100,'o',
        markerfacecolor = 'darkblue',markeredgecolor = 'darkblue', markersize = s)

ax.tick_params(length = 4, labelsize= 14.5, width =2)
ax.set_ylabel(' $C_{predicted}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
ax.set_xlabel('$C_{real}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
#ax.set_title('LR',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(6.5, 6.5)

ax.set_xlim(0,3.5) 
ax.set_ylim(0,3.5)

ax.text(0.675, 0.06, "PN", transform=ax.transAxes, fontsize=18.5,
        verticalalignment='bottom',bbox =dict(boxstyle='Square',facecolor = 'White', edgecolor = 'gray'))



plt.show()

fig = plt.figure()
ax = plt.axes()


s= 7

ax.plot(y['glycerol %']*100,pd.DataFrame(y_predict)[3]*100,'o',
        markerfacecolor = 'orange',markeredgecolor = 'orange', markersize = s)

ax.plot([0,10],[0,10],'-', color = 'gray')

ax.plot(y['glycerol %']*100,pd.DataFrame(y_predict)[3]*100,'o',
        markerfacecolor = 'orange',markeredgecolor = 'orange', markersize = s)

ax.tick_params(length = 4, labelsize= 14.5, width =2)
ax.set_ylabel(' $C_{predicted}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
ax.set_xlabel('$C_{real}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
#ax.set_title('LR',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(6.5, 6.5)

ax.set_xlim(0,3.5) 
ax.set_ylim(0,3.5)

ax.text(0.675, 0.06, "glycerol", transform=ax.transAxes, fontsize=18.5,
        verticalalignment='bottom',bbox =dict(boxstyle='Square',facecolor = 'White', edgecolor = 'gray'))



plt.show()


# In[19]:


y_predict = pd.DataFrame(y_predict, columns = ['AN %','ADN %','PN %','glycerol %'])
y_predict.to_csv('y_predict.csv')

y.to_csv('y_real.csv')


# In[53]:


#For AN

X = df.iloc[:,:-4]
y = df.iloc[:,-4]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    pca = PCA(5)

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

print(np.mean(r_scores_LR ))
print(np.mean(r_scores_ANN))


# In[31]:


#For ADN

X = df.iloc[:,:-4]
y = df.iloc[:,-3]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    pca = PCA(5)

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

print(np.mean(r_scores_LR ))
print(np.mean(r_scores_ANN))


# In[32]:


#For PN

X = df.iloc[:,:-4]
y = df.iloc[:,-2]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    pca = PCA(5)

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

print(np.mean(r_scores_LR ))
print(np.mean(r_scores_ANN))


# In[33]:


#For glycerol

X = df.iloc[:,:-4]
y = df.iloc[:,-4]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []


y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    pca = PCA(5)

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

print(np.mean(r_scores_LR ))
print(np.mean(r_scores_ANN))

