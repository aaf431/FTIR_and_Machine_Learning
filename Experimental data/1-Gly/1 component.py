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
plt.switch_backend('agg')


# In[2]:


import os

path = "."

filenames= os.listdir (path) # get all files' and folders' names in the current directory


# In[3]:


paths = ['1106syringepump','1109syringepump']


# In[4]:



dic_keys = ['1106','1109']

filenames = os.listdir (paths[0])
fnames = []

for f in filenames:
    if f[-3:]  == 'CSV':
        fnames.append(f)
        


# In[5]:


filenames = os.listdir (paths[1])
fnames = []

for f in filenames:
    if f[-3:]  == 'CSV':
        fnames.append(f)
        


# In[6]:


dic_samples = {}

low_ind = 900
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


    


# In[7]:


df = dic_samples[paths[1]]
df


# In[8]:


labels_1109 = pd.read_csv(paths[1]+'/Labels.csv')

labels_1109.set_index('Sample')


# In[9]:


idx = 0
df.insert(loc=idx, column='F water', value = labels_1109['F water'])
idx = 1
df.insert(loc=idx, column='F glycerol', value = labels_1109['F glycerol'])


df['F water'] = labels_1109['F water'].values
df['F glycerol'] = labels_1109['F glycerol'].values


# In[10]:


df['glycerol %'] = df['F glycerol']*0.1/(df['F glycerol'] + df['F water'])


df.reset_index(inplace = True)

df.drop(['F water', 'F glycerol'],axis = 1, inplace = True)

df.drop('index', axis = 1, inplace = True)


# In[11]:


df.sort_values(by = 'glycerol %', inplace = True)


# In[13]:


df2 = dic_samples[paths[0]]
df2

labels_1107 = pd.read_csv(paths[0]+'/Labels.csv')

labels_1107.set_index('Sample')

idx = 0
#df2.insert(loc=idx, column='F water', value = labels_1109['F water'])
idx = 1
#df2.insert(loc=idx, column='F glycerol', value = labels_1109['F glycerol'])


df2['F water'] = labels_1107['F water'].values
df2['F glycerol'] = labels_1107['F glycerol'].values

df2.head()

df2['glycerol %'] = df2['F glycerol']*0.1/(df2['F glycerol'] + df2['F water'])


df2.reset_index(inplace = True)

df2.drop(['F water', 'F glycerol'],axis = 1, inplace = True)

df2.drop('index', axis = 1, inplace = True)



df2.sort_values(by = 'glycerol %', inplace = True)

df2


# In[14]:


df2.drop(9, inplace = True)


# In[16]:


data = pd.concat([df2,df])
data.reset_index(inplace = True)
data.drop('index', axis = 1, inplace = True)


# In[17]:


data_sorted = data.sort_values(by = 'glycerol %')


# In[18]:


current_palette = sns.color_palette("mako",len(data_sorted))
sns.set_palette(current_palette)
sns.set_style(style='white')

fig = plt.figure()
ax = plt.axes()
j = 0
for j in range(len(data_sorted)):
    ax.plot(data_sorted.columns[:-1],data_sorted.iloc[j,:-1])

ax.tick_params(length = 4, labelsize= 12, width =2)
ax.set_ylabel('Absorbance', fontsize=14, fontname = 'Arial')
ax.set_xlabel('wavenumber [cm$^{-1}$]', fontsize=14, fontname = 'Arial')

ax.invert_xaxis()
fig.set_size_inches(10, 5)


plt.legend(data_sorted['glycerol %'],fontsize = 14, loc = 'right', bbox_to_anchor=(1.7, 0.5))

#ax.set_xlim(3000,900) 
#ax.set_ylim(0,0.25)

ax.set_xlim(1600,900) 
ax.set_ylim(0,0.25)

plt.show()


# In[19]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[23]:


#Principal component analysis

plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

color1 = 'darkblue'
color2 = 'darkorange'
pca.explained_variance_
fig = plt.figure()
ax = plt.axes()



n_pca = 10
n = np.arange(1,n_pca+1)
plt.bar(n,pca.explained_variance_ratio_[0:n_pca]*100,
            color = color2,linewidth = 1, edgecolor = 'black')
ax.plot(n,np.cumsum(pca.explained_variance_ratio_[0:n_pca]*100),marker = 'o',color = color1)

ax.set_title('PCA 1-component', fontsize=20, fontname = 'Arial')
for i,j in zip(n,np.cumsum(pca.explained_variance_ratio_[0:n_pca]*100)):
    ax.annotate(str(np.round(j,1))+ '%', xy=(i-0.2, j+4), fontsize = 14)
ax.set_xlabel('Number of Principal Components', fontsize=16, fontname = 'Arial')
ax.set_ylabel('Explained Variance [%]', fontsize=16, fontname = 'Arial')
ax.tick_params(length = 4, labelsize= 14, width =2)
ax.set_ylim([0,120])
plt.legend(['Cummulative','Per Component'], fontsize = 14, loc = 'lower right')
plt.xticks(np.arange(1,n_pca+1))

plt.yticks(np.arange(0,120,20))
fig.set_size_inches(10, 5)


# In[30]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]

pca = PCA(2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

mlp = LinearRegression()
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))

mean_err = np.mean(np.abs(y_test, mlp.predict(X_test)))*100

print(mean_err)

X = pca.transform(X)


y_predict = mlp.predict(X)


# In[32]:


y_predict = pd.DataFrame(y_predict)
y_real = pd.DataFrame(y)


y_real.to_csv('y_real.csv')

y_predict.to_csv('y_predict.csv')


# In[33]:


fig = plt.figure()
ax = plt.axes()

s= 9

colors2 = ['darkorange','darkturquoise']


ax.plot(y*100,y_predict*100,'o',
        markerfacecolor = colors2[0], markeredgecolor = colors2[0], markersize = s)

ax.legend(['Glycerol'],fontsize = 16.5)


ax.plot([0,10],[0,10],'-', color = 'gray')

ax.plot(y*100,y_predict*100,'o',
        markerfacecolor = colors2[0],markeredgecolor = colors2[0], markersize = s)


ax.tick_params(length = 4, labelsize= 14.5, width =2)
ax.set_ylabel(' $C_{predicted}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
ax.set_xlabel('$C_{real}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
#ax.set_title('LR',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(6.5, 6.5)

ax.set_xlim(0,10) 
ax.set_ylim(0,10)

ax.text(0.65, 0.06, "$R^{2}$ = " + str(np.round(mlp.score(X_test, y_test),3)) + ' %', transform=ax.transAxes, fontsize=18.5,
        verticalalignment='bottom',bbox =dict(boxstyle='Square',facecolor = 'White', edgecolor = 'gray'))


plt.show()


# In[76]:


from sklearn.metrics import mean_squared_error

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []

MAE_LR = []
MAE_ANN = []

y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    mlp =MLPRegressor(hidden_layer_sizes=(20,20),tol=1e-5, max_iter=500, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'relu')
    mlp.fit(X_train, y_train)
    r_scores_ANN.append(mlp.score(X_test,y_test))
    
    y_predict = mlp.predict(X_test)

    MAE = np.sqrt(mean_squared_error(y_test, y_predict))*100
    errors_abs = (np.abs(y_predict-y_test))*100
    
    mean_error_ANN.append(np.mean(errors_abs))
    MAE_ANN.append(MAE)
        
    pca = PCA(2)
    X_test = pca.fit_transform(X_test)
    X_train = pca.transform(X_train)
    

    LR = LinearRegression()
    LR.fit(X_train, y_train)
    r_scores_LR.append(LR.score(X_test, y_test))
    y_predict = LR.predict(X_test)

    MAE = np.sqrt(mean_squared_error(y_test, y_predict))*100
    errors_abs = (np.abs(y_predict-y_test))*100
    
    mean_error_LR.append(np.mean(errors_abs))
    MAE_LR.append(MAE)
    
    
    


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

print(np.mean(mean_error_LR))
print(np.mean(mean_error_ANN))

print(np.std(mean_error_LR))
print(np.std(mean_error_ANN))


# In[35]:


print(np.std(r_scores_LR ))
print(np.std(r_scores_ANN))


# In[36]:


print(np.mean(MAE_LR))
print(np.mean(MAE_ANN))

print(np.std(MAE_LR))
print(np.std(MAE_ANN))


# In[78]:


from sklearn.metrics import mean_squared_error

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
n_samples = 200

r_scores_LR = []
r_scores_ANN = []
mean_error_LR = []
mean_error_ANN = []

MAE_LR = []
MAE_ANN = []

y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
for j in range(n_samples):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pca = PCA(2)
    X_test = pca.fit_transform(X_test)
    X_train = pca.transform(X_train)
    mlp =MLPRegressor(hidden_layer_sizes=(2,2),tol=1e-5, max_iter=500, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'relu')
    mlp.fit(X_train, y_train)
    r_scores_ANN.append(mlp.score(X_test,y_test))
    
    y_predict = mlp.predict(X_test)

    MAE = np.sqrt(mean_squared_error(y_test, y_predict))*100
    errors_abs = (np.abs(y_predict-y_test))*100
    
    mean_error_ANN.append(np.mean(errors_abs))
    MAE_ANN.append(MAE)
        
    
    

    LR = LinearRegression()
    LR.fit(X_train, y_train)
    r_scores_LR.append(LR.score(X_test, y_test))
    y_predict = LR.predict(X_test)

    MAE = np.sqrt(mean_squared_error(y_test, y_predict))*100
    errors_abs = (np.abs(y_predict-y_test))*100
    
    mean_error_LR.append(np.mean(errors_abs))
    MAE_LR.append(MAE)
    
    
    


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

print(np.mean(mean_error_LR))
print(np.mean(mean_error_ANN))

print(np.std(mean_error_LR))
print(np.std(mean_error_ANN))


# In[ ]:


print(np.std(r_scores_LR ))
print(np.std(r_scores_ANN))

print(np.mean(MAE_LR))
print(np.mean(MAE_ANN))

print(np.std(MAE_LR))
print(np.std(MAE_ANN))

