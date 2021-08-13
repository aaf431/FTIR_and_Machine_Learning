#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Start from wn = 2900, some noisy data


# In[2]:


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


# In[3]:


df1 = pd.read_csv('data_0208_0211.csv', index_col = 0)
df1.head()


# In[4]:


df2 = pd.read_csv('data_0215_0216.csv', index_col = 0)
df2.head()


# In[5]:


df2.reset_index(inplace = True)
df1.reset_index(inplace = True)


# In[6]:


df = pd.concat([df1,df2])
df.drop(['index'],axis = 1, inplace = True)


# In[7]:


df.reset_index(inplace = True)
df.head()


# In[8]:


df.drop(['index'],axis = 1, inplace = True)


# In[9]:


len(df)
df


# In[10]:


X = df.iloc[:,:-3]
y = df.iloc[:,-3:]
len(X)


# In[11]:


df.sort_values(by = 'glycerol %', inplace = True)


# In[12]:


current_palette = sns.color_palette("mako", len(df))
sns.set_palette(current_palette)
#sns.set_style(style='white')

fig = plt.figure()
ax = plt.axes()
j = 0
for j in range(len(df)):
    ax.plot(df.columns[:-3],df.iloc[j,:-3])

ax.tick_params(length = 4, labelsize= 12, width =2)
ax.set_ylabel('Absorbance', fontsize=14, fontname = 'Arial')
ax.set_xlabel('wavenumber [cm$^{-1}$]', fontsize=14, fontname = 'Arial')


#ax.set_title('IPA in water',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(10, 4)

#plt.legend(np.round(df['glycerol %']*0.1,3),fontsize = 14, loc = 'right', bbox_to_anchor=(1.7, 0.5))

#plt.xlim(2700,1000) 
#ax.set_ylim(-0.25,0.25)
my_x_ticks = np.arange(2750, 900, 250)
plt.xticks(my_x_ticks)

plt.show()


# In[13]:


pca = PCA(40)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[15]:


X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)


# In[16]:


pca.explained_variance_ratio_

color1 = 'darkblue'
color2 = 'darkorange'
pca.explained_variance_
fig = plt.figure()
ax = plt.axes()


n_pca = len(pca.explained_variance_ratio_)
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



pca = PCA(15)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.score(X_test, y_test))

y_predict = LR.predict(X_test)

errors = (np.abs(y_predict-y_test)/y_test)*100
errors_abs = (np.abs(y_predict-y_test))*100
print(np.mean(errors_abs))


# In[18]:


np.mean(errors)


# In[54]:


X = df.iloc[:,:-3]
y = df.iloc[:,-3:]
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


# In[55]:




for j in range(len(r_scores_LR)):
    score = r_scores_LR[j]
    if score <0:
        r_scores_LR[j] = 0
        
for j in range(len(r_scores_ANN)):
    score = r_scores_ANN[j]
    if score <0:
        r_scores_ANN[j] = 0


# In[56]:


print(np.mean(r_scores_LR ))
print(np.mean(r_scores_ANN))


# In[57]:


error_gly = []
error_IPA = []
error_but = []
for item in mean_error_LR:
    error_gly.append(item[0])
    error_IPA.append(item[1])
    error_but.append(item[2])

print(np.mean(error_gly))
print(np.mean(error_IPA))
print(np.mean(error_but))


# In[23]:


error_gly = []
error_IPA = []
error_but = []
for item in mean_error_ANN:
    error_gly.append(item[0])
    error_IPA.append(item[1])
    error_but.append(item[2])

print(np.mean(error_gly))
print(np.mean(error_IPA))
print(np.mean(error_but))


# In[24]:


X = df.iloc[:,:-3]
y = df.iloc[:,-3:]

pca = PCA(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

mlp = LinearRegression()
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))


# In[25]:


X = pca.transform(X)


# In[26]:


y_predict = mlp.predict(X)
fig = plt.figure()
ax = plt.axes()


s = 7

ax.plot(y['glycerol %']*100,pd.DataFrame(y_predict)[0]*100,'o',
        markerfacecolor = 'orangered',markeredgecolor = 'orangered', markersize = s)
ax.plot(y['IPA %']*100,pd.DataFrame(y_predict)[1]*100,'o',
        markerfacecolor = 'slateblue',markeredgecolor = 'slateblue', markersize = s)
ax.plot(y['1-butanol %']*100,pd.DataFrame(y_predict)[2]*100,'o', 
        markerfacecolor = 'lightgreen',markeredgecolor = 'lightgreen', markersize = s)

ax.legend(['Glycerol','IPA','1-butanol'],fontsize = 16.5)


ax.plot([0,17.5],[0,17.5],'-', color = 'gray')

ax.tick_params(length = 4, labelsize= 14.5, width =2)
ax.set_ylabel(' $C_{predicted}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
ax.set_xlabel('$C_{real}$ [% m/m]', fontsize=18.5, fontname = 'Arial')
#ax.set_title('LR',fontsize=16,fontname = 'Arial')
ax.invert_xaxis()
fig.set_size_inches(6.5, 6.5)

ax.set_xlim(0,10) 
ax.set_ylim(0,10)

ax.text(0.675, 0.06, "$R^{2}= $" + str(0.985), transform=ax.transAxes, fontsize=18.5,
        verticalalignment='bottom',bbox =dict(boxstyle='Square',facecolor = 'White', edgecolor = 'gray'))



plt.show()


# In[28]:


#Data size dependance

data_size = np.append(np.arange(10,100,5),[98])

errors_n_samples_ANN = []
scores_n_samples_ANN = []

errors_std_ANN  = []
scores_std_ANN = []
n_samples = 200

for n in data_size:

    #ind_rand = np.random.randint(0,42,n)
    

    
   

    r_scores_LR = []
    r_scores_ANN = []
    mean_error_LR = []
    mean_error_ANN = []


    #y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
    for j in range(n_samples):
        #ind_rand = np.random.randint(0,109,n)
        ind_rand = np.random.choice(98,n,replace = False)
        X = df.iloc[ind_rand,:-3]
        y = df.iloc[ind_rand,-3:]
        pca = PCA(5)
        X = pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        

        mlp =MLPRegressor(hidden_layer_sizes=(4,2),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                         learning_rate = 'adaptive',batch_size = 10, activation = 'identity')
        mlp.fit(X_train, y_train)
                          
        r_scores_ANN.append(mlp.score(X_test,y_test))

        y_predict = mlp.predict(X_test)

        errors = (np.abs(y_predict-y_test)/y_test)*100
        errors_abs = (np.abs(y_predict-y_test))
        mean_error_ANN.append(np.mean(errors_abs))
        #y_test_mat[j] = y_test
        for j in range(len(r_scores_ANN)):
            score = r_scores_ANN[j]
            if score <0:
                r_scores_ANN[j] = 0

    errors_n_samples_ANN.append(np.mean(pd.DataFrame(mean_error_ANN))*100)
    
    errors_std_ANN.append(np.std(mean_error_ANN)*100)
    
    scores_n_samples_ANN.append(np.mean(r_scores_ANN))
    
    scores_std_ANN.append(np.std(r_scores_ANN))


# In[29]:


scores_n_samples_ANN


# In[30]:


data_size = np.append(np.arange(10,100,5),[98])

errors_n_samples = []
scores_n_samples = []

errors_std  = []
scores_std = []
n_samples = 200

for n in data_size:

    #ind_rand = np.random.randint(0,42,n)

    
   

    r_scores_LR = []

    mean_error_LR = []
 


    #y_test_mat = pd.DataFrame(np.zeros((len(y),n_samples)),index = X.index)
    for j in range(n_samples):
        #ind_rand = np.random.randint(0,109,n)
        
        ind_rand = np.random.choice(98,n,replace = False)
        X = df.iloc[ind_rand,:-3]
        y = df.iloc[ind_rand,-3:]
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


        #mlp =MLPRegressor(hidden_layer_sizes=(5,),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                         #learning_rate = 'adaptive',batch_size = 10, activation = 'identity')
        #mlp.fit(X_train, y_train)
        #r_scores_ANN.append(mlp.score(X_test,y_test))

        #y_predict = mlp.predict(X_test)

        #errors = (np.abs(y_predict-y_test)/y_test)*100
        #errors_abs = (np.abs(y_predict-y_test))
        #mean_error_ANN.append(np.mean(errors_abs))
        #y_test_mat[j] = y_test
        for j in range(len(r_scores_LR)):
            score = r_scores_LR[j]
            if score <0:
                r_scores_LR[j] = 0

    errors_n_samples.append(np.mean(pd.DataFrame(mean_error_LR))*100)
    
    errors_std.append(np.std(mean_error_LR)*100)
    
    scores_n_samples.append(np.mean(r_scores_LR))
    
    scores_std.append(np.std(r_scores_LR))


# In[32]:


error_gly = []
error_IPA = []
error_but = []
for item in errors_n_samples_ANN:
    error_gly.append(item[0])
    error_IPA.append(item[1])
    error_but.append(item[2])

print(np.mean(error_gly))
print(np.mean(error_IPA))
print(np.mean(error_but))


# In[35]:


fig = plt.figure()
ax = plt.axes()

ax.plot(np.round(data_size*0.8),scores_n_samples,'-o', color = 'darkblue')
ax.plot(np.round(data_size*0.8),scores_n_samples_ANN,'-o', color = 'cornflowerblue')
ax.tick_params(length = 4, labelsize= 12, width =2)
ax.set_xlabel('X train size', fontsize=14, fontname = 'Arial')
ax.set_ylabel('$R^{2}$', fontsize=14, fontname = 'Arial')
#ax.set_ylim([0,1])
ax.legend(['LR','ANN'], fontsize = 14, loc = 'lower right')
plt.show()


# In[36]:


fig = plt.figure()
ax = plt.axes()

ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples)['glycerol %'],'-o', color = 'darkblue')
ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples_ANN)['glycerol %'],'-o', color = 'cornflowerblue')
ax.tick_params(length = 4, labelsize= 10, width =2)
ax.set_xlabel('X train size', fontsize=14, fontname = 'Arial')
ax.set_ylabel('Mean error for glycerol [% mas abs]', fontsize=14, fontname = 'Arial')
ax.legend(['LR','ANN'], fontsize = 14, loc = 'upper right')
#ax.set_ylim([0,1])
plt.show


# In[37]:


fig = plt.figure()
ax = plt.axes()

ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples)['glycerol %'],'-o', color = 'darkblue')
ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples_ANN)['glycerol %'],'-o', color = 'cornflowerblue')
ax.tick_params(length = 4, labelsize= 12, width =2)
ax.set_xlabel('X train size', fontsize=14, fontname = 'Arial')
ax.set_ylabel('Mean error [% mas abs]', fontsize=14, fontname = 'Arial')
ax.legend(['LR','ANN'], fontsize = 14, loc = 'upper right')
#ax.set_ylim([0,1])
plt.show()


# In[38]:


fig = plt.figure()
ax = plt.axes()

ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples)['IPA %'],'-o', color = 'darkblue')
ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples_ANN)['IPA %'],'-o', color = 'cornflowerblue')
ax.tick_params(length = 4, labelsize= 12, width =2)
ax.set_xlabel('X train size', fontsize=14, fontname = 'Arial')
ax.set_ylabel('Mean error for IPA [% mas abs]', fontsize=14, fontname = 'Arial')
ax.legend(['LR','ANN'], fontsize = 14, loc = 'upper right')
#ax.set_ylim([0,1])
plt.show()


# In[39]:


fig = plt.figure()
ax = plt.axes()

ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples)['1-butanol %'],'-o', color = 'darkblue')
ax.plot(np.round(data_size*0.8),pd.DataFrame(errors_n_samples_ANN)['1-butanol %'],'-o', color = 'cornflowerblue')
ax.tick_params(length = 4, labelsize= 12, width =2)
ax.set_xlabel('X train size', fontsize=14, fontname = 'Arial')
ax.set_ylabel('Mean error for 1-butanol [% mas abs]', fontsize=14, fontname = 'Arial')
ax.legend(['LR','ANN'], fontsize = 14, loc = 'upper right')
#ax.set_ylim([0,1])
plt.show()


# In[40]:


#Training with one set, testing with the other

X_train = df1.iloc[:,:-3]
X_test = df2.iloc[:,:-3]
y_train = df1.iloc[:,-3:]
y_test = df2.iloc[:,-3:]
X_train.drop(['index'],axis = 1, inplace = True)
X_test.drop(['index'], axis = 1, inplace = True)


# In[41]:



pca = PCA(4, random_state = 2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

LR = LinearRegression()
LR.fit(X_train_pca, y_train)
print(LR.score(X_test_pca, y_test))

y_predict = LR.predict(X_test_pca)

errors = (np.abs(y_predict-y_test)/y_test)*100
errors_abs = (np.abs(y_predict-y_test))*100
print(np.mean(errors_abs))


# In[42]:


pca = PCA(10, random_state = 2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

mlp =MLPRegressor(hidden_layer_sizes=(5,4),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'identity')
mlp.fit(X_train_pca, y_train)
print(mlp.score(X_test_pca, y_test))

y_predict = mlp.predict(X_test_pca)

errors = (np.abs(y_predict-y_test)/y_test)*100
errors_abs = (np.abs(y_predict-y_test))*100
print(np.mean(errors_abs))


# In[43]:


X = df.iloc[:,:-3]
y = df.iloc[:,-3:]

pca = PCA(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

mlp = LinearRegression()
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))


# In[44]:


X = pca.transform(X)


# In[ ]:





# In[45]:


X_train = df1.iloc[:,:-3]
X_test = df2.iloc[:,:-3]
y_train = df1.iloc[:,-3:]
y_test = df2.iloc[:,-3:]
X_train.drop(['index'],axis = 1, inplace = True)
X_test.drop(['index'], axis = 1, inplace = True)


# In[46]:



pca = PCA(4, random_state = 2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

LR = LinearRegression()
LR.fit(X_train_pca, y_train)
print(LR.score(X_test_pca, y_test))

y_predict = LR.predict(X_test_pca)

errors = (np.abs(y_predict-y_test)/y_test)*100
errors_abs = (np.abs(y_predict-y_test))*100
print(np.mean(errors_abs))


# In[ ]:




