#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#import tables
import seaborn as sns
plt.switch_backend('agg')


# In[2]:


#Quering filenames 
import os
from os import listdir
from os.path import isfile, join

mypath = "./Molecules/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
fnames_all = []

for f in onlyfiles:
    if f[-3:] == 'csv':
        fnames_all.append(f)
        
        
print(fnames_all)
fnames = fnames_all


# In[3]:


keys = ['AN','ADN','PN','Water']


# In[4]:


#Creating dictionary and storing pure component spectra
df_collection = {}
path = mypath

n_components = len(keys)
for j in range(len(fnames)):
    
    fname = fnames[j]
    key = keys[j]
    
    file = pd.read_csv(path + fname)
    if len(file) == 1:
        file = pd.read_csv(path + fname,header = None)
    first_wn = file.iloc[0,0]
    last_wn = file.iloc[0,-1]
    if last_wn <first_wn:
        file = file[file.columns[::-1]]
        file.columns = range(file.shape[1])
    df_collection[key] = file
    


# In[5]:


# Obtaining wn ranges for each spectrum
wn_ranges = pd.DataFrame(columns = ['min','max','max-min'], index = keys)

for key in keys:
    frame = df_collection[key]
    max_wn = frame.iloc[0,-1]
    min_wn = frame.iloc[0,0]
    wn_ranges.loc[key,:] = [min_wn,max_wn,max_wn-min_wn]
    
    


# In[6]:


# Selecting component with minimum range
idx_min = np.argmin(wn_ranges['max-min'].values)
key_range = keys[idx_min]


# In[7]:


# Aligning spectra to the same wn vector
frame_range = df_collection[key_range]
wn = frame_range.iloc[0,:].values

new_collection = {}
for key in keys:
    frame = df_collection[key]
    if key != key_range:
        
              
        frame_ = np.interp(frame_range.iloc[0,:].values,frame.iloc[0,:].values,frame.iloc[1,:].values)
        
    else:
        frame_ = frame.iloc[1,:].values
        
    frame_new = pd.DataFrame([np.round(wn),frame_])

    new_collection[key] = frame_new


# In[8]:



wn = frame_range.iloc[0,:].values
max_water = np.max(new_collection['Water'].iloc[1,:])


# In[9]:


n = 400; #number of random combinations to generate

n_noise = 5 #How many noise levels we will sample

nf = np.linspace(0,1,n_noise)


# In[10]:


#function to add noise to each spectral point, according to a weight, normalized by max absorbance which is usually 1

def assign_noise(spectra_point, max_p, weight):
    n = np.random.uniform(-0.05,0.05)*max_p*weight;
    n = spectra_point + n;
    return n

X_keys = []

for key in keys:
    if key != 'Water':
        X_keys.append('X_'+key)


# In[11]:


#Creating empy DataFrame to store test spectral data

col_labels = np.round(wn).astype('str')
col_labels = np.concatenate((X_keys,col_labels),axis = 0)
data_test = pd.DataFrame(np.zeros((n*n_noise,len(col_labels))),columns = col_labels)


# In[12]:


#Creating a DataFrame with the pure component spectra

pure_components = pd.DataFrame()
for key in new_collection.keys():
    frame = new_collection[key]
    pure_components = pd.concat([pure_components,frame.iloc[1,:]],axis = 1)
pure_components.columns = new_collection.keys()


# In[13]:


#Composition 1-D array for the test data

Comp = np.random.uniform(0,0.1,[n,len(keys)-1])
Comp = pd.DataFrame(Comp,columns = col_labels[0:len(keys)-1])


# In[14]:


#Performing a linear combination of pure component spectra and random compositons to create the test data
# test DataFrame lenght is n x len(NF)

for x in range(len(Comp)):
    vec = Comp.iloc[x,:].values
    vec = np.append(vec,1-sum(vec))
    spectra = np.sum(vec*pure_components,axis =1)
    data_test.iloc[x,len(keys)-1:] = np.asarray(spectra)

data_test.insert(0,'NF', np.zeros((len(data_test),1)))


# In[15]:



#Adding the compositions as a columns in the test DataFrame

for j in range(n_noise):
    for k in range(n_components - 1):
        data_test.iloc[n*j:n*(j+1),k+1] =np.asarray(Comp.iloc[:,k])
    

LC_main_400_rand = data_test.iloc[0:n,:]


# In[16]:



#Introducing noise to each data_test segment accorfing to nf

for segment in range(n_noise):
    for x in range(segment*n,(segment+1)*n):
        spectra = LC_main_400_rand.iloc[x-segment*n,n_components:]
        #max_peak = np.max(spectra)
        max_peak = np.max(spectra[1000:])
        spectrum_noise = [assign_noise(point,max_peak,nf[segment]) for point in spectra]
        data_test.iloc[x,n_components:] = spectrum_noise


# In[17]:


data_test.loc[:,'NF'] = np.repeat(nf,400)
data_test.head()


# In[18]:


# Creating a composition vector according to a sobol sequence


import sobol_seq

Comp = sobol_seq.i4_sobol_generate(len(keys)-1,n)*0.1
Comp = pd.DataFrame(Comp,columns = col_labels[0:len(keys)-1])

data_train = pd.DataFrame(np.zeros((n,len(col_labels))),columns = col_labels)


# In[19]:


#Performing a linear combination of pure component spectra and random compositons to create the train data

for x in range(len(Comp)):
    vec = Comp.iloc[x,:].values
    vec = np.append(vec,1-sum(vec))
    spectra = np.sum(vec*pure_components,axis =1)
    data_train.iloc[x,len(keys)-1:] = np.asarray(spectra)

data_train.insert(0,'NF', np.zeros((len(data_train),1)))

for k in range(n_components - 1):
    data_train.iloc[:,k+1] =np.asarray(Comp.iloc[:,k])


# In[20]:


data_train.head()


# In[21]:


from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression


# In[22]:



data_art2 = data_test


# In[23]:


#Defining target variables to predict


target_variables = []
for label in data_train.columns.values:
    if label[0] == 'X':
        target_variables.append(label)
        
    


# In[24]:


#Function to test models with data with different noise levels


def test_model_noise(model,nf,pca_param,pca,range_opt):
    
    data2_0 = data_art2.iloc[np.where(data_art2.NF == nf)[0],:]

    X_test = data2_0.loc[:,'1000.0':]
    #X_test.columns = X_train.columns
    y_test = data2_0.loc[:,target_variables]
    y_test.columns = target_variables
    if pca_param is True:
        X_test = pca.transform(X_test)


    #X_test = X_test[y_test.X_PN>0.005]
    #y_test = y_test[y_test.X_PN>0.005]
    
    #X_test = X_test[y_test.X_AN>0.005]
    #y_test = y_test[y_test.X_AN>0.005]
    X_test = X_test[(y_test.values > range_opt).all(1)]
    y_test = y_test[(y_test.values > range_opt).all(1)]

    R_score = model.score(X_test,y_test)

    y_predict = model.predict(X_test)

    errors = (np.abs(y_predict-y_test)/y_test)*100
    errors_abs = (np.abs(y_predict-y_test))
    #mean_err = np.mean(errors)
    mean_err = np.mean((np.abs(y_predict-y_test))*100)
    std_err = np.std(errors_abs)
    return (R_score,mean_err,std_err)

#Function to train a model with a certain number of training points, with or without PCA and

def train_and_test(n_train,nf,pca_param,model_type,range_opt):

# Define train set based on n_train

    #data_train = pd.read_csv('data_train_PN_Water_sobol.csv').iloc[0:int(n_train),:];

    X_train = data_train.loc[0:int(n_train),'1000.0':];

    column_names = X_train.columns
    y_train = data_train.loc[0:int(n_train),target_variables]


    #Doing PCA, optional
    #pca_param = True #One of the inputs of the function
    pca = PCA(3)
    pca.fit(X_train)
    #print(pca.explained_variance_ratio_)
    #print(len(pca.explained_variance_ratio_))
    if pca_param is True:
        
        X_train = pca.transform(X_train)

    #Create model, sticking to (10,10) ReLu
    if model_type == 'LR':
        model =LinearRegression()
    
    if model_type == 'ANN':
        if pca_param is True:
            model =MLPRegressor(hidden_layer_sizes=(12,),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'relu')
        else:
            model =MLPRegressor(hidden_layer_sizes=(50,50,50),tol=1e-5, max_iter=1000, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'relu')

    model.fit(X_train, y_train)
    
    R_score,mean_err,std_err = test_model_noise(model,nf,pca_param,pca,range_opt)
    return(R_score,mean_err,std_err)


#Function to train a model with a certain number of neurons, with or without PCA and 

def train_and_test_nneurons(n_neurons,nf,pca_param):

# Define train set based on n_train
    X_train = data_train.loc[0:int(n_train),'1000.0':];

    column_names = X_train.columns
    y_train = data_train.loc[0:int(n_train),target_variables]


    #Doing PCA, optional
    #pca_param = True #One of the inputs of the function
    pca = PCA(3)
    pca.fit(X_train)
    if pca_param is True:
        
        X_train = pca.transform(X_train)

    #Create model, sticking to (10,10) ReLu

    model =MLPRegressor(hidden_layer_sizes=(n_neurons,),tol=1e-5, max_iter=500, random_state=0,solver = 'lbfgs', 
                     learning_rate = 'adaptive',batch_size = 10, activation = 'relu')

    model.fit(X_train, y_train)
    n_test = 10
    range_opt = 0
    R_score,mean_err,std_err = test_model_noise(model,nf,pca_param,pca,range_opt)
    return(R_score,mean_err,std_err)

#Creating a dictionary to store relevant metrics for ANN models with different numbers of neurons
N_Neurons = np.linspace(2,60,30)
def create_dict_NN():
    
    dict_metrics = {}

    R_score_mat = pd.DataFrame(np.zeros((len(NF)*len(N_Neurons),3)),columns = ['NF','N_Neurons','R_score'])

    Errors_mat = pd.DataFrame(np.zeros((len(NF)*len(N_Neurons),len(target_variables)+2)),columns = ['NF','N_Neurons'] + target_variables)

    STD_mat = pd.DataFrame(np.zeros((len(NF)*len(N_Neurons),len(target_variables)+2)),columns = ['NF','N_Neurons'] + target_variables)

    dict_metrics['R_scores'] = R_score_mat
    dict_metrics['Errors'] = Errors_mat
    dict_metrics['STD'] = STD_mat
    for item in dict_metrics:
        frame = dict_metrics[item]
        frame.NF = np.tile(NF,len(N_Neurons))
        frame.N_Neurons = np.repeat(N_Neurons,len(NF))
        frame.set_index(['NF','N_Neurons'],inplace = True)
    return(dict_metrics)


NF = np.divide([0,25,50,75,100],100)

N_train = np.linspace(25,400,16)

#Creating a dictionary to store relevant metrics for ANN models with different numbers of training points

def create_dict():
    dict_metrics = {}

    R_score_mat = pd.DataFrame(np.zeros((len(NF)*len(N_train),3)),columns = ['NF','N_Train','R_score'])

    Errors_mat = pd.DataFrame(np.zeros((len(NF)*len(N_train),len(target_variables)+2)),columns = ['NF','N_Train'] + target_variables)

    STD_mat = pd.DataFrame(np.zeros((len(NF)*len(N_train),len(target_variables)+2)),columns = ['NF','N_Train'] + target_variables)

    dict_metrics['R_scores'] = R_score_mat
    dict_metrics['Errors'] = Errors_mat
    dict_metrics['STD'] = STD_mat
    for item in dict_metrics:
        frame = dict_metrics[item]
        frame.NF = np.tile(NF,len(N_train))
        frame.N_Train = np.repeat(N_train,len(NF))
        frame.set_index(['NF','N_Train'],inplace = True)
    return(dict_metrics)


#Creating plots of performance vs. number of training points


def make_plots(label_mat):

    mat = super_dict[label_mat]

    label_metric = 'R_scores'
    title = label_mat
    Metrics_mat = mat[label_metric]
    sns.set()
    current_palette = sns.color_palette("YlGnBu", 5)
    sns.set_palette(current_palette)
    sns.set_style(style='white')
    fig = plt.figure()
    ax = plt.axes()

    for nf in NF:
        ax.plot(N_train,Metrics_mat.loc[nf,'R_score'],marker = 'o')

    #ax.set_title('no PCA, 2 layers, 20 neurons each', fontsize=16, fontname = 'Arial')
    ax.set_title(title,fontsize=16, fontname = 'Arial')
    ax.set_ylabel('R scores', fontsize=16, fontname = 'Arial')
    ax.set_xlabel('Training set size', fontsize=16, fontname = 'Arial')
    ax.tick_params(length = 4, labelsize= 14, width =2)
    ax.legend(['NF = 0','NF = 0.25','NF = 0.5','NF = 0.75','NF = 1'], fontsize = 12, loc = 'upper right')
    #ax.set_ylim([0, 40])

    plt.show()

    label_metric = 'Errors'

    Metrics_mat = mat[label_metric]
    for item in Metrics_mat.columns:
        title = label_mat +' '+item[2:]
        fig = plt.figure()
        ax = plt.axes()

        for nf in NF:
            ax.plot(N_train,Metrics_mat.loc[nf,item],marker = 'o')

        #ax.set_title('no PCA, 2 layers, 20 neurons each', fontsize=16, fontname = 'Arial')
        ax.set_title(title,fontsize=16, fontname = 'Arial')
        ax.set_ylabel('Mean error [%]', fontsize=16, fontname = 'Arial')
        ax.set_xlabel('Training set size', fontsize=16, fontname = 'Arial')
        ax.tick_params(length = 4, labelsize= 14, width =2)
        ax.legend(['NF = 0','NF = 0.25','NF = 0.5','NF = 0.75','NF = 1'], fontsize = 12, loc = 'upper right')
        #ax.set_ylim([0, 40])

        plt.show()


    mat = super_dict[label_mat]

    label_metric = 'STD'

    Metrics_mat = mat[label_metric]
    for item in Metrics_mat.columns:
        title = label_mat +' '+item[2:]
        fig = plt.figure()
        ax = plt.axes()

        for nf in NF:
            ax.plot(N_train,Metrics_mat.loc[nf,item],marker = 'o')

        #ax.set_title('no PCA, 2 layers, 20 neurons each', fontsize=16, fontname = 'Arial')
        ax.set_title(title,fontsize=16, fontname = 'Arial')
        ax.set_ylabel('STD absolute error', fontsize=16, fontname = 'Arial')
        ax.set_xlabel('Training set size', fontsize=16, fontname = 'Arial')
        ax.tick_params(length = 4, labelsize= 14, width =2)
        ax.legend(['NF = 0','NF = 0.25','NF = 0.5','NF = 0.75','NF = 1'], fontsize = 12, loc = 'upper right')
        #ax.set_ylim([0, 40])

        plt.show()
        
#Creating plots of performance vs. number of neurons

def make_plots_NN(mat):

    #mat = super_dict[label_mat]

    label_metric = 'R_scores'
    title = 'ANN no PCA'
    Metrics_mat = mat[label_metric]
    sns.set()
    current_palette = sns.color_palette("YlGnBu", 5)
    sns.set_palette(current_palette)
    sns.set_style(style='white')
    fig = plt.figure()
    ax = plt.axes()

    for nf in NF:
        ax.plot(N_Neurons,Metrics_mat.loc[nf,'R_score'],marker = 'o')

    #ax.set_title('no PCA, 2 layers, 20 neurons each', fontsize=16, fontname = 'Arial')
    ax.set_title(title,fontsize=16, fontname = 'Arial')
    ax.set_ylabel('R scores', fontsize=16, fontname = 'Arial')
    ax.set_xlabel('Number of neurons', fontsize=16, fontname = 'Arial')
    ax.tick_params(length = 4, labelsize= 14, width =2)
    ax.legend(['NF = 0','NF = 0.25','NF = 0.5','NF = 0.75','NF = 1'], fontsize = 12, loc = 'upper right')
    #ax.set_ylim([0, 40])

    plt.show()

    label_metric = 'Errors'

    Metrics_mat = mat[label_metric]
    for item in Metrics_mat.columns:
        title = label_mat +' '+item[2:]
        fig = plt.figure()
        ax = plt.axes()

        for nf in NF:
            ax.plot(N_Neurons,Metrics_mat.loc[nf,item],marker = 'o')

        #ax.set_title('no PCA, 2 layers, 20 neurons each', fontsize=16, fontname = 'Arial')
        ax.set_title(title,fontsize=16, fontname = 'Arial')
        ax.set_ylabel('Mean error [%]', fontsize=16, fontname = 'Arial')
        ax.set_xlabel('Number of neurons', fontsize=16, fontname = 'Arial')
        ax.tick_params(length = 4, labelsize= 14, width =2)
        ax.legend(['NF = 0','NF = 0.25','NF = 0.5','NF = 0.75','NF = 1'], fontsize = 12, loc = 'upper right')
        #ax.set_ylim([0, 40])

        plt.show()



    label_metric = 'STD'

    Metrics_mat = mat[label_metric]
    for item in Metrics_mat.columns:
        print(item)
        title = label_mat +' '+item[2:]
        fig = plt.figure()
        ax = plt.axes()

        for nf in NF:
            ax.plot(N_Neurons,Metrics_mat.loc[nf,item],marker = 'o')

        #ax.set_title('no PCA, 2 layers, 20 neurons each', fontsize=16, fontname = 'Arial')
        ax.set_title(title,fontsize=16, fontname = 'Arial')
        ax.set_ylabel('STD absolute error', fontsize=16, fontname = 'Arial')
        ax.set_xlabel('Number of neurons', fontsize=16, fontname = 'Arial')
        ax.tick_params(length = 4, labelsize= 14, width =2)
        ax.legend(['NF = 0','NF = 0.25','NF = 0.5','NF = 0.75','NF = 1'], fontsize = 12, loc = 'upper right')
        #ax.set_ylim([0, 40])

        plt.show()


# In[25]:


n_train = 400
Metrics_Num_N = create_dict_NN()
for n_neurons in N_Neurons:
    for nf in NF:
        #R_score, mean_err, std_err = train_and_test(n_train,nf,True)
        #Metrics_mat.loc[(nf,n_train),:] = R_score, mean_err, std_err
        (scores,errors,std) = train_and_test_nneurons(int(n_neurons),nf,True)
        Metrics_Num_N['R_scores'].loc[(nf,n_neurons),:] = scores
        Metrics_Num_N['Errors'].loc[(nf,n_neurons),:] = errors
        Metrics_Num_N['STD'].loc[(nf,n_neurons),:] = std


# In[26]:


Metrics_linear = create_dict()

for n_train in N_train:
    for nf in NF:
        #R_score, mean_err, std_err = train_and_test(n_train,nf,True)
        #Metrics_mat.loc[(nf,n_train),:] = R_score, mean_err, std_err
        (scores,errors,std) = train_and_test(n_train,nf,True,'LR',0)
        Metrics_linear['R_scores'].loc[(nf,n_train),:] = scores
        Metrics_linear['Errors'].loc[(nf,n_train),:] = errors
        Metrics_linear['STD'].loc[(nf,n_train),:] = std


# In[27]:


Metrics_linear_05 = create_dict()
for n_train in N_train:
    for nf in NF:
        #R_score, mean_err, std_err = train_and_test(n_train,nf,True)
        #Metrics_mat.loc[(nf,n_train),:] = R_score, mean_err, std_err
        (scores,errors,std) = train_and_test(n_train,nf,True,'LR',0.005)
        Metrics_linear_05['R_scores'].loc[(nf,n_train),:] = scores
        Metrics_linear_05['Errors'].loc[(nf,n_train),:] = errors
        Metrics_linear_05['STD'].loc[(nf,n_train),:] = std

Metrics_linear_no_pca = create_dict()
for n_train in N_train:
    for nf in NF:
        #R_score, mean_err, std_err = train_and_test(n_train,nf,True)
        #Metrics_mat.loc[(nf,n_train),:] = R_score, mean_err, std_err
        (scores,errors,std) = train_and_test(n_train,nf,False,'LR',0)
        Metrics_linear_no_pca['R_scores'].loc[(nf,n_train),:] = scores
        Metrics_linear_no_pca['Errors'].loc[(nf,n_train),:] = errors
        Metrics_linear_no_pca['STD'].loc[(nf,n_train),:] = std

Metrics_linear_05_no_pca = create_dict()
for n_train in N_train:
    for nf in NF:
        #R_score, mean_err, std_err = train_and_test(n_train,nf,True)
        #Metrics_mat.loc[(nf,n_train),:] = R_score, mean_err, std_err
        (scores,errors,std) = train_and_test(n_train,nf,False,'LR',0.005)
        Metrics_linear_05_no_pca['R_scores'].loc[(nf,n_train),:] = scores
        Metrics_linear_05_no_pca['Errors'].loc[(nf,n_train),:] = errors
        Metrics_linear_05_no_pca['STD'].loc[(nf,n_train),:] = std

Metrics_ANN = create_dict()

for n_train in N_train:
    for nf in NF:
        #R_score, mean_err, std_err = train_and_test(n_train,nf,True)
        #Metrics_mat.loc[(nf,n_train),:] = R_score, mean_err, std_err
        (scores,errors,std) = train_and_test(n_train,nf,True,'ANN',0)
        Metrics_ANN['R_scores'].loc[(nf,n_train),:] = scores
        Metrics_ANN['Errors'].loc[(nf,n_train),:] = errors
        Metrics_ANN['STD'].loc[(nf,n_train),:] = std

Metrics_ANN = create_dict()

for n_train in N_train:
    for nf in NF:
        #R_score, mean_err, std_err = train_and_test(n_train,nf,True)
        #Metrics_mat.loc[(nf,n_train),:] = R_score, mean_err, std_err
        (scores,errors,std) = train_and_test(n_train,nf,True,'ANN',0)
        Metrics_ANN['R_scores'].loc[(nf,n_train),:] = scores
        Metrics_ANN['Errors'].loc[(nf,n_train),:] = errors
        Metrics_ANN['STD'].loc[(nf,n_train),:] = std


# In[28]:


super_dict_keys = ['Linear pca','Linear pca 0.005','Linear no pca', 'Linear no pca 0.005', 'ANN pca']
super_dict={}
super_dict[super_dict_keys[0]]=Metrics_linear
super_dict[super_dict_keys[1]]=Metrics_linear_05
super_dict[super_dict_keys[2]]=Metrics_linear_no_pca
super_dict[super_dict_keys[3]]=Metrics_linear_05_no_pca
super_dict[super_dict_keys[4]]=Metrics_ANN


# In[29]:


color1 = 'darkblue'
color2 = 'darkorange'
labels = ['ANN pca','Linear pca']
n_train  = 400

fig = plt.figure()
ax = plt.axes()
gap = 2 / len(NF)
X = np.arange(len(NF))

plt.bar(X + (-1) * gap/2,super_dict[labels[0]]['R_scores'].xs(n_train,level = 1).R_score,
            width = gap,
            color = color1,
            linewidth = 1, edgecolor = 'black')
plt.bar(X + (1) * gap/2, super_dict[labels[1]]['R_scores'].xs(n_train,level = 1).R_score,
            width = gap,
            color = color2,
            linewidth = 1, edgecolor = 'black')
#ax.set_title('1 Layer, 50 points, NF 0.25', fontsize=20, fontname = 'Arial')

ax.set_ylabel('R-score', fontsize=16, fontname = 'Arial')
ax.set_xlabel('NF', fontsize=16, fontname = 'Arial')
ax.tick_params(length = 4, labelsize= 14, width =2)
ax.set_ylim([0.98,1.001])
plt.legend(labels, fontsize = 14, loc = 'lower right')
plt.xticks(np.arange(5),NF)
fig.set_size_inches(6, 5)


# In[30]:


super_dict[labels[0]]['R_scores'].xs(n_train,level = 1).R_score


# In[31]:


for i in super_dict[labels[0]]['Errors'].columns:
    fig = plt.figure()
    ax = plt.axes()
    gap = 2 / len(NF)
    X = np.arange(len(NF))

    plt.bar(X + (-1) * gap/2,super_dict[labels[0]]['Errors'].xs(n_train,level = 1)[i],
                width = gap,
                color = color1,
                linewidth = 1, edgecolor = 'black')
    plt.bar(X + (1) * gap/2, super_dict[labels[1]]['Errors'].xs(n_train,level = 1)[i],
                width = gap,
                color = color2,
                linewidth = 1, edgecolor = 'black')
    #ax.set_title('1 Layer, 50 points, NF 0.25', fontsize=20, fontname = 'Arial')

    ax.set_ylabel('Mean Error [%]', fontsize=16, fontname = 'Arial')
    ax.set_xlabel('NF', fontsize=16, fontname = 'Arial')
    ax.tick_params(length = 4, labelsize= 14, width =2)
    ax.set_title(i[2:], fontsize = 16)
    #ax.set_ylim([0.98,1.001])
    plt.legend(labels, fontsize = 14, loc = 'lower right')
    plt.xticks(np.arange(5),NF)
    fig.set_size_inches(6, 5)


# In[32]:


for i in super_dict[labels[0]]['STD'].columns:
    fig = plt.figure()
    ax = plt.axes()
    gap = 2 / len(NF)
    X = np.arange(len(NF))

    plt.bar(X + (-1) * gap/2,super_dict[labels[0]]['STD'].xs(n_train,level = 1)[i],
                width = gap,
                color = color1,
                linewidth = 1, edgecolor = 'black')
    plt.bar(X + (1) * gap/2, super_dict[labels[1]]['STD'].xs(n_train,level = 1)[i],
                width = gap,
                color = color2,
                linewidth = 1, edgecolor = 'black')
    #ax.set_title('1 Layer, 50 points, NF 0.25', fontsize=20, fontname = 'Arial')

    ax.set_ylabel('STD of error', fontsize=16, fontname = 'Arial')
    ax.set_xlabel('NF', fontsize=16, fontname = 'Arial')
    ax.tick_params(length = 4, labelsize= 14, width =2)
    ax.set_title(i[2:], fontsize = 16)
    #ax.set_ylim([0.98,1.001])
    plt.legend(labels, fontsize = 14, loc = 'lower right')
    plt.xticks(np.arange(5),NF)
    fig.set_size_inches(6, 5)


# In[33]:


make_plots('ANN pca')


# In[34]:


label_mat = 'ANN pca'
make_plots_NN(Metrics_Num_N)


# In[35]:


make_plots('Linear pca')


# In[36]:


make_plots('Linear no pca')


# In[37]:


for label in super_dict.keys():
    sns.set()
    current_palette = sns.color_palette("YlGnBu", 6)
    sns.set_palette(current_palette)
    sns.set_style(style='white')
    fig = plt.figure()
    ax = plt.axes()


    n_train = 200

    components = super_dict[label]['Errors'].columns
    for i in components:
        ax.plot(NF,super_dict[label]['Errors'].xs(n_train,level = 1)[i],marker = 'o')

    #ax.set_title('no PCA, 2 layers, 20 neurons each', fontsize=16, fontname = 'Arial')
    ax.set_title(label,fontsize=16, fontname = 'Arial')
    ax.set_ylabel('Errors [%]', fontsize=16, fontname = 'Arial')
    ax.set_xlabel('NF', fontsize=16, fontname = 'Arial')
    ax.tick_params(length = 4, labelsize= 14, width =2)
    plt.xticks(NF)
    ax.legend(components, fontsize = 12, loc = 'upper left')


# In[38]:


import pickle

f = open("dict_AN_3_MAE.pkl","wb")
pickle.dump(super_dict,f)
f.close()


# In[ ]:




