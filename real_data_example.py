#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""@author: Yubai Yuan
"""


# This is the code for reproducing the real data example resutls in Table 3
# plesae first install following packages 
# "tensoeflow", "Keras", "econml", "rpy2" for competing methods

#import all the relevant modules   

import csv
import os

#set the path to the location of NAS dataset
os.chdir("Path_to_real data")


import random
import pandas
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
import scipy
import seaborn as sns
import copy
#import tensorly as tl
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from scipy.linalg import hankel
import time

from numpy.linalg import inv
from sklearn.decomposition import PCA
import keras
from keras import layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from econml.sklearn_extensions.model_selection import GridSearchCVList
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from econml.orf import DMLOrthoForest

from sklearn.datasets import make_circles
from keras import backend as K
from keras.constraints import UnitNorm, Constraint    
from sklearn.datasets import make_swiss_roll

#define functions used for proposed deconfounding algorithms and comopeting methods
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def para_grad(var,Loss_1,Loss_2,tune,ind):
    part_1 = 2*np.sum(-Loss_1*var)
    part_2 = 2*tune*(np.sum(Loss_1*Loss_2)/(LA.norm(Loss_1)*LA.norm(Loss_2)))*(np.sum(-var*Loss_2)/(LA.norm(Loss_1)*LA.norm(Loss_2)) -  (np.sum(Loss_1*Loss_2)*np.sum(-Loss_1*var))/(LA.norm(Loss_1)**3*LA.norm(Loss_2)))
    if ind:
       return (part_1+part_2)
    else:
       return ((1/2)*part_1+part_2)

def para_grad_vec(var,Loss_1,Loss_2,tune,ind):
    part_1 = -2*np.matmul(Loss_1.transpose(),var)
    part_2 = 2*tune*(np.sum(Loss_1*Loss_2)/(LA.norm(Loss_1)*LA.norm(Loss_2)))*(-np.matmul(Loss_2.transpose(),var)/(LA.norm(Loss_1)*LA.norm(Loss_2)) +  (np.sum(Loss_1*Loss_2)*np.matmul(Loss_1.transpose(),var))/(LA.norm(Loss_1)**3*LA.norm(Loss_2)))
    if ind:
       return (part_1+part_2)
    else:
       return ((1/2)*part_1+part_2)
   
def kernel_similarity(X, bandwidth, self_metric):
    m = X.shape[0]
    kernel_sim = np.zeros([m,m])
    for i in range(X.shape[1]):
      kernel_sim =  kernel_sim + self_metric[i,i]*abs(X[:,i].reshape(m,1)-X[:,i].reshape(1,m))
    return np.exp(-kernel_sim/bandwidth)

def corr_grad(Loss_M,gamma_U): 
    m = Loss_M.shape[1]
    corr_mat = np.corrcoef(Loss_M.transpose()) - np.diag(np.ones(m))
    grad_U = np.zeros(n)
    for i in range(Loss_M.shape[1]-1):
        for j in range((i+1),Loss_M.shape[1]):
            numrate = - gamma_U[i]*Loss_M[:,j] - gamma_U[j]*Loss_M[:,i] + np.sum(Loss_M[:,i]*Loss_M[:,j])*(gamma_U[i]*Loss_M[:,i]/LA.norm(Loss_M[:,i])**2 + gamma_U[j]*Loss_M[:,j]/LA.norm(Loss_M[:,j])**2)
            grad_U = grad_U + 2*corr_mat[i,j]*numrate/(LA.norm(Loss_M[:,i])*LA.norm(Loss_M[:,j]))
    return grad_U 

def corr_grad_T(Loss_M,treatment,P_est,gamma_U,gamma_t):   
    m = Loss_M.shape[1]
    corr_mat = np.corrcoef(Loss_M.transpose()) - np.diag(np.ones(m))
    grad_U = np.zeros(n)
    for i in range(Loss_M.shape[1]-1):
        for j in range((i+1),Loss_M.shape[1]):
            numrate = - gamma_U[i]*Loss_M[:,j] - gamma_U[j]*Loss_M[:,i] + np.sum(Loss_M[:,i]*Loss_M[:,j])*(gamma_U[i]*Loss_M[:,i]/LA.norm(Loss_M[:,i])**2 + gamma_U[j]*Loss_M[:,j]/LA.norm(Loss_M[:,j])**2)
            grad_U = grad_U + 2*corr_mat[i,j]*numrate/(LA.norm(Loss_M[:,i])*LA.norm(Loss_M[:,j]))
    
    T_res = (treatment - P_est).reshape(n,)
    corr_t = np.corrcoef(np.concatenate((T_res.reshape(n,1),Loss_M),axis=1).transpose())[0,1:]
    grad_U_t = np.zeros(n)
    for i in range(Loss_M.shape[1]):
        numrate = - gamma_U[i]*T_res - gamma_t*(P_est-P_est**2).reshape(n,)*Loss_M[:,i] + np.sum(Loss_M[:,i]*T_res)*(gamma_U[i]*Loss_M[:,i]/LA.norm(Loss_M[:,i])**2 + gamma_t*T_res*(P_est-P_est**2).reshape(n,)/LA.norm(T_res)**2)
        grad_U_t = grad_U_t + 2*corr_t[i]*numrate/(LA.norm(Loss_M[:,i])*LA.norm(T_res))

    return grad_U + grad_U_t 


def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
 
def RBF_map(U,phi,order):
        feature_map = np.zeros([U.shape[0],order+1])
        scale = np.exp(-phi*(LA.norm(U,axis=1)**2).reshape(U.shape[0],1))
        a = 1
        if order > 1:
           a = (((2*phi)**order)/np.math.factorial(order))**0.5
        for i in range(order+1):
            j = order - i
            feature_map[:,i] = (a*scale*((U[:,0]**i)*(U[:,1]**j)).reshape(U.shape[0],1)).reshape(U.shape[0],)
        return feature_map        
    
 
def first_stage_reg():
    return GridSearchCVList([Lasso(),
                              RandomForestRegressor(n_estimators=100, random_state=123),
                              GradientBoostingRegressor(random_state=123)],
                              param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                                {'max_depth': [3, 10, 15],
                                                'min_samples_leaf': [10, 25, 50]},
                                              {'n_estimators': [50, 100, 200],
                                                'max_depth': [3,10,15],
                                                'min_samples_leaf': [10, 25, 50]}],
                              cv=5,
                              scoring='neg_mean_squared_error')

def first_stage_reg_1():
    return GridSearchCVList([
                             RandomForestRegressor(n_estimators=10, random_state=123),
                             GradientBoostingRegressor(random_state=123)],
                             param_grid_list=[
                                               {'max_depth': [3, 5, 7],
                                               'min_samples_leaf': [5, 10, 15]},
                                              {'n_estimators': [5, 10, 15],
                                               'max_depth': [3,5,7],
                                               'min_samples_leaf': [5, 10,15]}],
                             cv=3,
                             scoring='neg_mean_squared_error')



def first_stage_clf():
    return GridSearchCVList([LogisticRegression()],
                             param_grid_list=[{'C': [0.01, .1, 1, 10]}],
                             cv=3,
                             scoring='neg_mean_squared_error')    

 
    
def treat_loss_fn(y_true,y_pred):
      
        y_pred_t = tf.clip_by_value(y_pred, clip_value_min=0.05, clip_value_max=0.95)
        
        treatment_loss =  -tf.reduce_sum(tf.log(y_pred_t)*y_true +  tf.log(1-y_pred_t)*(1-y_true)) - tf.reduce_sum(y_pred_t*tf.log(y_pred_t)) 
        return treatment_loss
        
       
   
def corr_loss_fn(Data):
       
     def corr_fn(T_M_Y,T_M_Y_hat):   
        
        residual = T_M_Y - T_M_Y_hat
        corr_1 = tf.reduce_sum(tf.square(tfp.stats.correlation(residual)) - tf.linalg.diag(np.float32(np.ones(m+2))))
        corr_2 = tf.reduce_sum(tf.square(tf.linalg.tensor_diag_part(tfp.stats.correlation(residual,Data - residual))))
        return corr_1 + corr_2
     return corr_fn
    
def outcome_loss_fn(y_true,y_pred):
        a1 = tf.reduce_sum(tf.square(y_true[:,0] - y_pred[:,0]))
        a2 = tf.reduce_sum(tf.square(y_true[:,1:] - y_pred[:,1:]))
        return a1 + a2  
    
def norm_loss_fn(y_true,y_pred):
        a = tf.reduce_sum(tf.square(y_true - y_pred))
        return a
    
class UncorrelatedFeaturesConstraint_target (Constraint):
    
    def __init__(self, encoding_dim, target, weightage = 1.0 ):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.target = tf.cast(target, tf.float32)
    
    def get_covariance_target(self, x):
        corr_target_list = []

        for i in range(self.encoding_dim):
            corr_target_list.append(tf.math.abs(tfp.stats.correlation(x[:, i], self.target, sample_axis=0, event_axis=None)))
            
        corr_target = tf.stack(corr_target_list)
        total_corr_target = K.sum(K.square(corr_target))
        return total_corr_target
            
   

    def __call__(self, x):
        self.covariance = self.get_covariance_target(x)
        return self.weightage * self.covariance 
    
    
class correlatedFeaturesConstraint_target (Constraint):
    
    def __init__(self, encoding_dim, target, weightage = 1.0 ):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.target = tf.cast(target, tf.float32)
    
    def get_covariance_target_2(self, x):
        corr_target_list = []

        for i in range(self.encoding_dim):
            corr_target_list.append( 1 - tf.math.abs(tfp.stats.correlation(x[:, i], self.target, sample_axis=0, event_axis=None))  )
            
        corr_target = tf.stack(corr_target_list)
        total_corr_target = K.sum(K.square(corr_target))
        return total_corr_target
 
    def __call__(self, x):
        self.covariance = self.get_covariance_target_2(x)
        return self.weightage * self.covariance     


sess = tf.InteractiveSession() 


###import module for running HIMA method

import rpy2
print(rpy2.__version__)

from rpy2.rinterface import R_VERSION_BUILD
print(R_VERSION_BUILD)

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1) 


packnames = ('ncvreg', 'doParallel', 'HIMA')
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
    
    
from rpy2.robjects import FloatVector
from rpy2.robjects import DataFrame
import rpy2.robjects.numpy2ri
from rpy2.robjects import r
import rpy2.robjects as ro
    
rpy2.robjects.numpy2ri.activate()
importr('HIMA')   



#############read the NAS data as pandas dataframe "NIH_pheno_data" (657 rows and 17 columns)
#NIH_pheno_data = pandas.read_csv(‘phs000853.v1.pht004429.v1.p1.c1.Normative_Aging_Study_Subject_Phenotypes.GRU.txt’, 
                                  #sep=’ ‘, header=None)

#############read the index for selected 22 mediators ("mediator_index") and 
#############index for selected samples after preprocessing ("sample_index")
#############both files of "mediator_index.csv" and "sample_index.csv" are stored in the github 

#############read the NAS data as pandas dataframe "NIH_pheno_data" (657 rows and 17 columns)
# NIH_geno_data = pandas.read_csv(‘DNA_Methylation_data_beta_values.txt’, 
                                  #sep=’ ‘, header=None)

select_ID = csv.reader('sample_index.csv', delimiter=',')
mediator_index = csv.reader('mediator_index.csv', delimiter=',')

N = 657
N_1 = 519
m = 22
NIH_mediator = NIH_geno_data.loc[mediator_index]

treatment = NIH_pheno_data["Smoking_categories_NAS"].values
treatment = treatment.reshape(N,1)
outcome = NIH_pheno_data["Time_to_Death_NAS"].values
Y = NIH_pheno_data["Time_to_Death_NAS"].values
outcome = outcome.reshape(N,1)
#certralize the outcome 
outcome = outcome - np.mean(outcome)
##select covariates
X = np.concatenate((NIH_pheno_data["Years_Education_NAS"].values.reshape(N,1),NIH_pheno_data["Diabetes_NAS"].values.reshape(N,1),NIH_pheno_data["Hypertension_NAS"].values.reshape(N,1),
                    NIH_pheno_data["Coronary_Heart_Disease_NAS"].values.reshape(N,1),NIH_pheno_data["Neutrophils_NAS"].values.reshape(N,1),NIH_pheno_data["Lymphocytes_NAS"].values.reshape(N,1),
                    NIH_pheno_data["Monocytes_NAS"].values.reshape(N,1),NIH_pheno_data["Eosinophils_NAS"].values.reshape(N,1),
                    NIH_pheno_data["Basophils_NAS"].values.reshape(N,1)),axis = 1)

NIH_med = NIH_mediator.values[:,1:]
NIH_med = NIH_med.astype(np.float64)

treatment_select = treatment[select_ID]
outcome_select = outcome[select_ID]
X_select = X[select_ID,:]
NIH_med_select = NIH_med[select_ID,:]


###repeat 30 times 

repeat = 30
treat_fa = np.zeros(repeat)
treat_m_fa_1 = np.zeros(repeat)
treat_m_fa_2 = np.zeros(repeat)
treat_d_fa = np.zeros(repeat)
#mse_fa = np.zeros(repeat)
med_fa = np.zeros(repeat)

treat_auto = np.zeros(repeat)
treat_m_auto_1 = np.zeros(repeat)
treat_m_auto_2 = np.zeros(repeat)
treat_d_auto = np.zeros(repeat)
#mse_auto_3 = np.zeros(repeat)
med_auto_3 = np.zeros(repeat)

treat_lr = np.zeros(repeat)
treat_m_lr = np.zeros(repeat)
treat_d_lr = np.zeros(repeat)
#mse_lr = np.zeros(repeat)
med_lr = np.zeros(repeat)

treat_forest = np.zeros(repeat)
mse_forest = np.zeros(repeat)
med_forest = np.zeros(repeat)

treat_X = np.zeros(repeat)
#mse_X = np.zeros(repeat)
med_X = np.zeros(repeat)


treat_hima = np.zeros(repeat)
treat_m_hima = np.zeros(repeat)
treat_d_hima = np.zeros(repeat)
#mse_hima = np.zeros(repeat)
med_hima = np.zeros(repeat)



for ii in range(repeat):

#fix the random seed for each run       
    np.random.seed(ii)
#split data into training and testing set

    test_ID = np.random.choice(range(N_1),math.floor(0.2*N_1),replace = False)
    train_ID = np.setdiff1d(np.array(range(N_1)), test_ID) 
    
    train_X = X_select[train_ID,:]
    test_X = X_select[test_ID,:]
    train_outcome = outcome_select[train_ID] 
    test_outcome = outcome_select[test_ID] 
    train_treatment = treatment_select[train_ID] 
    test_treatment = treatment_select[test_ID] 
    train_med = NIH_med_select[train_ID,:] 
    test_med = NIH_med_select[test_ID,:]
    
    
    r_X = X[flat_outlier,:]
    r_outcome = outcome[flat_outlier] 
    r_treatment = treatment[flat_outlier] 
   
    r_med = NIH_med[flat_outlier,:]
    
    
    n_1 = test_ID.shape[0]
    n = train_ID.shape[0] 
    n_r = flat_outlier.shape[0]

#####proposed method using matrix factorization as latent confounding model    
##set dimension of surrogate confounder as k = 12

    k = 12
#initialize the outcome and mediators model based on random forest 
       
    rf_m_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
    rf_m_y.fit(train_med,train_outcome.reshape(n,))
    m_y_fit = rf_m_y.predict(X = train_med).reshape(n,1)
    m_y_test = rf_m_y.predict(X = test_med).reshape(n_1,1)
    rf_x_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
    rf_x_y.fit(train_X,train_outcome.reshape(n,))
    x_y_fit = rf_x_y.predict(X = train_X).reshape(n,1)
    x_y_test = rf_x_y.predict(X = test_X).reshape(n_1,1)
    
    ####
    
    X_Y_U = np.concatenate((m_y_fit.reshape(n,1),x_y_fit.reshape(n,1),train_treatment),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
    res_Y = train_outcome - reg_Y_U.predict(X_Y_U)
    
    X_M_U = np.concatenate((train_X,train_treatment),axis=1) 
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U,train_med)
    res_M = train_med - reg_M_U.predict(X_M_U)

##obtain the initial of U  
     
    U_scale = 1
    pca = PCA(n_components=k)
    pca.fit(np.concatenate((res_Y.reshape(n,1),res_M),axis=1))
    pca.components_.T
    pca.explained_variance_ratio_
    #np.corrcoef(np.concatenate((pca.fit_transform(M - reg_M.predict(X_M_r)),U),axis=1).transpose())    
    U_ini = pca.fit_transform(np.concatenate((res_Y.reshape(n,1),res_M),axis=1)) #+ 2*np.random.randn(n, k) 
    U_ini = ((U_scale/np.linalg.norm(U_ini, axis=0))*U_ini).reshape(n,k)
    #Uhat = np.zeros([n,k])
    Uhat = copy.copy(U_ini)
    U_begin = copy.copy(U_ini)

    lambda_1 = 10
    lambda_2 = 0.1
    
    U_est = copy.copy(U_ini)

##initialize the parameters        
    X_M_U = np.concatenate((U_est,train_X,train_treatment.reshape(n,1)),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
    gamma_M = reg_M_U.coef_[:,0:k].transpose()
    beta_M = reg_M_U.coef_[:,k:(k+9)]
    T_M =  reg_M_U.coef_[:,-1]
    M_int = reg_M_U.intercept_

   
    X_Y_U = np.concatenate((train_treatment.reshape(n,1),U_est,x_y_fit,m_y_fit),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
    beta_Y = reg_Y_U.coef_[0,k+1]
    beta_Y_M = reg_Y_U.coef_[0,k+2]
    Y_int = reg_Y_U.intercept_
    gamma_Y = reg_Y_U.coef_[0,1:(k+1)].reshape(k,1)
    beta_Y_T = reg_Y_U.coef_[0,0]

    
    Loss_Y = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T)
    #Loss_Y = train_outcome - reg_Y_U.predict(X_Y_U)
    Loss_Y_center_U =  Loss_Y - np.mean(Loss_Y,axis = 0)   
    #reg_Y_bias = LinearRegression(fit_intercept=False).fit(U_begin, Loss_Y_center_U)
    reg_Y_bias = LinearRegression(fit_intercept=True).fit(U_begin, Loss_Y)
    gamma_Y = reg_Y_bias.coef_.transpose()

    
    Loss_M = train_med - reg_M_U.predict(X_M_U)
    Loss_M_center_U =  Loss_M - np.mean(Loss_M,axis = 0)  

    gamma_T = 0.1*np.ones(k).reshape(k,1)
    T_int = 0
    P_est = sigmoid(np.matmul(U_begin,gamma_T)+T_int)
    
    Loss_M_T = np.concatenate((train_treatment - P_est,Loss_Y_center_U,Loss_M_center_U),axis=1)
    
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
    loss_T = -np.sum(np.log(P_est)*train_treatment + np.log(1-P_est)*(1-train_treatment)) #- np.sum(P_est*np.log(P_est))
  
    scale = 1/n
    
    Loss_total = np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
    
    outcome_loss_track = [np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2)]
    
     
    
    step_size = 0.00002
    
    total_loss_1 = copy.copy(Loss_total)
    
    total_loss = 0
    iter = 1
    loss_track = [total_loss_1]
    loss_track_1 = [np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2)]
    loss_track_Y = [np.sum(Loss_Y**2)]
    loss_track_M = [np.sum(Loss_M**2)]
    loss_track_corr_res = [corr_res]
    loss_track_T = [loss_T]
    
    loss_ind = 1000*total_loss_1
    
    
    #T_M_ini = 0 
    beta_Y_M_ini = 0
    #beta_M_ini = 0
    beta_Y_ini = 0
    beta_Y_T_ini = 0
    
    h_PCA_treatment = 0
    h_PCA_treatment_m = 0
    h_PCA_MSE = 0
    h_PCA_Y_T = 0
    med_effect_1 = 0
    med_effect_2 = 0
    M_pred_test = np.zeros([n_1,1])
    
    x_y_fit_ini =  np.zeros([n,1])
    m_y_fit_ini =  np.zeros([n,1])
    
    med_effect_1_track = []
    med_effect_2_track = []
    dir_effect_track = []
    pred_track = []
    pred_track_mse = []
    r_pred_track = []
    r_pred_track_mse = []
    
    fit_track = []
    
    Mse_fa = 0
    mse_fa_3 = 0
    
    pred_track_res = []
    
    
    while iter<= 100:
##update surrogate confounder 
        
          total_loss_1 = copy.copy(total_loss)
           
          Uhat = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(train_X,np.matmul(inv(np.matmul(train_X.transpose(),train_X)),train_X.transpose()))),Uhat)
   
          grad_U_k = np.zeros([n,k])
            
          for kk in range(k):
                
                gamma_U = gamma_M[kk,:]
                
                gamma_t = gamma_T[kk,0]
                #grad_corr = corr_grad(Loss_M,gamma_U) 
                
                Loss_M_Y = np.concatenate((Loss_Y_center_U,Loss_M_center_U),axis=1)
                gamma_U_Y = np.concatenate((gamma_Y[kk,:],gamma_U))
                
                grad_corr = corr_grad_T(Loss_M_Y,train_treatment,P_est,gamma_U_Y,gamma_t)   
                
                grad_U_k[:,kk] = ( - 2*gamma_Y[kk,:]*Loss_Y_center_U  - np.sum(2*gamma_M[kk,:]*Loss_M_center_U,1).reshape(n,1) + lambda_1*grad_corr.reshape(n,1) + lambda_2*(-gamma_T[kk,0]*(train_treatment - P_est)).reshape(n,1)).reshape(n,)

          Uhat = Uhat - step_size*grad_U_k
          Uhat = Uhat*(U_scale/np.linalg.norm(Uhat, axis=0))
          
          ##update rf mediator
          
          m_y_outcome = train_outcome - x_y_fit*beta_Y - train_treatment*beta_Y_T - np.matmul(Uhat,gamma_Y).reshape(n,1) - Y_int
          
          rf_m_y.fit(train_med,m_y_outcome.reshape(n,))
          m_y_fit = rf_m_y.predict(train_med).reshape(n,1)
          
          ##update rf x
          
          x_y_outcome = train_outcome - m_y_fit*beta_Y_M - train_treatment*beta_Y_T - np.matmul(Uhat,gamma_Y).reshape(n,1) - Y_int
          rf_x_y.fit(train_X,x_y_outcome.reshape(n,))
          x_y_fit = rf_x_y.predict(train_X).reshape(n,1)
         
          
          ##update y
          
          U_est = copy.copy(Uhat)
          Y_bias = np.matmul(U_est,gamma_Y)
          X_Y_U = np.concatenate((train_treatment.reshape(n,1),Y_bias,x_y_fit,m_y_fit),axis=1)
          reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
          beta_Y = reg_Y_U.coef_[0,2]
          beta_Y_M = reg_Y_U.coef_[0,3]
          Y_int = reg_Y_U.intercept_
          gamma_Y_bias = reg_Y_U.coef_[0,1]
          beta_Y_T = reg_Y_U.coef_[0,0]
          
          
          
          M_pred  =  beta_Y_M*rf_m_y.predict(test_med).reshape(n_1,1) + beta_Y*rf_x_y.predict(test_X).reshape(n_1,1)
          r_M_pred  =  beta_Y_M*rf_m_y.predict(r_med).reshape(n_r,1) + beta_Y*rf_x_y.predict(r_X).reshape(n_r,1)
          
          
          ##update m
         
          X_M_U = np.concatenate((U_est, train_X,train_treatment.reshape(n,1)),axis=1)
          reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
          gamma_M = reg_M_U.coef_[:,0:k].transpose()
          beta_M = reg_M_U.coef_[:,k:(k+9)]
          T_M =  reg_M_U.coef_[:,-1]
          M_int = reg_M_U.intercept_
          
          
          reg_U = LinearRegression(fit_intercept=False).fit(U_est, train_treatment.reshape(n,))
          np.corrcoef(np.concatenate((train_treatment.reshape(n,1),reg_U.predict(U_est).reshape(n,1)),axis=1).transpose())          
          np.corrcoef(np.concatenate((train_treatment.reshape(n,1),train_outcome.reshape(n,1)),axis=1).transpose())          

          ##update P

          treat_model = LogisticRegression(fit_intercept=False).fit(U_est, train_treatment.reshape(n,))
          P_est = treat_model.predict_proba(U_est)[:,1].reshape(n,1)
          gamma_T = treat_model.coef_.transpose()
         
         ### calculate mediator effect 1
          X_M_U_1 = np.concatenate((U_est,train_X,np.ones(n).reshape(n,1)),axis=1)
          X_M_U_0 = np.concatenate((U_est,train_X,np.zeros(n).reshape(n,1)),axis=1)
          
          Med_effect_1 = np.mean(rf_m_y.predict(reg_M_U.predict(X_M_U_1)).reshape(n,1)- rf_m_y.predict(reg_M_U.predict(X_M_U_0)).reshape(n,1))
          
         ### calculate mediator effect 2
          med_1_fak = reg_M_U.predict(X_M_U_1)
          med_0_fak = reg_M_U.predict(X_M_U_0)
         
          med_1 = np.zeros([n,m])
          med_0 = np.zeros([n,m])
          for i in range(n):
              if train_treatment[i] == 1:
                  med_1[i,:] = train_med[i,:]
                  med_0[i,:] = med_0_fak[i,:]
              else:
                  med_1[i,:] = med_1_fak[i,:]
                  med_0[i,:] = train_med[i,:]
         
          Med_effect_2 = np.mean(rf_m_y.predict(med_1).reshape(n,1) - rf_m_y.predict(med_0).reshape(n,1))
        
          ##update loss
            
          Loss_Y = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T)
          #Loss_Y = train_outcome - (np.matmul(train_X,beta_Y).reshape(n,1) + np.matmul(train_med,beta_Y_M).reshape(n,1) + train_treatment*beta_Y_T)
          reg_Y_bias = LinearRegression(fit_intercept=True).fit(U_est, Loss_Y) #- np.mean(Loss_Y))
          #reg_Y_bias = LinearRegression(fit_intercept=False).fit(U_est, Loss_Y)
          gamma_Y = reg_Y_bias.coef_.transpose()
        
          Loss_Y_U = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T + np.matmul(U_est, gamma_Y))
          #Loss_Y_U = train_outcome - (np.matmul(train_X,beta_Y).reshape(n,1) + np.matmul(train_med,beta_Y_M).reshape(n,1) + train_treatment*beta_Y_T + np.matmul(U_est, gamma_Y))
          Loss_Y_center_U =  Loss_Y_U - np.mean(Loss_Y_U,axis = 0) 
          fit_track.append((sum(Loss_Y_U**2)/n)**0.5)

          Loss_M_U = train_med - reg_M_U.predict(X_M_U)
          Loss_M_center_U =  Loss_M_U - np.mean(Loss_M_U,axis = 0)  
          Loss_M_pred = train_med - (np.matmul(train_X[:,0:9],beta_M.transpose()).reshape(n,m) +  train_treatment*T_M)
          
          Loss_M_T = np.concatenate((train_treatment - P_est,Loss_Y_center_U,Loss_M_center_U),axis=1)
  
    
          corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
    
          loss_T = -np.sum(np.log(P_est)*train_treatment + np.log(1-P_est)*(1-train_treatment)) #- np.sum(P_est*np.log(P_est))

          total_loss = np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
          
          iter = iter + 1

          loss_track.append(total_loss)
          loss_track_1.append(np.sum(Loss_Y_center_U**2) + np.sum(Loss_M_center_U**2))
          loss_track_corr_res.append(corr_res)
          loss_track_T.append(loss_T)
          
          med_effect_1_track.append(Med_effect_1)
          med_effect_2_track.append(Med_effect_2)
          dir_effect_track.append(beta_Y_T)
          
          #######  out-sample prediction
          
          rf_train = np.concatenate((train_treatment,train_med),axis=1)
          rf_label = Y_bias 

          rf = RandomForestRegressor(n_estimators = 100, max_depth=12,random_state = 123)
          #rf = RandomForestRegressor(n_estimators = 50, max_depth=6,random_state = 123)

   
          Loss_M_U_test = test_med - (np.matmul(test_X[:,0:9],beta_M.transpose()).reshape(n_1,m) +  test_treatment*T_M)
          r_Loss_M_U_test = r_med - (np.matmul(r_X[:,0:9],beta_M.transpose()).reshape(n_r,m) +  r_treatment*T_M)
        
          rf.fit(rf_train,rf_label.reshape(n,))
          #Y_bias_pred = rf.predict(np.concatenate((test_treatment,Loss_M_U_test),axis=1)).reshape(n_1,1)
          Y_bias_pred = rf.predict(np.concatenate((test_treatment,test_med),axis=1)).reshape(n_1,1)

          r_Y_bias_pred = rf.predict(np.concatenate((r_treatment,r_Loss_M_U_test),axis=1)).reshape(n_r,1)
    
          pred_Y = M_pred + test_treatment*beta_Y_T + gamma_Y_bias*Y_bias_pred.reshape(n_1,1) + Y_int
          r_pred_Y = r_M_pred + r_treatment*beta_Y_T + gamma_Y_bias*r_Y_bias_pred.reshape(n_r,1) + Y_int
          
          mse_pca_2 = (sum((pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5
          mse_pca_2_r = (sum((r_pred_Y.reshape(n_r,1) - r_outcome.reshape(n_r,1))**2)/n_r)**0.5
          pred_track_mse.append(mse_pca_2)
          r_pred_track_mse.append(mse_pca_2_r)
          
          mse_pca_3 = np.median(abs(pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1)))
          mse_pca_3_r = np.median(abs(r_pred_Y.reshape(n_r,1) - r_outcome.reshape(n_r,1)))
          pred_track.append(mse_pca_3)
          r_pred_track.append(mse_pca_3_r)
          
          
            
          if total_loss < loss_ind:
                
            
                  
                U_his = Uhat
                X_Y = np.concatenate((train_treatment,U_his,x_y_fit,m_y_fit),axis=1)
                reg_Y = LinearRegression(fit_intercept=False).fit(X_Y, train_outcome)
                Y_bias = np.matmul(U_his,reg_Y.coef_[0,1:(k+1)].transpose()).reshape(n,1)
                
                h_PCA_Y_T = beta_Y_T 
                
                
                loss_ind  = total_loss
                M_pred_test = M_pred 
                
                med_effect_1 = Med_effect_1
                med_effect_2 = Med_effect_2
                
                T_M_ini = T_M
                beta_Y_M_ini = beta_Y_M
                beta_M_ini = beta_M
                beta_Y_ini = beta_Y
                beta_Y_T_ini = beta_Y_T   
                gamma_Y_ini = gamma_Y
                gamma_Y_bias_ini = gamma_Y_bias
                x_y_fit_ini =  x_y_fit
                m_y_fit_ini =  m_y_fit
                Y_int_ini = Y_int
                
                Med_fa = min(pred_track)
                Mse_fa = min(pred_track_mse)
                
               
                
          print(iter)
          
          
    
    #mse_fa[ii] = Mse_fa
    med_fa[ii] = Med_fa
    treat_m_fa_1[ii] = med_effect_1*beta_Y_M_ini
    treat_m_fa_2[ii] = med_effect_2*beta_Y_M_ini
    treat_d_fa[ii] = beta_Y_T
    treat_fa[ii] = beta_Y_T + med_effect_1*beta_Y_M_ini
    
   



########proposed method using autoencoder as latent confounding model 
    
##set up penality parameters

    lambda_1 = 1
    lambda_2 = 0.1
    
    res_Y = train_outcome - (Y_int + x_y_fit_ini*beta_Y_ini + m_y_fit_ini*beta_Y_M_ini + train_treatment*beta_Y_T_ini)
    res_Y = res_Y - np.mean(res_Y)
   
    
    res_M = train_med - (M_int + np.matmul(train_X,beta_M_ini.transpose()).reshape(n,m) + T_M_ini*train_treatment)
    res_M = res_M - np.mean(res_M,axis = 0)
  
    Loss_M_T = np.concatenate((train_treatment,res_Y,res_M),axis=1)
   
    corr_res = LA.norm(np.corrcoef(Loss_M_T.transpose()) - np.diag(np.ones(m+2)))**2
   
    P_est = sigmoid(np.matmul(U_begin,gamma_T)+T_int)
  
    loss_T = 1000
    
    Loss_total = np.sum(res_Y**2) + np.sum(res_M**2) + lambda_1*corr_res + lambda_2*loss_T #+ lambda_3*loss_L #+ lambda_4*loss_orth
 
    step_size = 0.05
    
    total_loss_1 = copy.copy(Loss_total)
    #loss_L_1 = copy.copy(loss_L)
    #loss_L = 0
    total_loss = 0
    iter = 1
    loss_track = [total_loss_1]
    loss_track_Y = [np.sum(res_Y**2)+np.sum(res_M**2)]
    #loss_track_M = [np.sum(Loss_M**2)]
    loss_track_corr_res = [corr_res**2]
    loss_track_T = [loss_T]
    loss_norm = [0]
   
   
    loss_ind = 100*total_loss_1
    corr_track= []
    
    mse_auto = 0
    MSE_auto = 0
    pred_track_auto = []
    pred_track_auto_mse = []
    
    r_pred_track_auto = []
    r_pred_track_auto_mse = []
    
    Med_effect_auto_1 = 0
    Med_effect_auto_2 = 0
    med_effect_auto_1 = 0
    med_effect_auto_2 = 0
    dir_effect_auto = 0
  
    
    Data = np.float32(np.concatenate((train_treatment,train_outcome,train_med),axis=1))
     
###set up the architecture of autoencoder      
    input_U_1 = keras.Input(shape=(m+1,))
    
    x1 = layers.Dense((m+1), activation='selu')(input_U_1)
    x1 = layers.Dense(int(np.ceil(2*(m+1))), activation='selu')(x1)
    x3 = layers.Dense(int(np.ceil(3*(m+1))), activation='selu', activity_regularizer = UncorrelatedFeaturesConstraint_target(int(np.ceil(3*(m+1))),train_treatment.reshape(n,), weightage = 10))(x1)
 
    x2 = layers.Dense(4, activation='selu')(x1)
    x2 = layers.Dense(4, activation='selu')(x2)
     
    treatment_pred = layers.Dense(1, activation='sigmoid')(x2)
    
    encoded = layers.concatenate([x2,x3])
    
    y = layers.Dense(int(np.ceil(2*(m+1))), activation='selu')(x3)
    y = layers.Dense(int(np.ceil(1*(m+1))), activation='selu')(y)
    decoded = layers.Dense((m+1))(y)
    T_M_Y_hat = layers.concatenate([treatment_pred,decoded])
    
    autoencoder = keras.Model(inputs = [input_U_1], outputs=[treatment_pred,decoded,T_M_Y_hat])
    encoder = keras.Model(inputs = [input_U_1], outputs= encoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02),loss=["binary_crossentropy",outcome_loss_fn,corr_loss_fn(Data = Data),],loss_weights=[lambda_2,1,lambda_1])
 
    x_y_fit = x_y_fit_ini
    m_y_fit = m_y_fit_ini
    beta_Y = beta_Y_ini
    beta_Y_T = beta_Y_T_ini
    gamma_Y = gamma_Y_ini
    beta_Y_M = beta_Y_M_ini

    while iter<=5:
       
         total_loss_1 = copy.copy(total_loss)
         M_Y = np.float32(np.concatenate((res_Y,res_M),axis=1))
           
         X_tr = np.concatenate((train_treatment,np.ones([n,1])),axis = 1)
         Bias_norm_orth = np.matmul((np.diag(np.ones([n,1]).reshape(n,)) - np.matmul(X_tr,np.matmul(inv(np.matmul(X_tr.transpose(),X_tr)),X_tr.transpose()))),M_Y)
         T_M_Y = np.float32(np.concatenate((train_treatment,res_Y,res_M),axis=1))
           
         autoencoder.fit(M_Y, [train_treatment,Bias_norm_orth,T_M_Y],
                    epochs= 1500,   #+ 100*math.floor(iter),
                    batch_size= n,verbose = 1)
         
     
         T_M_Y_hat = autoencoder.predict(M_Y)[2]
         if math.isnan(np.sum(T_M_Y_hat)):
               break
         
         encoded_U = encoder.predict(M_Y)
         Uhat = encoded_U
          
         bce = tf.keras.losses.BinaryCrossentropy()
            
         loss_T = bce(train_treatment,T_M_Y_hat[:,0].reshape(n,1)).eval(session=sess)      
       
         Loss_Y_M_T = np.concatenate((train_treatment - T_M_Y_hat[:,0].reshape(n,1),res_Y - T_M_Y_hat[:,1].reshape(n,1),res_M - T_M_Y_hat[:,2:].reshape(n,m)),axis=1)
     
         corr_res = LA.norm(np.corrcoef(Loss_Y_M_T.transpose()) - np.diag(np.ones(m+2)))**2
       
         total_loss = outcome_loss_fn(M_Y,T_M_Y_hat[:,1:]).eval() + lambda_1*corr_res + lambda_2*loss_T
     
         loss_track.append(total_loss) 
         
         k1 = int(np.ceil(3*(m+1)))+4
        
         U_est = copy.copy(Uhat)
         X_M_U = np.concatenate((U_est,train_X,train_treatment),axis=1)
         reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
         gamma_M = reg_M_U.coef_[:,0:k1].transpose()
         beta_M = reg_M_U.coef_[:,k1:(k1+9)]
         T_M =  reg_M_U.coef_[:,-1]
         M_int = reg_M_U.intercept_
      
         res_M = train_med - (M_int + np.matmul(train_X,beta_M.transpose()).reshape(n,m) + T_M*train_treatment)
         res_M = res_M - np.mean(res_M,axis =0)
         
         Loss_M_pred = train_med - (np.matmul(train_X[:,0:9],beta_M.transpose()).reshape(n,m) +  train_treatment*T_M)
         
        ######
        ##update rf mediator
          
         m_y_outcome = train_outcome - x_y_fit*beta_Y - train_treatment*beta_Y_T -  T_M_Y_hat[:,1].reshape(n,1) -Y_int
         rf_m_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
         rf_m_y.fit(train_med,m_y_outcome.reshape(n,))
         m_y_fit = rf_m_y.predict(train_med).reshape(n,1)
        
        ##update rf x
        
         x_y_outcome = train_outcome - m_y_fit*beta_Y_M - train_treatment*beta_Y_T -  T_M_Y_hat[:,1].reshape(n,1) -Y_int
         rf_x_y = RandomForestRegressor(max_depth=6, min_samples_leaf=5, random_state=123)
         rf_x_y.fit(train_X,x_y_outcome.reshape(n,))
         x_y_fit = rf_x_y.predict(train_X).reshape(n,1)
       
        
        ##update y
        
         
         Y_bias = T_M_Y_hat[:,1].reshape(n,1)
         X_Y_U = np.concatenate((train_treatment.reshape(n,1),Y_bias,x_y_fit,m_y_fit),axis=1)
         reg_Y_U = LinearRegression(fit_intercept=True).fit(X_Y_U, train_outcome)
         beta_Y = reg_Y_U.coef_[0,2]
         beta_Y_M = reg_Y_U.coef_[0,3]
         Y_int = reg_Y_U.intercept_
         gamma_Y_bias = 1
         beta_Y_T = reg_Y_U.coef_[0,0]
        
        
         M_pred  =  beta_Y_M*rf_m_y.predict(test_med).reshape(n_1,1) + beta_Y*rf_x_y.predict(test_X).reshape(n_1,1)
     
         ### calculate mediator effect 1
         X_M_U_1 = np.concatenate((U_est,train_X,np.ones(n).reshape(n,1)),axis=1)
         X_M_U_0 = np.concatenate((U_est,train_X,np.zeros(n).reshape(n,1)),axis=1)
         
         Med_effect_auto_1 = np.mean(rf_m_y.predict(reg_M_U.predict(X_M_U_1)).reshape(n,1)- rf_m_y.predict(reg_M_U.predict(X_M_U_0)).reshape(n,1))
         
        ### calculate mediator effect 2
         med_1_fak = reg_M_U.predict(X_M_U_1)
         med_0_fak = reg_M_U.predict(X_M_U_0)
        
         med_1 = np.zeros([n,m])
         med_0 = np.zeros([n,m])
         for i in range(n):
             if train_treatment[i] == 1:
                 med_1[i,:] = train_med[i,:]
                 med_0[i,:] = med_0_fak[i,:]
             else:
                 med_1[i,:] = med_1_fak[i,:]
                 med_0[i,:] = train_med[i,:]
        
         Med_effect_auto_2 = np.mean(rf_m_y.predict(med_1).reshape(n,1) - rf_m_y.predict(med_0).reshape(n,1))
   
     
        ######
         res_Y = train_outcome - (Y_int + x_y_fit*beta_Y + m_y_fit*beta_Y_M + train_treatment*beta_Y_T)
         res_Y = res_Y - np.mean(res_Y,axis = 0)
         rf_train = np.concatenate((train_treatment,train_med),axis=1)
         rf_label = Y_bias 
         rf = RandomForestRegressor(n_estimators = 50, max_depth=6,random_state = 123)
 
          
         rf.fit(rf_train,rf_label.reshape(n,))
         Y_bias_pred = rf.predict(np.concatenate((test_treatment,test_med),axis=1)).reshape(n_1,1)
         
         pred_Y = M_pred + test_treatment*beta_Y_T + gamma_Y_bias*Y_bias_pred.reshape(n_1,1) + Y_int
          
         pred_track_auto_mse.append((sum((pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5)
         pred_track_auto.append(np.median(abs(pred_Y.reshape(n_1,1) - test_outcome.reshape(n_1,1))))
         
      
         
         if min(pred_track_auto) < loss_ind:
             
    
                U_his = Uhat
                  
                med_effect_auto_1 = Med_effect_auto_1
                med_effect_auto_2 = Med_effect_auto_2
                dir_effect_auto = beta_Y_T
                
                Y_bias = T_M_Y_hat[:,1].reshape(n,1)
                 
                
                loss_ind  = min(pred_track_auto)
                Med_auto = min(pred_track_auto)
                Mse_auto = min(pred_track_auto_mse)
                
                beta_Y_M_auto = beta_Y_M
    
         iter = iter + 1
         print(iter)
          
    treat_m_auto_1[ii] = med_effect_auto_1*beta_Y_M_auto
    treat_m_auto_2[ii] = med_effect_auto_2*beta_Y_M_auto
    treat_d_auto[ii] = dir_effect_auto
    med_auto_3[ii] = Med_auto
    #mse_auto_3[ii] = Mse_auto
    treat_auto[ii] = dir_effect_auto + med_effect_auto_1*beta_Y_M_auto
    
   
#########Competing method


###########Causal Forest method

    
   
    covariate = np.concatenate((train_X,train_med),axis=1)
   
    model_y = clone(first_stage_reg_1().fit(covariate, train_outcome).best_estimator_)
   
    model_t = clone(first_stage_clf().fit(covariate, train_treatment).best_estimator_)
   
    est = CausalForestDML(model_y=model_y,
                      model_t=model_t,
                      discrete_treatment=False,
                      cv=3,
                      n_estimators=20,
                      random_state=123)
    
    est.fit(train_outcome, train_treatment, X=covariate)
    
    pred = est.effect(X = covariate)
    
    treat_forest[ii] = np.mean((pred))
   
    pred_covariate = np.concatenate((test_X, test_med),axis=1)  
    
    #mse_forest[ii] = (sum((est.models_y[0][0].predict(X = pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_forest[ii] = np.median(abs(est.models_y[0][0].predict(X = pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1)))
  
      

    

    ###X learner method
    
    X_learner_models = GradientBoostingRegressor(n_estimators=20, max_depth=4, min_samples_leaf=int(n/10))
    X_learner_propensity_model = RandomForestClassifier(n_estimators=20, max_depth=4, 
                                               min_samples_leaf=int(n/10))
    X_learner = XLearner(models=X_learner_models, propensity_model=X_learner_propensity_model)
    # Train X_learner
    X_learner.fit(train_outcome, train_treatment, X=covariate)
    # Estimate treatment effects on test data
    X_te = X_learner.effect(covariate)
    treat_X[ii] = np.mean(X_te)
    X_learner_models.fit(covariate,train_outcome)
    
    
    #mse_X[ii] = (sum((X_learner_models.predict(pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_X[ii] = np.median(abs(X_learner_models.predict(pred_covariate).reshape(n_1,1) - test_outcome.reshape(n_1,1)))
    
    #(sum((X_learner_models.predict(covariate).reshape(n,1) - train_outcome.reshape(n,1))**2)/n)**0.5 



###########LSEM method
    
    X_M_U = np.concatenate((train_X,train_treatment),axis=1)
    reg_M_U = LinearRegression(fit_intercept=False).fit(X_M_U, train_med)
    beta_M = reg_M_U.coef_[:,0:9]
    T_M =  reg_M_U.coef_[:,-1]
    M_int = reg_M_U.intercept_
   
    res_M = train_med - reg_M_U.predict(X_M_U) 
    X_Y_U = np.concatenate((train_treatment,train_med,train_X),axis=1)
    reg_Y_U = LinearRegression(fit_intercept=False).fit(X_Y_U, train_outcome)
    beta_Y_M = reg_Y_U.coef_[0,1:(m+1)].reshape(m,1)
    beta_Y_T = reg_Y_U.coef_[0,0]
    beta_Y = reg_Y_U.coef_[0,(m+1):].reshape(9,1)
    Y_ini =  reg_Y_U.intercept_
      
    treat_m_lr[ii] = np.sum(beta_Y_M.transpose()*T_M)
    
    treat_d_lr[ii] = beta_Y_T
    
    treat_lr[ii] = np.sum(beta_Y_M.transpose()*T_M) + beta_Y_T
    
    pred_Y_SEM =  reg_Y_U.intercept_ + np.matmul(test_X,beta_Y) + np.matmul(test_med,beta_Y_M) + test_treatment*beta_Y_T + Y_ini
 
#    mse_lr[ii] = (sum((pred_Y_SEM.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_lr[ii] = np.median(abs(pred_Y_SEM.reshape(n_1,1) - test_outcome.reshape(n_1,1)))
      
 ###### HIMA method

    treatment_R = robjects.FloatVector(train_treatment)
    Y_R = robjects.FloatVector(train_outcome)
    nr,nc = train_med.shape
    M_R = rpy2.robjects.r.matrix(train_med, nrow=nr, ncol=nc)
    rpy2.robjects.r.assign("M", M_R)
   
    
    HIMA_R = rpy2.robjects.r['hima'](X = treatment_R, Y = Y_R, M = M_R)
    hima_para = np.asarray(HIMA_R)
    num = hima_para.shape[1]
    select_med = np.int16(np.asarray(HIMA_R.rownames))-1
    
    treat_m_hima[ii] = np.sum(hima_para[3,:])
    treat_hima[ii] = hima_para[2,0]
    treat_d_hima[ii] = hima_para[2,0] - np.sum(hima_para[3,:])
   
     
    pred_Y_hima = np.matmul(test_med[:,select_med],hima_para[1,:].reshape(num,1)) + test_treatment*(hima_para[2,0] - np.sum(hima_para[3,:]))
#    mse_hima[ii] = (sum((pred_Y_hima.reshape(n_1,1) - test_outcome.reshape(n_1,1))**2)/n_1)**0.5 
    med_hima[ii] = np.median(abs(pred_Y_hima.reshape(n_1,1) - test_outcome.reshape(n_1,1)))



### generate the result in Table 3
np.mean(treat_fa)
np.mean(treat_m_fa_1) 
np.mean(treat_d_fa) 
np.mean(med_fa)

np.mean(treat_auto)
np.mean(treat_m_auto_1) 
np.mean(treat_d_auto) 
np.mean(med_auto_3)

np.mean(treat_lr)
np.mean(treat_m_lr) 
np.mean(treat_d_lr) 
np.mean(med_lr)

np.mean(treat_forest )
np.mean(med_forest)

np.mean(treat_X)
np.mean(med_X)

np.mean(treat_hima)
np.mean(treat_m_hima)
np.mean(treat_d_hima)
np.mean(med_hima)













        


