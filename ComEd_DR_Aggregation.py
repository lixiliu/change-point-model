#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:07:39 2020

@author: lliu2
"""

### setting
import os
import re
import sys
import json
import pandas as pd
import functools
import itertools
import zipfile
import tempfile
import shutil
import pytz
import datetime as dt
from dask.distributed import Client, wait, LocalCluster
import dask.dataframe as dd
import dask
from dask import delayed
from dask import compute
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from getpass import getuser
import numpy as np
from dateutil.tz import tzoffset
from dateutil.relativedelta import relativedelta
import traceback
from IPython.display import HTML
from operator import itemgetter
import pickle
import math
from scipy.stats import gaussian_kde, linregress, norm
from scipy.stats import t as tstats
from scipy import optimize
import statistics as stats
import lmfit
from calendar import monthrange
#import ruptures as rpt
#from ruptures.utils import pairwise
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
import time
from multiprocessing import Pool
from sklearn.preprocessing import LabelBinarizer
import random
import scipy
from pandas.tseries.offsets import DateOffset

#get_ipython().run_line_magic('matplotlib', 'inline')
#sns.set_style('whitegrid')
#sns.set_context('talk')
plt.style.use('default')

username = getuser()
username

# specify paths
prefix = '/Users/lliu2/Documents/' #'/Users/lliu2/OneDrive - NREL/Documents/'
datadir = os.path.join(prefix,'Project 1 - Energy Analysis/Python/zip codes/ComEd_zipcode_avg_AWS/')
datadir2 = os.path.join(prefix, 'Project 1 - Energy Analysis/Python/zip codes/')
outdir = os.path.join(prefix, 'Project 1 - Energy Analysis/Python/zip codes/DR_heating_cooling/')


# use dir(modulename) to check available attributes for the module

# %%
def linear(x, a0, a1):
    return a0 + a1*x

def linearc(x, a0, a1, x0):
    return a0 + a1*(x-x0)

def cubic(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*(x**2) + a3*(x**3)

def quartic(x, a0, a1, a2, a3, a4):
    return a0 + a1*x + a2*(x**2) + a3*(x**3) + a4*(x**4)

def quintic(x, a0, a1, a2, a3, a4, a5):
    a1 *= 1e-1; a2 *= 1e-2; a3 *= 1e-3; a4 *= 1e-4; a5 *= 1e-5;
    return a0 + a1*x + a2*(x**2) + a3*(x**3) + a4*(x**4) + a5*(x**5)

def logistic(x, L, k, x0, b0):
    return L/(1+np.exp(-k*(x-x0)))+b0

def fourier1(t, p, c, a1, b1):
    # 2pi/w = p or period or cycle length (e.g. 24-hr)
    # if p, cycle = 24hr, then:
    # a1, b1 = effects at 24-hr cycle (24/1), larger val = stronger effects
    # a2, b2 = effects at 12-hr cycle, (24/2)...
    
    w = 2*np.pi/p # in radian
    return c + a1*np.cos(t*w) + b1*np.sin(t*w)

def fourier2(t, p, c, a1, b1, a2, b2):
    # 2pi/w = p or period or cycle length (e.g. 24-hr)
    # if p, cycle = 24hr, then:
    # a1, b1 = effects at 24-hr cycle (24/1), larger val = stronger effects
    # a2, b2 = effects at 12-hr cycle, (24/2)...
    
    w = 2*np.pi/p # in radian
    # w: 24 hr;  2w: 12 hr
    return c + a1*np.cos(t*w) + b1*np.sin(t*w) + a2*np.cos(2*t*w) + b2*np.sin(2*t*w)

def fourier3(t, p, c, a1, b1, a2, b2, a3, b3, k2, k3):
    w = 2*np.pi/p # in radian
    return c + a1*np.cos(t*w) + b1*np.sin(t*w) + a2*np.cos(k2*t*w) + b2*np.sin(k2*t*w) +\
            a3*np.cos(k3*t*w) + b3*np.sin(k3*t*w)

def fourier14(t, p, c, a1, b1, a7, b7, a14, b14):
    w = 2*np.pi/p # in radian
    # w: 1 week; 7w: 24 hr;  14w: 12 hr
    return c + a1*np.cos(t*w) + b1*np.sin(t*w) + a7*np.cos(7*t*w) + b7*np.sin(7*t*w) +\
            a14*np.cos(14*t*w) + b14*np.sin(14*t*w)
            
def fourier_top4(t, p, c, a1, b1, a2, b2, a3, b3, a4, b4, k2, k3, k4):
    w = 2*np.pi/p # in radian
    # w: 1 year; 2w: half year;  366w: 24hr; 732w: 12 hr
    return c + a1*np.cos(t*w)+b1*np.sin(t*w) + a2*np.cos(k2*t*w)+b2*np.sin(k2*t*w) +\
            a3*np.cos(k3*t*w)+b3*np.sin(k3*t*w) + a4*np.cos(k4*t*w)+b4*np.sin(k4*t*w)
            
def fourier_top8(t, p, c, a1, b1, a2, b2, a3, b3, a4, b4, 
                  a5, b5, a6, b6, a7, b7, a8, b8,
                  k2, k3, k4, k5, k6, k7, k8):
    w = 2*np.pi/p # in radian
    # w: 1 year; 2w: half year;  366w: 24hr; 732w: 12 hr
    return c + a1*np.cos(t*w)+b1*np.sin(t*w) + a2*np.cos(k2*t*w)+b2*np.sin(k2*t*w) +\
            a3*np.cos(k3*t*w)+b3*np.sin(k3*t*w) + a4*np.cos(k4*t*w)+b4*np.sin(k4*t*w) +\
                a5*np.cos(k5*t*w)+b5*np.sin(k5*t*w) + a6*np.cos(k6*t*w)+b6*np.sin(k6*t*w) +\
                    a7*np.cos(k7*t*w)+b7*np.sin(k7*t*w) + a8*np.cos(k8*t*w)+b8*np.sin(k8*t*w)

def piecewise_linear(x, x0, b0, mh, mc):
    condlist = [x<=x0] # + else
    funclist = [mh*(x-x0)+b0, mc*(x-x0)+b0]
    return np.where(condlist[0], funclist[0], funclist[1])
    
def rsq(data,model):
    return (np.corrcoef(data,model)[0,1])**2

##############################################################################
def load_pickle(name):
    """ Function to load an object from a pickle """
    with open(f'{name}.pkl', 'rb') as f:
        temp = pickle.load(f)
    return temp


def save_pickle(contents, name):
    """ Function to save to an object as a pickle """
    with open(f'{name}.pkl', 'wb') as output:
        pickle.dump(contents, output, pickle.HIGHEST_PROTOCOL)


def load_dill(name):
    """ Function to load an object from a dill """
    with open(f'{name}.dll', 'rb') as f:
        temp = dill.load(f)
    return temp


def save_dill(contents, name):
    """ Function to save to an object as a dill """
    with open(f'{name}.dll', 'wb') as output:
        dill.dump(contents, output, dill.HIGHEST_PROTOCOL)

print('Model math and pickle/dill functions ran.')

# %% [1] load data
data_type = 'half_hourly' #<------ hourly or half_hourly

# check if these files exist:
file_status = 1
for C in ['C23', 'C24', 'C25', 'C26']:
    file_status = file_status*os.path.exists(os.path.join(datadir,
                                        'ComEd_zipcode_{}_avg_w_temp_{}.parquet'.format(data_type,C)))

# if processed files exist, call directly, else create from AWS queries
if file_status:
    # % option (1) - load directly, if file exists
    print('DF exists, loading DF directly...')
    DF = {}
    for C in ['C23', 'C24', 'C25', 'C26']:
        DF[C] = pd.read_parquet(os.path.join(datadir,'ComEd_zipcode_{}_avg_w_temp_{}.parquet'.format(data_type,C)),
          engine='pyarrow')

else:
    # option (2) - create from AWS queried dfs
    print('Creating DF from scratch...')
    df = pd.read_parquet(os.path.join(datadir,'comed_zipcode_{}_avg_201510-201703.parquet'.format(data_type)),engine='pyarrow')
    
    # break out df into a dictionary
    DF = {}
    for C in ['C23', 'C24', 'C25', 'C26']:
        dfi = df.query('type == @C').reset_index(drop=True)        
        
        ### missing timestamp 2016-03-31 23:30, fill with 5th order polynomial
        dfi = dfi.drop(['type'], axis=1) # remove type
        dfi = dfi[dfi['mean']>0] # take only all positive values
        dfi = dfi.set_index(['time','zip_code']).unstack(level='zip_code')
        IDX = dfi.index.get_level_values(level='time')
        IDX = pd.date_range(IDX[0], IDX[-1], freq='0.5H').rename('time')
        dfi = dfi.reindex(IDX, fill_value=np.NAN)
        dfi = dfi.interpolate(method='time', limit=2, limit_area='inside', axis=0) # fill missing val with 5th order
        dfi = dfi.stack().reset_index().sort_values(
            by=['zip_code','time']).reset_index(drop=True)
        dfi['type'] = C # add type back in
        DF[C] = dfi
    display(DF['C23'])
    
    df = pd.concat(DF.values(), axis=0).reset_index(drop=True)
    
    # save new df
    df.to_parquet(os.path.join(datadir,'comed_zipcode_{}_avg_201510-201703.parquet'.format(data_type)),engine='pyarrow', flavor='spark')
    
    del df # save memory
    
    pref = 'HalfHourly' if data_type == 'half_hourly' else 'Hourly'
    HHTemp = pd.read_csv(os.path.join(datadir2,'W{}_temp_by_station_201510-201703.csv'.format(pref)), 
                       parse_dates=['time']).set_index('time')
    
    zip_stn = pd.read_csv(os.path.join(datadir2,'Wzipcode_station_lookup.csv'))
    zip_stn = dict(zip(zip_stn['zip_code'],zip_stn['station'].astype(str)))
    
    for C in DF.keys():
        print('>>> mapping temp to {}...'.format(C))
        ### add temp
        DF[C]['degC'] = DF[C].set_index([DF[C]['zip_code'].map(zip_stn),
           'time']).index.map(HHTemp.unstack().to_dict())
        ### export as parquets
        DF[C].to_parquet(os.path.join(datadir,'ComEd_zipcode_{}_avg_w_temp_{}.parquet'.format(data_type,C)),
          engine='pyarrow', flavor='spark')
    
display(DF['C26'])

def data_set_up(service_type, in_W=True, log_transform=True):
    global DF, data_type
    
    ts1 = pd.Timestamp('2015-12-01')
    ts2 = pd.Timestamp('2016-12-01')
    
    # filter on timestamps
    DF2 = DF[service_type].query('(time>=@ts1)&(time<@ts2)').set_index(
        ['time','zip_code'])[['mean','degC']].unstack(level='zip_code')

    # assign df_temp, y, t
    m = 1000 if in_W else 1
    multiplier = 2*m if data_type == 'half_hourly' else m
    df_temp,df_demand = DF2['degC'],DF2['mean']*multiplier # <--- degC, W
    
    # log transformation
    df_demand = np.log(df_demand) if log_transform else df_demand
    
    ### get FFT values
    ext = '_log' if log_transform else ''
    
    ### FFT from shoulder
    fft = pd.read_parquet(os.path.join(datadir,'df_zipcode_sorted_FFT_{}{}.parquet'.format(service_type, ext)),
      engine='pyarrow').set_index(['type','index'])
    fft.columns = fft.columns.astype('int')
    
    fftn = pd.read_parquet(os.path.join(datadir,'df_zipcode_non_HVAC_sorted_FFT_{}{}.parquet'.format(service_type, ext)),
      engine='pyarrow').set_index(['type','index'])
    fftn.columns = fftn.columns.astype('int')
    
    ffth = pd.read_parquet(os.path.join(datadir,'df_zipcode_heating_sorted_FFT_{}{}.parquet'.format(service_type, ext)),
      engine='pyarrow').set_index(['type','index'])
    ffth.columns = ffth.columns.astype('int')
    
    fftc = pd.read_parquet(os.path.join(datadir,'df_zipcode_cooling_sorted_FFT_{}.parquet'.format(service_type, ext)),
      engine='pyarrow').set_index(['type','index'])
    fftc.columns = fftc.columns.astype('int')
    
    print('    data is in [{}]'.format('W(h)' if in_W else 'kW(h)'))
    
    return df_temp, df_demand, fft, fftn, ffth, fftc
    

def folder_set_up(service_type, fourier_type='42', log_transform=True):
    global DF, data_type, outdir
    log = 'log' if log_transform else ''
    
    # sub folder for results
    outdir_sub = os.path.join(outdir, 'Model{}_{}_{}_{}'.format(fourier_type,log,service_type,data_type))
    if not os.path.exists(outdir_sub):
        os.mkdir(outdir_sub)
    
    # sub-sub folder for plots
    outdir_fig = os.path.join(outdir_sub,'plots')
    if not os.path.exists(outdir_fig):
        os.mkdir(outdir_fig)
        
    return outdir_sub

print('ran')

# %% model funcs (orig - no time constraint)
def linear_fourier_top4(x, y, t, # data
                    xh, yh, xc, yc, # domain bounds
                    mh, bh, mc, bc, # heat/cool lines
                    b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, # fourier 0 HVAC-off reg
                    bh0, bh1c, bh1s, bh2c, bh2s, bh3c, bh3s, bh4c, bh4s, # fourier heat
                    bc0, bc1c, bc1s, bc2c, bc2s, bc3c, bc3s, bc4c, bc4s, # fourier cool reg
                    p, ph, pc, k2, k3, k4, kh2, kh3, kh4, kc2, kc3, kc4, # fourier period and fourier term multiplers
                    modtype='B'):
    
    # domain-wise regressions
    l0 = fourier_top4(t, p, b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, k2, k3, k4) # HVAC-off domain
    lh = linearc(x, bh, mh, xh)*fourier_top4(t, ph, bh0, bh1c, bh1s, bh2c, bh2s, bh3c, bh3s, bh4c, bh4s, kh2, kh3, kh4) # heat domain
    lc = linearc(x, bc, mc, xc)*fourier_top4(t, pc, bc0, bc1c, bc1s, bc2c, bc2s, bc3c, bc3s, bc4c, bc4s, kc2, kc3, kc4) # cool domain
    if modtype != 'A':
        lh = lh + l0
        lc = lc + l0
        
    condlist = [(x<xh) & (y>yh), (x>xc) & (y>yc)] # + else
    funclist = [lh, lc, l0]
    return np.where(condlist[0], funclist[0], 
                    np.where(condlist[1], funclist[1], funclist[2]))  

def HVAC_demands_linear_fourier_top4(x, y, t, tidx, V, log_transform=True, modtype='B'):
    
    Xh = x[(x<V['xh'])&(y>V['yh'])]; Th = t[(x<V['xh'])&(y>V['yh'])]; 
    tidx = np.array(tidx); tidxh = tidx[(x<V['xh'])&(y>V['yh'])];
    
    l0h = fourier_top4(Th, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lh = linearc(Xh, V['bh'], V['mh'], V['xh']) * fourier_top4(Th, V['ph'], V['bh0'], 
                    V['bh1c'], V['bh1s'], V['bh2c'], V['bh2s'], V['bh3c'], V['bh3s'], 
                    V['bh4c'], V['bh4s'], V['kh2'], V['kh3'], V['kh4']) # heat domain
        
    Xc = x[(x>V['xc']) & (y>V['yc'])]; Tc = t[(x>V['xc']) & (y>V['yc'])];
    tidxc = tidx[(x>V['xc']) & (y>V['yc'])]
    l0c = fourier_top4(Tc, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'], V['b4c'], V['b4s'],
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lc = linearc(Xc, V['bc'], V['mc'], V['xc']) * fourier_top4(Tc, V['pc'], V['bc0'], 
                    V['bc1c'], V['bc1s'], V['bc2c'], V['bc2s'], V['bc3c'], V['bc3s'], 
                    V['bc4c'], V['bc4s'], V['kc2'], V['kc3'], V['kc4']) # cool domain
    
    if log_transform:
        # need to back-transform
        if modtype == 'A':
            lh = np.exp(lh) - np.exp(l0h)
            lc = np.exp(lc) - np.exp(l0c)
        else:
            lh = np.exp(lh+l0h) - np.exp(l0h)
            lc = np.exp(lc+l0c) - np.exp(l0c)
    else:
        if modtype == 'A':
            lh -= l0h
            lc -= l0c
    return [tidxh, tidxc], [Xh, Xc], [lh, lc]

def linear_fourier_top4_24_12(x, y, t, # data
                    xh, yh, xc, yc, # domain bounds
                    mh, bh, mc, bc, # heat/cool lines
                    b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, # fourier 0 HVAC-off reg
                    bh0, bh1c, bh1s, bh2c, bh2s, # fourier heat
                    bc0, bc1c, bc1s, bc2c, bc2s, # fourier cool reg
                    p, ph, pc, k2, k3, k4, # fourier period and fourier term multiplers
                    modtype='B'):
    
    # domain-wise regressions
    l0 = fourier_top4(t, p, b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, k2, k3, k4) # HVAC-off domain
    lh = linearc(x, bh, mh, xh)*fourier2(t, ph, bh0, bh1c, bh1s, bh2c, bh2s) # heat domain
    lc = linearc(x, bc, mc, xc)*fourier2(t, pc, bc0, bc1c, bc1s, bc2c, bc2s) # cool domain
    #if modtype != 'A':
    lh = lh + l0
    lc = lc + l0
        
    condlist = [(x<xh) & (y>yh), (x>xc) & (y>yc)] # + else
    funclist = [lh, lc, l0]
    return np.where(condlist[0], funclist[0], 
                    np.where(condlist[1], funclist[1], funclist[2]))  

def HVAC_demands_linear_fourier_top4_24_12(x, y, t, tidx, V, log_transform=True, modtype='B'):
    
    Xh = x[(x<V['xh'])&(y>V['yh'])]; Th = t[(x<V['xh'])&(y>V['yh'])]; 
    tidx = np.array(tidx); tidxh = tidx[(x<V['xh'])&(y>V['yh'])];
    
    l0h = fourier_top4(Th, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lh = linearc(Xh, V['bh'], V['mh'], V['xh']) * fourier2(Th, V['ph'], V['bh0'], 
                    V['bh1c'], V['bh1s'], V['bh2c'], V['bh2s']) # heat domain
        
    Xc = x[(x>V['xc']) & (y>V['yc'])]; Tc = t[(x>V['xc']) & (y>V['yc'])]; tidxc = tidx[(x>V['xc']) & (y>V['yc'])]
    l0c = fourier_top4(Tc, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'], V['b4c'], V['b4s'],
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lc = linearc(Xc, V['bc'], V['mc'], V['xc']) * fourier2(Tc, V['pc'], V['bc0'], 
                    V['bc1c'], V['bc1s'], V['bc2c'], V['bc2s']) # cool domain
    
    if log_transform:
        # need to back-transform
        if modtype == 'A':
            lh = np.exp(lh) - np.exp(l0h)
            lc = np.exp(lc) - np.exp(l0c)
        else:
            lh = np.exp(lh+l0h) - np.exp(l0h)
            lc = np.exp(lc+l0c) - np.exp(l0c)
    else:
        if modtype == 'A':
            lh -= l0h
            lc -= l0c
    return [tidxh, tidxc], [Xh, Xc], [lh, lc]

print(' "linear_fourier models" and "get HVAC load models" (original) ran.')

# %% model func (with time constraints)

def linear_fourier_top4(x, y, t, # data
                    xh, yh, xc, yc, # domain bounds
                    mh, bh, mc, bc, # heat/cool lines
                    b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, # fourier 0 HVAC-off reg
                    bh0, bh1c, bh1s, bh2c, bh2s, bh3c, bh3s, bh4c, bh4s, # fourier heat
                    bc0, bc1c, bc1s, bc2c, bc2s, bc3c, bc3s, bc4c, bc4s, # fourier cool reg
                    p, ph, pc, k2, k3, k4, kh2, kh3, kh4, kc2, kc3, kc4, # fourier period and fourier term multiplers
                    modtype='B'):
    global tseason
    
    # domain-wise regressions
    l0 = fourier_top4(t, p, b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, k2, k3, k4) # HVAC-off domain
    lh = linearc(x, bh, mh, xh)*fourier_top4(t, ph, bh0, bh1c, bh1s, bh2c, bh2s, bh3c, bh3s, bh4c, bh4s, kh2, kh3, kh4) # heat domain
    lc = linearc(x, bc, mc, xc)*fourier_top4(t, pc, bc0, bc1c, bc1s, bc2c, bc2s, bc3c, bc3s, bc4c, bc4s, kc2, kc3, kc4) # cool domain
    if modtype != 'A':
        lh = lh + l0
        lc = lc + l0
        
    condlist = [(x<xh) & (y>yh) & ((t<tseason[0])|(t>=tseason[3])), 
                (x>xc) & (y>yc) & (t>=tseason[1]) & (t<tseason[2])] # + else
    funclist = [lh, lc, l0]
    return np.where(condlist[0], funclist[0], 
                    np.where(condlist[1], funclist[1], funclist[2]))  

def HVAC_demands_linear_fourier_top4(x, y, t, tidx, V, log_transform=True, modtype='B'):
    global tseason
    
    Hfilt = (x<V['xh'])&(y>V['yh']) & ((t<tseason[0])|(t>=tseason[3]))
    Xh = x[Hfilt]; Th = t[Hfilt]; tidxh = tidx[Hfilt];
    l0h = fourier_top4(Th, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lh = linearc(Xh, V['bh'], V['mh'], V['xh']) * fourier_top4(Th, V['ph'], V['bh0'], 
                    V['bh1c'], V['bh1s'], V['bh2c'], V['bh2s'], V['bh3c'], V['bh3s'], 
                    V['bh4c'], V['bh4s'], V['kh2'], V['kh3'], V['kh4']) # heat domain
    
    Cfilt = (x>V['xc']) & (y>V['yc']) & (t>=tseason[1]) & (t<tseason[2])    
    Xc = x[Cfilt]; Tc = t[Cfilt]; tidxc = tidx[Cfilt];
    l0c = fourier_top4(Tc, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'], V['b4c'], V['b4s'],
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lc = linearc(Xc, V['bc'], V['mc'], V['xc']) * fourier_top4(Tc, V['pc'], V['bc0'], 
                    V['bc1c'], V['bc1s'], V['bc2c'], V['bc2s'], V['bc3c'], V['bc3s'], 
                    V['bc4c'], V['bc4s'], V['kc2'], V['kc3'], V['kc4']) # cool domain
    
    if log_transform:
        # need to back-transform
        if modtype == 'A':
            lh = np.exp(lh) - np.exp(l0h)
            lc = np.exp(lc) - np.exp(l0c)
        else:
            lh = np.exp(lh+l0h) - np.exp(l0h)
            lc = np.exp(lc+l0c) - np.exp(l0c)
    else:
        if modtype == 'A':
            lh -= l0h
            lc -= l0c
    return [tidxh, tidxc], [Xh, Xc], [lh, lc]

def linear_fourier_top4_24_12(x, y, t, # data
                    xh, yh, xc, yc, # domain bounds
                    mh, bh, mc, bc, # heat/cool lines
                    b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, # fourier 0 HVAC-off reg
                    bh0, bh1c, bh1s, bh2c, bh2s, # fourier heat
                    bc0, bc1c, bc1s, bc2c, bc2s, # fourier cool reg
                    p, ph, pc, k2, k3, k4, # fourier period and fourier term multiplers
                    modtype='B'):
    global tseason
    
    # domain-wise regressions
    l0 = fourier_top4(t, p, b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, k2, k3, k4) # HVAC-off domain
    lh = linearc(x, bh, mh, xh)*fourier2(t, ph, bh0, bh1c, bh1s, bh2c, bh2s) # heat domain
    lc = linearc(x, bc, mc, xc)*fourier2(t, pc, bc0, bc1c, bc1s, bc2c, bc2s) # cool domain
    #if modtype != 'A':
    lh = lh + l0
    lc = lc + l0
        
    condlist = [(x<xh) & (y>yh) & ((t<tseason[0])|(t>=tseason[3])), 
                (x>xc) & (y>yc) & (t>=tseason[1]) & (t<tseason[2])] # + else
    funclist = [lh, lc, l0]
    return np.where(condlist[0], funclist[0], 
                    np.where(condlist[1], funclist[1], funclist[2]))  

def HVAC_demands_linear_fourier_top4_24_12(x, y, t, tidx, V, log_transform=True, modtype='B'):
    global tseason
    
    Hfilt = (x<V['xh'])&(y>V['yh']) & ((t<tseason[0])|(t>=tseason[3]))
    Xh = x[Hfilt]; Th = t[Hfilt]; tidxh = tidx[Hfilt];
    l0h = fourier_top4(Th, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lh = linearc(Xh, V['bh'], V['mh'], V['xh']) * fourier2(Th, V['ph'], V['bh0'], 
                    V['bh1c'], V['bh1s'], V['bh2c'], V['bh2s']) # heat domain
        
    Cfilt = (x>V['xc']) & (y>V['yc']) & (t>=tseason[1]) & (t<tseason[2])    
    Xc = x[Cfilt]; Tc = t[Cfilt]; tidxc = tidx[Cfilt];
    l0c = fourier_top4(Tc, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'], V['b4c'], V['b4s'],
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    lc = linearc(Xc, V['bc'], V['mc'], V['xc']) * fourier2(Tc, V['pc'], V['bc0'], 
                    V['bc1c'], V['bc1s'], V['bc2c'], V['bc2s']) # cool domain
    
    if log_transform:
        # need to back-transform
        if modtype == 'A':
            lh = np.exp(lh) - np.exp(l0h)
            lc = np.exp(lc) - np.exp(l0c)
        else:
            lh = np.exp(lh+l0h) - np.exp(l0h)
            lc = np.exp(lc+l0c) - np.exp(l0c)
    else:
        if modtype == 'A':
            lh -= l0h
            lc -= l0c
    return [tidxh, tidxc], [Xh, Xc], [lh, lc]

def Baseload_linear_fourier_top4(x, y, t, tidx, V, log_transform):

    
    l0 = fourier_top4(t, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['k2'], V['k3'], V['k4']) # HVAC-off domain
    
    if log_transform:
       l0 = np.exp(l0)
       
    return l0

def linear_fourier_top8_24_12(x, y, t, # data
                    xh, yh, xc, yc, # domain bounds
                    mh, bh, mc, bc, # heat/cool lines
                    b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s,  # fourier 0 HVAC-off reg
                    b5c, b5s, b6c, b6s, b7c, b7s, b8c, b8s,
                    bh0, bh1c, bh1s, bh2c, bh2s, # fourier heat
                    bc0, bc1c, bc1s, bc2c, bc2s, # fourier cool reg
                    p, ph, pc, k2, k3, k4, k5, k6, k7, k8, # fourier period and fourier term multiplers
                    modtype='B'):
    global tseason
    
    # domain-wise regressions
    l0 = fourier_top8(t, p, b0, b1c, b1s, b2c, b2s, b3c, b3s, b4c, b4s, 
                      b5c, b5s, b6c, b6s, b7c, b7s, b8c, b8s,
                      k2, k3, k4, k5, k6, k7, k8) # HVAC-off domain
    lh = linearc(x, bh, mh, xh)*fourier2(t, ph, bh0, bh1c, bh1s, bh2c, bh2s) # heat domain
    lc = linearc(x, bc, mc, xc)*fourier2(t, pc, bc0, bc1c, bc1s, bc2c, bc2s) # cool domain
    if modtype != 'A':
        lh = lh + l0
        lc = lc + l0
        
    condlist = [(x<xh) & (y>yh) & ((t<tseason[0])|(t>=tseason[3])), 
                (x>xc) & (y>yc) & (t>=tseason[1]) & (t<tseason[2])] # + else
    funclist = [lh, lc, l0]
    return np.where(condlist[0], funclist[0], 
                    np.where(condlist[1], funclist[1], funclist[2]))  

def HVAC_demands_linear_fourier_top8_24_12(x, y, t, tidx, V, log_transform, modtype='B'):
    global tseason
    
    Hfilt = (x<V['xh'])&(y>V['yh']) & ((t<tseason[0])|(t>=tseason[3]))
    Xh = x[Hfilt]; Th = t[Hfilt]; tidxh = tidx[Hfilt];
    l0h = fourier_top8(Th, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['b5c'], V['b5s'], V['b6c'], V['b6s'], 
                       V['b7c'], V['b7s'],V['b8c'], V['b8s'], 
                       V['k2'], V['k3'], V['k4'],V['k5'], V['k6'], V['k7'], V['k8']) # HVAC-off domain
    lh = linearc(Xh, V['bh'], V['mh'], V['xh']) * fourier2(Th, V['ph'], V['bh0'], 
                    V['bh1c'], V['bh1s'], V['bh2c'], V['bh2s']) # heat domain
    
    Cfilt = (x>V['xc']) & (y>V['yc']) & (t>=tseason[1]) & (t<tseason[2])    
    Xc = x[Cfilt]; Tc = t[Cfilt]; tidxc = tidx[Cfilt];
    l0c = fourier_top8(Tc, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['b5c'], V['b5s'], V['b6c'], V['b6s'], 
                       V['b7c'], V['b7s'],V['b8c'], V['b8s'], 
                       V['k2'], V['k3'], V['k4'],V['k5'], V['k6'], V['k7'], V['k8']) # HVAC-off domain
    lc = linearc(Xc, V['bc'], V['mc'], V['xc']) * fourier2(Tc, V['pc'], V['bc0'], 
                    V['bc1c'], V['bc1s'], V['bc2c'], V['bc2s']) # cool domain
    
    if log_transform:
        # need to back-transform
        if modtype == 'A':
            lh = np.exp(lh) - np.exp(l0h)
            lc = np.exp(lc) - np.exp(l0c)
        else:
            lh = np.exp(lh+l0h) - np.exp(l0h)
            lc = np.exp(lc+l0c) - np.exp(l0c)
    else:
        if modtype == 'A':
            lh -= l0h
            lc -= l0c
    return [tidxh, tidxc], [Xh, Xc], [lh, lc]

def Baseload_linear_fourier_top8(x, y, t, tidx, V, log_transform):

    
    l0 = fourier_top8(t, V['p'], V['b0'], V['b1c'], V['b1s'], V['b2c'], V['b2s'], 
                       V['b3c'], V['b3s'],V['b4c'], V['b4s'], 
                       V['b5c'], V['b5s'], V['b6c'], V['b6s'], 
                       V['b7c'], V['b7s'],V['b8c'], V['b8s'], 
                       V['k2'], V['k3'], V['k4'],V['k5'], V['k6'], V['k7'], V['k8']) # HVAC-off domain
    
    if log_transform:
       l0 = np.exp(l0)
       
    return l0

print(' "linear_fourier models" and "get HVAC load models" (with time constraints) ran.')

# %% [2] get DR from results
def get_disagg_load_kW(zc, service_type, LB_pdata=None, LB_r2=None, fourier_type='42', log_transform=True):
    global data_type
    """
    LB_pdata: lower bound for zip codes to be included in aggregation based on their 'pdata', [0,1]
    LB_r2: lower bound for zip codes to be included in aggregation based on their 'r2', [0,1]
    """
    
    modtype = 'B'
    log = 'log' if log_transform else ''
    outdir_sub = os.path.join(outdir, 'Model{}_{}_{}_{}'.format(fourier_type,log,service_type,data_type))
   
    Output = load_pickle(os.path.join(outdir_sub,'Model{}{}_{}_{}'.format(fourier_type, '_log' if log_transform else '',
                                                                              service_type, zc)))
    data = Output['data']
    xx, yy, t, tidx, best_fit = data['temp'], data['demand'], data['time_index'], data['time'], data['best_fit']
    tidx = np.array(tidx)
    it = Output['opt']; V = Output['df_val'].iloc[it]
    
    if fourier_type == '44':
        Thc, Xhc, Yhc = HVAC_demands_linear_fourier_top4(xx, yy, t, tidx, V, log_transform, modtype) # [xh, xc], [yh, yc]
    elif fourier_type == '42':
        Thc, Xhc, Yhc = HVAC_demands_linear_fourier_top4_24_12(xx, yy, t, tidx, V, log_transform, modtype)
    elif fourier_type == '82':
        Thc, Xhc, Yhc = HVAC_demands_linear_fourier_top8_24_12(xx, yy, t, tidx, V, log_transform, modtype)
    
    if fourier_type in ['44','42']:
        BL = Baseload_linear_fourier_top4(xx, yy, t, tidx, V, log_transform)
    elif fourier_type == '82':
        BL = Baseload_linear_fourier_top8(xx, yy, t, tidx, V, log_transform)
    
    for i in range(len(Yhc)):
        ### retain only positive DR
        Thc[i] = Thc[i][Yhc[i]>0]; Xhc[i] = Xhc[i][Yhc[i]>0]; Yhc[i] = Yhc[i][Yhc[i]>0]
    
        ### apply filters
        if LB_pdata is not None:
            if V['pdata'] < LB_pdata:
                Yhc[i] = np.zeros_like(Xhc[i])

        if LB_r2 is not None:
            if V['r2'] < LB_r2:
                Yhc[i] = np.zeros_like(Xhc[i])
                
    ### put into df
    ttheat = pd.DataFrame(Xhc[0], index=Thc[0], columns=[zc]); ttheat.index.name='time' # time temp
    ttcool = pd.DataFrame(Xhc[1], index=Thc[1], columns=[zc]); ttcool.index.name='time'
    
    tdheat = pd.DataFrame(Yhc[0], index=Thc[0], columns=[zc]).divide(1000); tdheat.index.name='time' # time DR kW
    tdcool = pd.DataFrame(Yhc[1], index=Thc[1], columns=[zc]).divide(1000); tdcool.index.name='time'
    
    baseload = pd.DataFrame(BL, index=tidx, columns=[zc]).divide(1000); baseload.index.name='time' # time baseload kW
    
    return ttheat, ttcool, tdheat, tdcool, baseload

    
def get_results(zc, service_type, fourier_type='42', log_transform=True):
    global data_type
    """
    LB_pdata: lower bound for zip codes to be included in aggregation based on their 'pdata', [0,1]
    LB_r2: lower bound for zip codes to be included in aggregation based on their 'r2', [0,1]
    """
    
    modtype = 'B'
    log = 'log' if log_transform else ''
    outdir_sub = os.path.join(outdir, 'Model{}_{}_{}_{}'.format(fourier_type,log,service_type,data_type))
   
    Output = load_pickle(os.path.join(outdir_sub,'Model{}{}_{}_{}'.format(fourier_type, '_log' if log_transform else '',
                                                                              service_type, zc)))
    #### Ouput structure #####
    # Output = dict({'zipcode': zc, # single val
    #                    'data': {'temp': xx, 'demand': yy, 'time_index': t, 'time': tidx, 'best_fit': res.best_fit}, # list of data
    #                    'df_val': DFV, # df of best val and stats from all iterations
    #                    #'res': Results, # list of lmfit res output
    #                    'opt': Opt, # single val, opt iter
    #                    'opt_DR': Opt_posDR # single val, opt iter out of all non neg DR
    #                    })
    
    it = Output['opt']; V = Output['df_val'].iloc[it] # optimal iter only
    df_val = Output['df_val']
    df_val['type'] = service_type
    df_val['zip_code'] = zc
    
    return df_val

print('"get DR kWh and results" func ran')

def diurnal(df):
    """
    df :  single col dataframe

    """
    df = df.groupby(df.index.hour).agg(['mean','std'])
    df.index.name = 'hour of day'
    return  df

def diurnal_box(df):
    df.index = df.index.hour
    df.index.name = 'hour of day'
    df = df.reset_index().pivot(columns='hour of day', values=df.name)
    return  df

def get_weighted_mean_and_std(dfx, dfw):
    N0w = len(dfw.dropna()) # no. non-zero weights
    AD_mean = (dfx*dfw).sum()/dfw.sum() # weighted mean
    AD_std = np.sqrt((dfw*((dfx-AD_mean)**2)).sum() / ((N0w-1)/N0w*dfw.sum())) # weighted std
    return AD_mean, AD_std

def get_DR(DRC, DR_type):
    global heat1e, cools, coole, heat2s
    
    DR = DRC.copy()
    # make heating available only in winter and cooling in summer
    DR.loc[heat1e:cools-pd.Timedelta(0.5,'h'), :] = np.NAN # shoulder 1
    DR.loc[coole:heat2s-pd.Timedelta(0.5,'h'), :] = np.NAN # shoulder 2
    if DR_type == 'cooling':
        DR.loc[:heat1e-pd.Timedelta(0.5,'h'), :] = np.NAN # winter 1 - no cool
        DR.loc[heat2s:, :] = np.NAN # winter 2 - no cool
    elif DR_type == 'heating':
        DR.loc[cools:coole-pd.Timedelta(0.5,'h'), :] = np.NAN # summer - no heat

    return DR

def get_annual_DR(DRC, DR_type):
    DR = get_DR(DRC, DR_type)
    annual_DR = DR.sum(axis=0)/2 # half hourly kW to kWh
    return annual_DR

def get_annual_percent_DR(df_demand, DRC, DR_type):
    
    # get annual qty
    dfD = df_demand.mean(axis=0)*8784 # [kWh]
    dfDR = get_annual_DR(DRC, DR_type)
    percent_DR = dfDR/dfD # percent DR
    
    return percent_DR

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

# %% [3] run panel for Average premise DR

###### agg DR by service type ######
fourier_type = '82' # <---------- # '44' or '42' or '82'
log_transform = False # <------- # if modeling data as log(data) in [W]
LB_pdata = 95 # <------ lower bound for pdata: 0~100
LB_r2 = None # <------- lower bound for r2: 0~100
    
fig_ext = 'png' # <-------

log_ext = '_log' if log_transform else ''
ylog_msg = 'log ' if log_transform else ''
pdata_ext = '_al{}pct'.format(LB_pdata) if LB_pdata is not None else ''
r2_ext = '_al{}ptR2'.format(LB_r2) if LB_r2 is not None else ''
LB_pdata = LB_pdata/100 if LB_pdata is not None else LB_pdata
LB_r2 = LB_r2/100 if LB_r2 is not None else LB_r2

type_descr = {'C23': 'SF-NE',
              'C24': 'MF-NE',
              'C25': 'SF-E',
              'C26': 'MF-E'}
long_type_descr = {'C23': 'Single family non-electric',
                   'C24': 'Multi-family non-electric',
                   'C25': 'Single family electric',
                   'C26': 'Multi-family electric'}

outdir_dr = os.path.join(outdir, 'Model{}{}_DR'.format(fourier_type, '_log' if log_transform else ''))
if not os.path.exists(outdir_dr):
        os.mkdir(outdir_dr)

### seasons by months
# heat1e = pd.Timestamp('2016-05-01') #
# cools = pd.Timestamp('2016-06-01') #
# coole = pd.Timestamp('2016-09-01') #
# heat2s = pd.Timestamp('2016-10-01') #

### seasons by rupture
heat1e = pd.Timestamp('2016-04-11') 
cools = pd.Timestamp('2016-05-26') 
coole = pd.Timestamp('2016-09-28') 
heat2s = pd.Timestamp('2016-11-17') 

# tseason
tseason = np.array([heat1e, cools, coole, heat2s]) - np.array([pd.Timestamp('2015-12-01')])
tseason = np.array([i.days*24 + i.seconds/3600 for i in tseason])

ComEd = pd.read_csv(os.path.join(outdir,'ComEd System Loads','hrl_load_estimated.csv'),
                                 parse_dates=['datetime_beginning_utc']).rename({
                                     'datetime_beginning_utc':'time',
                                     'estimated_load_hourly':'system load'}, 
                                     axis=1)[['time','system load']]

# system load
ComEd['time'] = ComEd['time']-DateOffset(hours=6); ComEd.index.freq = 'h'
ComEd.set_index(['time'], inplace=True)
ComEd = ComEd/1e3 # MW to GW

### system load duration by season
ComEd3H = get_DR(ComEd, 'heating').sort_values(by='system load', ascending=False)
ComEd3C = get_DR(ComEd, 'cooling').sort_values(by='system load', ascending=False)


print('>>> output directory: {}'.format(outdir_dr))

ADR = {}; # ADRC = {}, {}; # for storing zipcode DR timeseries
AnnDRH, AnnDRC = {}, {}; # for storing annual zipcode heat/cool load
PDRH, PDRC = {}, {}; # for storing annual zipcode heat/cool load percents
SPDRH, SPDRC = {}, {}; # for storing seasonal zipcode heat/cool percents

PeakDRH, PeakDRC = {}, {}; # for storing avg DR for top 5 seasonal peaks
SPeakDRH, SPeakDRC = {}, {}; # for storing % DR for top 5 seasonal peaks
NACC = {}; # for storing no. acct per zip code

for counter, typ in enumerate(['C23','C24','C25','C26'],1):
    # need to break 'C23' into two partitions
    service_type = typ
    outdir_sub = folder_set_up(service_type, fourier_type, log_transform)
    
    print('>>> {}. load data: {} in {} [{}W]...'.format(counter, service_type, data_type, ylog_msg))

    ############### (1) get zip code DR from results
    ## get completed list 
    zc_list = list(os.path.splitext(os.path.basename(y))[0][-5:] for y in
        list(filter(lambda x: x.endswith('.pkl'), os.listdir(outdir_sub))))
    zc_list = list(map(int, zc_list))
    #zc_list = zc_list[:5] # <-----


    TTH, TTC, TDH, TDC, BL = [], [], [], [], [] # in kWh (avg per acct)
    for zc in zc_list:
        DR = get_disagg_load_kW(zc=zc, service_type=service_type, LB_pdata=LB_pdata, LB_r2=LB_r2,
                        fourier_type=fourier_type, log_transform=log_transform)
                    
        # add to lists
        TTH.append(DR[0]); TTC.append(DR[1]) # time temp
        TDH.append(DR[2]); TDC.append(DR[3]) # time DR kW
        BL.append(DR[4]) # time baseload kW
    print('     >>> for-loop compute completed!\n')

    # convert to df
    TTH = pd.concat(TTH, axis=1); TTC = pd.concat(TTC, axis=1) # premise level zipcode [time, temp]
    TDH = pd.concat(TDH, axis=1); TDC = pd.concat(TDC, axis=1) # premise level zipcode [time, demand(kW)]
    BL = pd.concat(BL, axis=1);
    
    ##### (2) remove zip codes with total DR = 0 as they are filtered out by LBs
    filtered_zc_list = TDH.columns[(TDH.sum(axis=0)>0) | (TDC.sum(axis=0)>0)]
    TTH = TTH[filtered_zc_list]; TTC = TTC[filtered_zc_list]; # filtered
    TDH = TDH[filtered_zc_list]; TDC = TDC[filtered_zc_list]; # filtered
    BL = BL[filtered_zc_list];
    
    ##### (3) load demand data
    df_temp, df_demand, fft, fftn, ffth, fftc = data_set_up(service_type, in_W=False, log_transform=False)
    df_temp = df_temp[filtered_zc_list]; df_demand = df_demand[filtered_zc_list] # filtered

    ts1 = pd.Timestamp('2015-12-01')
    ts2 = pd.Timestamp('2016-12-01')

    # get number of acct per zip code
    df_count = DF[service_type].query('(time>=@ts1)&(time<@ts2)').set_index(
        ['time','zip_code'])[['count']].unstack(level='zip_code')
    df_count_max = df_count.max().astype('int')
    df_count = (df_count/df_count*df_count_max)['count'] # make all count time series the same within a zip code
    df_count = df_count[filtered_zc_list] # filtered
    Nacc = df_count.max(axis=0)
    Nacc = Nacc.astype('int') # make into integers
    
    atd = DF[service_type].query('(time>=@ts1)&(time<@ts2)&(zip_code in @filtered_zc_list)') # ['mean'] is in kWh
    atd['mean'] = 2*atd['mean'] # convert from kWh for half hourly to kW
    atd.loc[:,'count'] = atd['zip_code'].map(atd.groupby(['zip_code'])['count'].max().astype('int'))
    atd = atd.set_index(['zip_code','time'])
    
    print('     >>> demand data loaded!\n')
    
    ##### (4) consolidate DR, temp, and count and save
    TTD = pd.concat([TDH,TDC,BL], keys=['heating_kW', 'cooling_kW','baseload_kW']) # time temp + time demand (kWh)
    TTD = TTD.rename_axis(columns='zip_code')
    TTD = TTD.unstack(level=0).stack(level='zip_code').reset_index(
        ).sort_values(by=['zip_code','time']).reset_index(drop=True)
    TTD = TTD.set_index(['zip_code','time']).join(atd[['count', 'mean','degC']], how='right').fillna(0)
    TTD = TTD.rename({'mean':'total_kW'}, axis=1)
    TTD.to_parquet(os.path.join(outdir_dr, 'Disagg_load_{}_{}{}{}{}.parquet'.format(
        service_type, fourier_type, log_ext, pdata_ext, r2_ext)))
    
    ADR[service_type] = TTD;
    
    print('filtered zip code list: ', len(filtered_zc_list))
    print('len(TTDH): ', TTD.index.get_level_values(level='zip_code').nunique())
    print('len(atd): ', atd.index.get_level_values(level='zip_code').nunique())
    
    ### turn avg DR into a single avg DR
    # get total demand
    TD = ((df_demand*df_count).sum(axis=1)/df_count.sum(axis=1)).rename('whole-home demand') #[kWh] - weighted per premise
    TotalD = (df_demand*df_count).div(df_count.sum(axis=1), axis=0) #[kWh] - weighted per premise (disagg)
    
    ##### (5) get annual data 
    ### whole-home demand and 95%CI
    Ademand = df_demand.mean(axis=0)*len(df_demand)/2
    AD_mean, AD_std = get_weighted_mean_and_std(Ademand, Nacc)
    Percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    Ademand_Ptile = pd.DataFrame([weighted_quantile(np.array(Ademand), Percentiles, sample_weight=np.array(Nacc))],
                 columns=Percentiles)
    print('---> Avg annual whole-home demand [kWh] for {}: {} +/- {} (95%CI)'.format(
        service_type, round(AD_mean,0), round(AD_std*1.96,0)))
    print('     In percentiles:')
    display(Ademand_Ptile)
    
    print('\n >>> SANITY CHECK...')
    print(' SUM(AVG_TS): ', TD.sum()*Nacc.sum()/2)
    print(' SUM(ANNUAL): ', (Ademand*Nacc).sum())
    print(' % DIFF FROM TOP TO BOTTOM: {}%'.format(
        round((TD.sum()*Nacc.sum()/2-(Ademand*Nacc).sum())/(Ademand*Nacc).sum()*100,2)
    ))
    
    ### annual DR by zip codes
    Annual_DRH = get_annual_DR(TDH, 'heating')
    Annual_DRC = get_annual_DR(TDC, 'cooling')
    
    AnnDRH[service_type] = Annual_DRH; AnnDRC[service_type] = Annual_DRC
    
    ### annual baseload by zip codes
    Annual_BL = BL.sum(axis=0)/2
    # print('     % diff of DR+baseload from ground truth total: \n{}'.format(
    #     round((Annual_DRH + Annual_BL-Ademand)/Ademand*100, 2)))
        
    ### annual percent DR by zip codes
    PctDRH = Annual_DRH/(Annual_DRH + Annual_BL) #Ademand
    PctDRC = Annual_DRC/(Annual_DRC + Annual_BL) #Ademand
    
    PDRH[service_type] = PctDRH;  PDRC[service_type] = PctDRC
    
    ### print annual DR percents and 95%CI
    PctDRH_mean, PctDRH_std = get_weighted_mean_and_std(PctDRH, Nacc)
    HPct_Ptile = pd.DataFrame([weighted_quantile(np.array(PctDRH), Percentiles, sample_weight=np.array(Nacc))],
                 columns=Percentiles)
    
    print('---> Avg % annual heating load for {}: {} +/- {} (95%CI)'.format(
        service_type, round(PctDRH_mean*100,2), round(PctDRH_std*100*1.96,2)))
    print('     In percentiles (%):')
    display(HPct_Ptile*100)
    
    PctDRC_mean, PctDRC_std = get_weighted_mean_and_std(PctDRC, Nacc)
    CPct_Ptile = pd.DataFrame([weighted_quantile(np.array(PctDRC), Percentiles, sample_weight=np.array(Nacc))],
                 columns=Percentiles)
    print('---> Avg % annual cooling load for {}: {} +/- {} (95%CI)'.format(
        service_type, round(PctDRC_mean*100,2), round(PctDRC_std*100*1.96,2)))
    print('     In percentiles (%):')
    display(CPct_Ptile*100)
    print()
    
    ### seasonal percent DR by zip code
    DMH = get_DR(BL, 'heating').replace({0:np.NAN}).dropna(axis=0, how='all') 
    print(' No. hours in winter: ', len(DMH)/2)
    DMH = DMH.mean(axis=0)*len(DMH)/2 # total demand in winter
    DMC = get_DR(BL, 'cooling').replace({0:np.NAN}).dropna(axis=0, how='all') 
    print(' No. hours in summer: ', len(DMC)/2)
    DMC = DMC.mean(axis=0)*len(DMC)/2 # total demand in summer
    
    SPctDRH = Annual_DRH/(DMH + Annual_DRH)
    SPctDRC = Annual_DRC/(DMC + Annual_DRC)
    
    SPDRH[service_type] = SPctDRH;  SPDRC[service_type] = SPctDRC
    
    ### print annual DR percents and 95%CI
    SPctDRH_mean, SPctDRH_std = get_weighted_mean_and_std(SPctDRH, Nacc)
    HSPct_Ptile = pd.DataFrame([weighted_quantile(np.array(SPctDRH), Percentiles, sample_weight=np.array(Nacc))],
                 columns=Percentiles)
    print('---> Avg % winter heating load for {}: {} +/- {} (95%CI)'.format(
        service_type, round(SPctDRH_mean*100,2), round(SPctDRH_std*100*1.96,2)))
    print('     In percentiles (%):')
    display(HSPct_Ptile*100)
    
    SPctDRC_mean, SPctDRC_std = get_weighted_mean_and_std(SPctDRC, Nacc)
    CSPct_Ptile = pd.DataFrame([weighted_quantile(np.array(SPctDRC), Percentiles, sample_weight=np.array(Nacc))],
                 columns=Percentiles)
    print('---> Avg % summer cooling load for {}: {} +/- {} (95%CI)\n'.format(
        service_type, round(SPctDRC_mean*100,2), round(SPctDRC_std*100*1.96,2)))
    print('     In percentiles (%):')
    display(CSPct_Ptile*100)
    
    ### store Nacc
    NACC[service_type] = Nacc
    
    ### top 5 peak DR
    DRH3 = TDH.reindex(ComEd3H.index); TDDH3 = TDH.reindex(ComEd3H.index) + BL.reindex(ComEd3H.index)
    DRC3 = TDC.reindex(ComEd3C.index); TDDC3 = TDC.reindex(ComEd3C.index) + BL.reindex(ComEd3C.index)
    PeakDRH[service_type] = DRH3.head(5).mean(axis=0); 
    PeakDRC[service_type] = DRC3.head(5).mean(axis=0); 
    
    SPeakDRH[service_type] = (DRH3.head(5)/TDDH3.head(5).values).mean(axis=0); 
    SPeakDRC[service_type] = (DRC3.head(5)/TDDC3.head(5).values).mean(axis=0);
    
    DRH3_mean, DRH3_std =  get_weighted_mean_and_std(PeakDRH[service_type], Nacc)
    DRC3_mean, DRC3_std =  get_weighted_mean_and_std(PeakDRC[service_type], Nacc)
    
    SDRH3_mean, SDRH3_std =  get_weighted_mean_and_std(SPeakDRH[service_type], Nacc)
    SDRC3_mean, SDRC3_std =  get_weighted_mean_and_std(SPeakDRC[service_type], Nacc)
    
    print('\nFor top 5 seasonal system peak hours:')
    print('---> Avg total inst. heating DR during winter (kW): {} +/- {}'.format(round(DRH3_mean,3), round(DRH3_std,3)))
    print('     as % daily load: {} +/- {}\n'.format(round(SDRH3_mean*100,2), round(SDRH3_std*1.96,2)))
    
    print('---> Avg total inst. cooling DR during summer (kW): {} +/- {}'.format(round(DRC3_mean,3), round(DRC3_std,3)))
    print('     as % daily load: {} +/- {}\n'.format(round(SDRC3_mean*100,2), round(SDRC3_std*1.96,2)))
    
    
    # get temp
    TT = ((df_temp*df_count).sum(axis=1)/df_count.sum(axis=1)).rename('avg temp') #[degC] - per premise
    TotalT = (df_temp*df_count).div(df_count.sum(axis=1), axis=0) #[degC] - per premise (disagg)
    
    # get service type level avg DR
    NDRH = pd.DataFrame()
    TotalDRH = TDH.copy()
    for zc in TotalDRH.columns:
        TotalDRH[zc] = TotalDRH[zc]*Nacc[zc] 
        NDRH[zc] = df_count[zc] # map col to original data
    
    DRH = (TotalDRH.sum(axis=1)/NDRH.sum(axis=1)).rename('inst heating DR') #[kW] - weighted per premise
    TotalDRH = TotalDRH.div(NDRH.sum(axis=1), axis=0) #[kWh] - per premise (disagg)
    
    NDRC = pd.DataFrame()
    TotalDRC = TDC.copy()
    for zc in TotalDRC.columns:
        TotalDRC[zc] = TotalDRC[zc]*Nacc[zc] 
        NDRC[zc] = df_count[zc] # map col to original data
    
    DRC = (TotalDRC.sum(axis=1)/NDRC.sum(axis=1)).rename('inst cooling DR') #[kW] - weighted per premise
    TotalDRC = TotalDRC.div(NDRC.sum(axis=1), axis=0) #[kWh] - per premise (disagg)
    
    NBL = pd.DataFrame()
    TotalBL = BL.copy()
    for zc in TotalBL.columns:
        TotalBL[zc] = TotalBL[zc]*Nacc[zc] 
        NBL[zc] = df_count[zc] # map col to original data
    
    DBL = (TotalBL.sum(axis=1)/NBL.sum(axis=1)).rename('baseload') #[kW] - weighted per premise
    TotalBL = TotalBL.div(NBL.sum(axis=1), axis=0) #[kWh] - per premise (disagg)
    
    print('     >>> dataframes prepared, plotting now...')
    
    ### plot type (1) - DR disagg by zip code (TS)
    # fig, ax = plt.subplots(figsize=(6,4))
    # TotalD.plot(kind='area', legend=False, title='{}: Avg whole-home demand'.format(type_descr[service_type]), linewidth=1, ax=ax); 
    # ax.set_ylabel('kW'); ax.margins(x=0);
    # fig.savefig(os.path.join(outdir_dr,
    #                             'Total_demand_{}_{}{}_avg_per_premise{}{}.{}'.format(
    #                                 service_type, fourier_type, log_ext, pdata_ext, r2_ext,
    #                                 'png')), dpi=200, bbox_inches='tight')
    
    # fig1, ax1 = plt.subplots(figsize=(6,4))
    # TotalDRH.plot(kind='area', legend=False, title='{}: Avg inst heating demand'.format(type_descr[service_type]), linewidth=1, ax=ax1); 
    # ax1.set_xlim(TD.index[0],TD.index[-1]);
    # ax1.set_ylabel('kW'); ax1.margins(x=0);
    # ax1.xaxis.label.set_visible(False)
    # fig1.savefig(os.path.join(outdir_dr,
    #                             'DHeating_DR_{}_{}{}_avg_per_premise{}{}.{}'.format(
    #                                 service_type, fourier_type, log_ext, pdata_ext, r2_ext,
    #                                 'png')), dpi=200, bbox_inches='tight')

    # fig2, ax2 = plt.subplots(figsize=(6,4))
    # TotalDRC.plot(kind='area', legend=False, title='{}: Avg inst cooling demand'.format(type_descr[service_type]), linewidth=1, ax=ax2); 
    # ax2.set_xlim(TD.index[0],TD.index[-1]);
    # ax2.set_ylabel('kW'); ax2.margins(x=0);
    # ax2.xaxis.label.set_visible(False)
    # fig2.savefig(os.path.join(outdir_dr,
    #                             'DCooling_DR_{}_{}{}_avg_per_premise{}{}.{}'.format(
    #                                 service_type, fourier_type, log_ext, pdata_ext, r2_ext,
    #                                 'png')), dpi=200, bbox_inches='tight')

    ### plot type (2) - DR vs. total demand (TS)
    fig3, (ax3a, ax3b) = plt.subplots(2,1, sharex=True, figsize=(6,4))
    ax3a.plot(TT.index, TT, color='green', linewidth=1, label=TT.name)
    ax3a.set_ylabel('degC'); ax3a.margins(x=0); 
    ax3a.axhline(0, linestyle='--', linewidth=1, color='k')
    ax3a.xaxis.set_major_locator(plt.LinearLocator(5)); ax3a.minorticks_off()
    ax3a.set_title(long_type_descr[service_type])
    
    ax3b.plot(TD.index, TD, color='grey', linewidth=1, label=TD.name)
    pDRH = DRH.sum()/TD.sum()
    DRH.plot.area(alpha=0.5, color='m', label='{} ({}%)'.format('heating demand', round(pDRH*100,2)), linewidth=1, ax=ax3b); 
    ax3b.set_ylabel('kW'); ax3b.margins(x=0); 
    ax3b.autoscale(); ax3b.set_ylim(0,None); 
    ax3b.margins(x=0); ax3b.legend();
    ax3b.xaxis.set_major_locator(plt.LinearLocator(5)); ax3b.minorticks_off()
    ax3b.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d"))
    ax3b.xaxis.label.set_visible(False)
    plt.setp(ax3b.get_xticklabels(), rotation=0, ha="center")
    
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir_dr,
                                'Heating_DR_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    service_type, fourier_type, log_ext, pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')


    fig4, (ax4a, ax4b) = plt.subplots(2,1, sharex=True, figsize=(6,4))
    ax4a.plot(TT.index, TT, color='green', linewidth=1, label=TT.name)
    ax4a.set_ylabel('degC'); ax4a.margins(x=0); 
    ax4a.axhline(0, linestyle='--', linewidth=1, color='k')
    ax4a.xaxis.set_major_locator(plt.LinearLocator(5)); ax4a.minorticks_off()
    ax4a.set_title(long_type_descr[service_type])
    
    ax4b.plot(TD.index, TD, color='grey', linewidth=1, label=TD.name)
    pDRC = DRC.sum()/TD.sum()
    DRC.plot.area(alpha=0.5, color='c', label='{} ({}%)'.format('cooling demand', round(pDRC*100,2)), linewidth=1, ax=ax4b); 
    ax4b.set_ylabel('kW'); ax4b.margins(x=0); 
    ax4b.autoscale(); ax4b.set_ylim(0,None); 
    ax4b.margins(x=0); ax4b.legend();
    ax4b.xaxis.set_major_locator(plt.LinearLocator(5)); ax4b.minorticks_off()
    ax4b.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d"))
    ax4b.xaxis.label.set_visible(False)
    plt.setp(ax4b.get_xticklabels(), rotation=0, ha="center")
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(outdir_dr,
                                'Cooling_DR_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    service_type, fourier_type, log_ext, pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')
    
    ############## (1A) combining heating and cooling DEMAND into specified freq ###########
    freq = 'd' # <--------- 'hh', 'h', 'd', 'm'
    freq_descr = {'hh': 'half-hourly',
                  'h': 'hourly',
                  'd': 'daily',
                  'm': 'monthly'}
    
    fig6, (ax6a, ax6b) = plt.subplots(2,1, sharex=True, figsize=(6,4))
    if freq != 'hh':
        TTd = TT.resample(freq).mean()
    TTd.plot(color='green', linewidth=1, ax=ax6a)
    ax6a.axhline(0, linestyle='--', linewidth=1, color='k'); ax6a.set_ylabel('degC');
    ax6a.margins(x=0); ax6a.minorticks_off()
    #ax6a.xaxis.set_major_locator(plt.LinearLocator(5)); 
    ax6a.set_title('{} ({})'.format(long_type_descr[service_type], freq_descr[freq]))
    
    DR = pd.concat([DRH, DRC, DBL], axis=1)
    DR = DR.rename({'inst heating DR': 'heating load', 'inst cooling DR': 'cooling load'}, axis=1)
    #DR['baseload'] = TD-DR.sum(axis=1)
    DR = DR[['baseload','heating load','cooling load']]
    
    if freq != 'hh':
        DR = DR.resample(freq).sum()/2 # half hourly kW -> kWh
        td = TD.resample(freq).sum()/2 # half hourly kW -> kWh
        ylab = 'kWh'
    else:
        ylab = 'kW'
        
    for col in DR.columns:
        pDR = DR[col].sum()/td.sum()
        DR = DR.rename({col:'{} ({}%)'.format(col, round(pDR*100,2))}, axis=1)
    
    DR.cumsum(axis=1).iloc[:,2].plot.area(color='c', linewidth=1, ax=ax6b)
    DR.cumsum(axis=1).iloc[:,1].plot.area(color='m', linewidth=1, ax=ax6b)   
    DR.cumsum(axis=1).iloc[:,0].plot.area(color='tab:blue', linewidth=1, ax=ax6b)   
    
    # ax6b.fill_between(DR.index, np.zeros_like(DR.iloc[:,0]), DR.cumsum(axis=1).iloc[:,2], color='c') 
    # ax6b.fill_between(DR.index, np.zeros_like(DR.iloc[:,0]), DR.cumsum(axis=1).iloc[:,1], color='m') 
    # ax6b.fill_between(DR.index, np.zeros_like(DR.iloc[:,0]), DR.iloc[:,0], color='tab:blue')
     
    td.plot(color='dimgrey', linewidth=1, label=td.name, ax=ax6b)
    # DR.plot.area(stacked=True, color=['tab:blue','m','c'], linewidth=1, ax=ax6b, rot=0); 
    ax6b.autoscale(); 
    ax6b.set_ylim(0,None); ax6b.set_ylabel(ylab);
    ax6b.margins(x=0);  ax6b.legend(prop={'size': 7}); ax6b.minorticks_off()
    #ax6b.xaxis.set_major_locator(plt.LinearLocator(5)); 
    #ax6b.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d"))
    ax6b.xaxis.label.set_visible(False)
    plt.setp(ax6b.get_xticklabels(), rotation=0, ha="center")
    fig6.tight_layout()
    fig6.savefig(os.path.join(outdir_dr,
                                'Heat_cool_demand_{}_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    freq_descr[freq], service_type, fourier_type, log_ext, pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')
    
    ############## (1B) combining heating and cooling DR into specified freq ###########
    freq = 'd' # <--------- 'hh', 'h', 'd', 'm'
    freq_descr = {'hh': 'half-hourly',
                  'h': 'hourly',
                  'd': 'daily',
                  'm': 'monthly'}
    
    fig6, (ax6a, ax6b) = plt.subplots(2,1, sharex=True, figsize=(6,4))
    if freq != 'hh':
        TTd = TT.resample(freq).mean()
    ax6a.plot(TTd.index, TTd, color='green', linewidth=1, label=TT.name)
    ax6a.axhline(0, linestyle='--', linewidth=1, color='k'); ax6a.set_ylabel('degC');
    ax6a.margins(x=0);
    ax6a.xaxis.set_major_locator(plt.LinearLocator(5)); ax6a.minorticks_off()
    ax6a.set_title('{} ({})'.format(long_type_descr[service_type], freq_descr[freq]))
    
    DR = pd.concat([DRH, DRC], axis=1)
    # make heating available only in winter and cooling in summer
    DR.loc[heat1e:cools-pd.Timedelta(0.5,'h'), :] = np.NAN # shoulder 1
    DR.loc[coole:heat2s-pd.Timedelta(0.5,'h'), :] = np.NAN # shoulder 2
    DR.loc[:heat1e-pd.Timedelta(0.5,'h'), 'inst cooling DR'] = np.NAN # winter 1
    DR.loc[heat2s:, 'inst cooling DR'] = np.NAN # winter 2
    DR.loc[cools:coole-pd.Timedelta(0.5,'h'), 'inst heating DR'] = np.NAN # summer
    
    
    if freq != 'hh':
        DR = DR.resample(freq).mean() # kW
        td = TD.resample(freq).mean() # kW
        ylab = 'kW'
        
    for col in DR.columns:
        pDR = DR[col].sum()/td.sum()
        DR = DR.rename({col:'{} ({}%)'.format(col, round(pDR*100,2))}, axis=1)
        
    ax6b.plot(td.index, td, color='grey', linewidth=1, label=td.name)
    DR.plot.area(stacked=False, color=['m','c'], linewidth=1, ax=ax6b, rot=0); 
    ax6b.autoscale(); ax6b.set_ylim(0,None); ax6b.set_ylabel(ylab);
    ax6b.margins(x=0);  ax6b.legend();
    ax6b.xaxis.set_major_locator(plt.LinearLocator(5)); ax6b.minorticks_off()
    ax6b.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d"))
    ax6b.xaxis.label.set_visible(False)
    plt.setp(ax6b.get_xticklabels(), rotation=0, ha="center")
    fig6.tight_layout()
    fig6.savefig(os.path.join(outdir_dr,
                                'Heat_cool_DR_{}_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    freq_descr[freq], service_type, fourier_type, log_ext, pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')
    
    ###################### diurnal profiles ######################
    
    winter = (TD.index<heat1e)|(TD.index>=heat2s)
    summer = (TD.index>=cools) & (TD.index<coole)
    shoulder = ((TD.index>=heat1e)&(TD.index<cools)) | ((TD.index>=coole)&(TD.index<heat2s))
    
    TTs = {}
    TTs['winter'] = diurnal(TT[winter])
    TTs['summer'] = diurnal(TT[summer])

    TDs = {}
    TDs['winter'] = diurnal(TD[winter])/2 # half hourly kW -> kWh
    TDs['summer'] = diurnal(TD[summer])/2 # half hourly kW -> kWh

    DRs = {}; BLs = {};
    DR = pd.concat([DRH, DRC], axis=1)
    BLl = pd.concat([DBL, DBL], axis=1)
    BLl.columns = ['inst heating DR', 'inst cooling DR']
    # BLl = pd.concat([TD-DRH, TD-DRC], axis=1).rename({
    #     0:'inst heating DR', 1:'inst cooling DR'}, axis=1)
    DRs['winter'] = diurnal(DR[winter])/2 # half hourly kW -> kWh
    DRs['summer'] = diurnal(DR[summer])/2 # half hourly kW -> kWh
    BLs['winter'] = diurnal(BLl[winter])/2 # half hourly kW -> kWh
    BLs['summer'] = diurnal(BLl[summer])/2 # half hourly kW -> kWh
    
    # for season in ['winter', 'summer']:
    #     for col in DRs[season].columns:
    #         pDR = DRs[season][col].sum()/TDs[season].sum()
    #         DRs[season] = DRs[season].rename({col:'{} ({}%)'.format(col, round(pDR*100,2))}, axis=1)
    #         BLs[season] = BLs[season].rename({col:'{} ({}%)'.format(col, round((1-pDR)*100,2))}, axis=1)
        
    
    fig7, ([ax71,ax72], [ax73,ax74], [ax75,ax76]) = plt.subplots(3,2, sharex=True, sharey='row', figsize=(5,5))
    
    ### set (1): temp
    season = 'winter'; ax = ax71
    idx = TTs[season].index; 
    lb = TTs[season]['mean']-1.96*TTs[season]['std']; #95CI
    ub = TTs[season]['mean']+1.96*TTs[season]['std']
    ax.fill_between(idx, lb, ub, color='green', alpha=0.25)
    TTs[season]['mean'].plot(color='green', linewidth=2, ax=ax)
    ax.axhline(0, linestyle='--', linewidth=1, color='k')
    ax.set_ylabel('degC'); ax.margins(x=0); ax.xaxis.set_major_locator(plt.MultipleLocator(6)); ax.minorticks_off()
    ax.set_title(season)
    
    season = 'summer'; ax = ax72
    idx = TTs[season].index; 
    lb = TTs[season]['mean']-1.96*TTs[season]['std']; 
    ub = TTs[season]['mean']+1.96*TTs[season]['std']
    ax.fill_between(idx, lb, ub, color='green', alpha=0.25)
    TTs[season]['mean'].plot(color='green', linewidth=2, ax=ax)
    ax.axhline(0, linestyle='--', linewidth=1, color='k')
    ax.set_ylabel('degC'); ax.margins(x=0); ax.xaxis.set_major_locator(plt.MultipleLocator(6)); ax.minorticks_off()
    ax.set_title(season)
    
    ### set (2): stacked
    season = 'winter'; DRtype = 'inst heating DR'; ax = ax73 # show heating only in winter
    pDR =  DRs[season][DRtype]['mean'].sum()/TDs[season]['mean'].sum()
    (DRs[season]+BLs[season])[DRtype]['mean'].plot.area(color='m', linewidth=2, ax=ax, label='avg heating load')
    BLs[season][DRtype]['mean'].plot.area(color='tab:blue', linewidth=2, ax=ax, label='avg baseload')
    TDs[season]['mean'].plot(color='grey', linewidth=1, ax=ax, linestyle='dashdot',  label='whole-home demand')
    ax.autoscale(); ax.set_ylim(0,None);  ax.legend(loc=2, prop={'size': 6})
    ax.set_ylabel('kW'); ax.margins(x=0); ax.xaxis.set_major_locator(plt.MultipleLocator(3));
    print('... check heating demand in winter (avg across time) : {}%'.format(round(pDR*100,2)))
    
    season = 'summer'; DRtype = 'inst cooling DR'; ax = ax74 # show heating only in winter
    pDR =  DRs[season][DRtype]['mean'].sum()/TDs[season]['mean'].sum()
    (DRs[season]+BLs[season])[DRtype]['mean'].plot.area(color='c', linewidth=2, ax=ax, label='avg cooling load')
    BLs[season][DRtype]['mean'].plot.area(color='tab:blue', linewidth=2, ax=ax, label='avg baseload')
    TDs[season]['mean'].plot(color='grey', linewidth=1, ax=ax, linestyle='dashdot', label='whole-home demand')
    ax.autoscale(); ax.set_ylim(0,None); ax.legend(loc=2, prop={'size': 6})
    ax.set_ylabel('kW'); ax.margins(x=0); ax.xaxis.set_major_locator(plt.MultipleLocator(3));
    print('... check: cooling demand in summer (avg across time) : {}%'.format(round(pDR*100,2)))
    
    ### set (3): line w/ 95CI
    season = 'winter'; DRtype = 'inst heating DR'; ax = ax75 # show heating only in winter
    iidx = TDs[season].index; 
    lb = TDs[season]['mean']-1.96*TDs[season]['std']; 
    ub = TDs[season]['mean']+1.96*TDs[season]['std']
    ax.fill_between(idx, lb, ub, color='grey', alpha=0.25)
    TDs[season]['mean'].plot(color='grey', linewidth=2, ax=ax, label='whole-home demand')
    idx = DRs[season].index; 
    lb = DRs[season][DRtype]['mean']-1.96*DRs[season][DRtype]['std']; 
    ub = DRs[season][DRtype]['mean']+1.96*DRs[season][DRtype]['std']
    ax.fill_between(idx, lb, ub, color='m', alpha=0.5)
    pDR =  DRs[season][DRtype]['mean'].sum()/TDs[season]['mean'].sum()
    DRs[season][DRtype]['mean'].plot(color='m', linewidth=2, ax=ax, label='heating load')
    ax.autoscale(); ax.set_ylim(0,None); ax.legend(loc=2, prop={'size': 6})#loc='upper center', bbox_to_anchor=(0.5, -0.275)); 
    ax.minorticks_off()
    ax.set_ylabel('kW'); ax.margins(x=0); ax.xaxis.set_major_locator(plt.MultipleLocator(3));
    
    season = 'summer';  DRtype = 'inst cooling DR'; ax = ax76 # show cooling only in summer
    idx = TDs[season].index; 
    lb = TDs[season]['mean']-1.96*TDs[season]['std']; 
    ub = TDs[season]['mean']+1.96*TDs[season]['std']
    ax.fill_between(idx, lb, ub, color='grey', alpha=0.25)
    TDs[season]['mean'].plot(color='grey', linewidth=2, ax=ax, label='whole-home demand')
    idx = DRs[season].index; 
    lb = DRs[season][DRtype]['mean']-1.96*DRs[season][DRtype]['std']; 
    ub = DRs[season][DRtype]['mean']+1.96*DRs[season][DRtype]['std']
    ax.fill_between(idx, lb, ub, color='c', alpha=0.5)
    pDR =  DRs[season][DRtype]['mean'].sum()/TDs[season]['mean'].sum()
    DRs[season][DRtype]['mean'].plot(color='c', linewidth=2, ax=ax, label='cooling load')
    ax.autoscale(); ax.set_ylim(0,None); ax.legend(loc=2, prop={'size': 6})#loc='upper center', bbox_to_anchor=(0.5, -0.275))
    ax.minorticks_off()
    ax.set_ylabel('kW'); ax.margins(x=0); ax.xaxis.set_major_locator(plt.MultipleLocator(3));
    
    ax74.set_ylim(ax76.get_ylim()) # make same y axis
    
    fig7.suptitle(long_type_descr[service_type], y=1.015)
    fig7.tight_layout()
    fig7.savefig(os.path.join(outdir_dr,
                                'Diurnal_heat_cool_DR_seasons_95ci_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    service_type, fourier_type, log_ext, pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')
    
    
    ##### alternative boxplot diurnal  profiles #####
    # TTs = {}
    # TTs['winter'] = diurnal_box(TT[winter])
    # TTs['summer'] = diurnal_box(TT[summer])
    # TDs = {}
    # TDs['winter'] = diurnal_box(TD[winter])
    # TDs['summer'] = diurnal_box(TD[summer]) 
    # DRs = {}
    # DRs['winter'] = diurnal_box(DRH[winter]) # heating only in winter
    # DRs['summer'] = diurnal_box(DRC[summer]) # cooling only in summer
    
    # fig5, ([ax51,ax52], [ax53,ax54]) = plt.subplots(2,2, sharex=True, sharey='row', figsize=(7,5))
    
    # season = 'winter'; ax = ax51
    # TTs[season].boxplot(color='green', grid=False, ax=ax)
    # ax.axhline(0, linestyle='--', linewidth=1, color='k')
    # ax.set_ylabel('degC'); ax.margins(x=0); #ax.xaxis.set_major_locator(plt.MultipleLocator(6)); ax.minorticks_off()
    # ax.set_title(season)
    
    # season = 'summer'; ax = ax52
    # TTs[season].boxplot(color='green', grid=False, ax=ax)
    # ax.axhline(0, linestyle='--', linewidth=1, color='k')
    # ax.set_ylabel('degC'); ax.margins(x=0); #ax.xaxis.set_major_locator(plt.MultipleLocator(6)); ax.minorticks_off()
    # ax.set_title(season)
    
    # season = 'winter'; ax = ax53
    # bp1 = TDs[season].boxplot(color='grey', grid=False, ax=ax)
    # bp2 = DRs[season].boxplot(color='m', grid=False, ax=ax); 
    # ax.autoscale(); ax.set_ylim(0,None); 
    # pDR =  DRs[season].sum().sum()/2/(TDs[season].sum().sum()/2)
    # leg_total = mpatches.Patch(color='grey', label='whole-home demand')
    # leg_dr = mpatches.Patch(color='m', label='{}'.format(DRtype));#, round(pDR*100,2)))
    # ax.legend(handles=[leg_total, leg_dr], loc='upper center', bbox_to_anchor=(0.5, -0.275)); #ax.minorticks_off()
    # ax.set_ylabel('kW'); ax.margins(x=0); #ax.xaxis.set_major_locator(plt.MultipleLocator(3));
    
    # season = 'summer'; ax = ax54
    # bp1 = TDs[season].boxplot(color='grey', grid=False, ax=ax)
    # bp2 = DRs[season].boxplot(color='c', grid=False, ax=ax); 
    # ax.autoscale(); ax.set_ylim(0,None); 
    # pDR =  DRs[season].sum().sum()/2/(TDs[season].sum().sum()/2)
    # leg_total = mpatches.Patch(color='grey', label='whole-home demand')
    # leg_dr = mpatches.Patch(color='c', label='{}'.format(DRtype));#, round(pDR*100,2)))
    # ax.legend(handles=[leg_total, leg_dr], loc='upper center', bbox_to_anchor=(0.5, -0.275)); #ax.minorticks_off()
    # ax.set_ylabel('kW'); ax.margins(x=0); #ax.xaxis.set_major_locator(plt.MultipleLocator(3));
    
    # xticks = np.arange(1,25,3)
    # labels = np.arange(0,24,3)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(labels)
    
    # fig5.suptitle(long_type_descr[service_type], y=1.015)
    # fig5.tight_layout()
    
    # fig5.savefig(os.path.join(outdir_dr,
    #                             'Heat_cool_DR_seasons_box_{}_{}{}_avg_per_premise{}{}.{}'.format(
    #                                 service_type, fourier_type, log_ext, pdata_ext, r2_ext,
    #                                 fig_ext)), dpi=300, bbox_inches='tight')

    ############################ load duration ###############################
    ## add std ## from different service_class
    DR = pd.concat([DRH, DRC], axis=1)
    # make heating available only in winter and cooling in summer
    DR.loc[heat1e:cools-pd.Timedelta(0.5,'h'), :] = np.NAN # shoulder 1
    DR.loc[coole:heat2s-pd.Timedelta(0.5,'h'), :] = np.NAN # shoulder 2
    DR.loc[:heat1e-pd.Timedelta(0.5,'h'), 'inst cooling DR'] = np.NAN # winter 1
    DR.loc[heat2s:, 'inst cooling DR'] = np.NAN # winter 2
    DR.loc[cools:coole-pd.Timedelta(0.5,'h'), 'inst heating DR'] = np.NAN # summer
    
    DDR = (pd.concat([TD, DR], axis=1).resample('h').sum()/2).sort_values(
        by=['whole-home demand'], ascending=False).reset_index(drop=True).replace({0:np.NAN})
    DDR.index.name = 'No. hrs above'
    
    fig8, ax8 = plt.subplots(figsize=(5,3))
    DDR['whole-home demand'].plot(color='grey', ax=ax8)
    #DDR[['inst heating DR','inst cooling DR']].plot.area(stacked=True, color=['m','c'], ax=ax8)
    ax8.scatter(DDR.index, DDR['inst heating DR'], s=1, color='m', label='inst heating DR')
    ax8.scatter(DDR.index, DDR['inst cooling DR'], s=1, color='c', label='inst cooling DR')
    ax8.set_ylabel('kW'); ax8.margins(x=0); ax8.legend()
    ax8.set_title(long_type_descr[service_type])
    fig8.savefig(os.path.join(outdir_dr,
                                'Load_duration_DR_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    service_type, fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')
    
    print('     >>> plots completed!\n')
    
# %% Aggregate all DR and compare to ComEd system
###### spec for loading data ####
fourier_type = '82' # <---------- # '44' or '42'
log_transform = False # <------- # if modeling data as log(data) in [W]
LB_pdata = 95 # <------ lower bound for pdata: 0~100
LB_r2 = None # <------- lower bound for r2: 0~100
    
load_data_from_file = False # <--------

fig_ext = 'png' # <-------

log_ext = '_log' if log_transform else ''
ylog_msg = 'log ' if log_transform else ''
pdata_ext = '_al{}pct'.format(LB_pdata) if LB_pdata is not None else ''
r2_ext = '_al{}ptR2'.format(LB_r2) if LB_r2 is not None else ''
LB_pdata = LB_pdata/100 if LB_pdata is not None else LB_pdata
LB_r2 = LB_r2/100 if LB_r2 is not None else LB_r2

type_descr = {'C23': 'SF-NE',
              'C24': 'MF-NE',
              'C25': 'SF-E',
              'C26': 'MF-E'}
long_type_descr = {'C23': 'Single family non-electric',
                   'C24': 'Multi-family non-electric',
                   'C25': 'Single family electric',
                   'C26': 'Multi-family electric'}

outdir_dr = os.path.join(outdir, 'Model{}{}_DR'.format(fourier_type, '_log' if log_transform else ''))
if not os.path.exists(outdir_dr):
        os.mkdir(outdir_dr)

### seasons by months
# heat1e = pd.Timestamp('2016-05-01') #
# cools = pd.Timestamp('2016-06-01') #
# coole = pd.Timestamp('2016-09-01') #
# heat2s = pd.Timestamp('2016-10-01') #

### seasons by rupture
heat1e = pd.Timestamp('2016-04-11') 
cools = pd.Timestamp('2016-05-26') 
coole = pd.Timestamp('2016-09-28') 
heat2s = pd.Timestamp('2016-11-17') 

print('>>> output directory: {}'.format(outdir_dr))


if load_data_from_file:
    ADR = {}
    for counter, typ in enumerate(['C26','C25','C24','C23'],1):
        # need to break 'C23' into two partitions
        service_type = typ
        outdir_sub = folder_set_up(service_type, fourier_type, log_transform)
        
        print('>>> {}. load data: {} in {} [{}W]...'.format(counter, service_type, data_type, ylog_msg))
        TTD = pd.read_parquet(os.path.join(outdir_dr, 'Disagg_load_{}_{}{}{}{}.parquet'.format(
            service_type, fourier_type, log_ext, pdata_ext, r2_ext)))
        
        ADR[service_type] = TTD
    
### get aggregated DR by service class ##
N2019 = 3689000 # <------ total no. ComEd res customers 2019 (AMI penetration ~ 100%) EIA form 861 (2019)

Nacc2016 = {}; PH=[]; PC=[]; SPH=[]; SPC=[]; N=[]; ATH=[]; ATC=[]; AVG =[]; AVGH=[]; AVGC = []; 
for counter, typ in enumerate(['C26','C25','C24','C23'],1):
    service_type = typ
    
    Nacc = ADR[service_type]['count'].groupby(['zip_code']).max()
    N.append(Nacc)
    PH.append(PDRH[service_type]); PC.append(PDRC[service_type])
    SPH.append(SPDRH[service_type]); SPC.append(SPDRC[service_type])
    ATH.append(AnnDRH[service_type]); ATC.append(AnnDRC[service_type]) # total DR
    
    TotalDemand = ADR[service_type].groupby(['zip_code','time'])['total_kW'].mean().unstack(level='zip_code')
    AVG.append(TotalDemand.mean()*len(TotalDemand)/2) # annual total by zip code  
    avgH = get_DR(TotalDemand, 'heating').dropna(how='all') # winter loads
    AVGH.append(avgH.mean()*len(avgH)/2)
    avgC = get_DR(TotalDemand, 'cooling').dropna(how='all') # cooling loads
    AVGC.append(avgC.mean()*len(avgC)/2)
    
    Nacc2016[service_type] = Nacc.sum()
    
Nacc2016 = pd.DataFrame(Nacc2016, index=[0])
N2016 = Nacc2016.sum(axis=1).loc[0] # total count
print('>> Service class customer count in time horizon, total: {}:\n{}'.format(N2016, Nacc2016))
print('   % customers:\n', Nacc2016.divide(N2016)*100)

Nmult = N2019/N2016 # multipler to get agg to 2019 customer counts


#### get total agg stats
N = pd.concat(N, axis=0);
# PH = pd.concat(PH, axis=0); PC = pd.concat(PC, axis=0); 
# SPH = pd.concat(SPH, axis=0); SPC = pd.concat(SPC, axis=0); 
ATH = pd.concat(ATH, axis=0); ATC = pd.concat(ATC, axis=0); 

AVG = pd.concat(AVG, axis=0); AVGH = pd.concat(AVGH, axis=0); AVGC = pd.concat(AVGC, axis=0);

# AvggH, StddH = get_weighted_mean_and_std(PH, N)
# print('\n---> Overall % annual load = heating demand: {} +/- {}'.format(round(AvggH*100,2), round(StddH*100, 2)))
# AvggC, StddC = get_weighted_mean_and_std(PC, N)
# print('---> Overall % annual load = cooling demand: {} +/- {}'.format(round(AvggC*100,2), round(StddC*100, 2)))

# AvgH, StdH = get_weighted_mean_and_std(SPH, N)
# print('---> Overall % seasonal load = heating demand: {} +/- {}'.format(round(AvgH*100,2), round(StdH*100, 2)))
# AvgC, StdC = get_weighted_mean_and_std(SPC, N)
# print('---> Overall % seasonal load = cooling demand: {} +/- {}'.format(round(AvgC*100,2), round(StdC*100, 2)))

print('\n(Do not use) Per time horizon (2015/12 ~ 2016/11), agg based on zip code annual total')
Avgg, Stdd = get_weighted_mean_and_std(AVG, N)
print('---> Total loads [kWh]: {} +/- {}'.format(round(Avgg,2), round(Stdd, 2)))
AvgSH, StdSH = get_weighted_mean_and_std(AVGH, N)
print('---> Winter loads [kWh]: {} +/- {}'.format(round(AvgSH,2), round(StdSH, 2)))
AvgSC, StdSC = get_weighted_mean_and_std(AVGC, N)
print('---> Summer loads [kWh]: {} +/- {}\n'.format(round(AvgSC,2), round(StdSC, 2)))

AvgH, StdH = get_weighted_mean_and_std(ATH, N)
print('---> Total heating loads [kWh]: {} +/- {}'.format(round(AvgH,2), round(StdH, 2)))
print('     this is {} % of total or {} % of winter loads'.format(
    round(AvgH/Avgg*100,2), round(AvgH/AvgSH*100,2)))
    
AvgC, StdC = get_weighted_mean_and_std(ATC, N)
print('---> Total cooling loads [kWh]: {} +/- {}'.format(round(AvgC,2), round(StdC, 2)))
print('     this is {} % of total or {} % of summer loads'.format(
    round(AvgC/Avgg*100,2), round(AvgC/AvgSC*100,2)))

# %% ########### Total 2016 AMI agg ################
DRH = []; DRC = []; DBL = []; TDD = []; # agg total
AvgDRH = []; AvgDRC = []; AvgBL = []; AvgTDD = []; # avg permise
Temp = []; N = [];
Nacc2016 = {};

for counter, typ in enumerate(['C26','C25','C24','C23'],1):
    service_type = typ
    
    Nzc = ADR[service_type].index.get_level_values(level='zip_code').nunique()
    ZCdrH = ADR[service_type].groupby('zip_code')['heating_kW'].sum()/2 # total annual DR kWh by zc
    NzcH_nz = len(ZCdrH.replace({0:np.NAN}).dropna())
    ZCdrC = ADR[service_type].groupby('zip_code')['cooling_kW'].sum()/2 # total annual DR kWh by zc
    NzcC_nz = len(ZCdrC.replace({0:np.NAN}).dropna())
    print('\nNumber of zip codes in {}: {}'.format(service_type, Nzc))
    print('# zip codes with no heating DR: {} ({}%)'.format((Nzc - NzcH_nz), round((Nzc-NzcH_nz)/Nzc*100, 2)))
    print('# zip codes with no cooling DR: {} ({}%)'.format((Nzc - NzcC_nz), round((Nzc-NzcC_nz)/Nzc*100, 2)))
    
    ### agg total
    Agg_drH = (ADR[service_type]['heating_kW']*ADR[service_type]['count']).groupby(['time']).sum().rename(
        type_descr[service_type]+' (heating)')
    DRH.append(Agg_drH)
    
    Agg_drC = (ADR[service_type]['cooling_kW']*ADR[service_type]['count']).groupby(['time']).sum().rename(
        type_descr[service_type]+' (cooling)')
    DRC.append(Agg_drC)
    
    Agg_bl = (ADR[service_type]['baseload_kW']*ADR[service_type]['count']).groupby(['time']).mean().rename(
        type_descr[service_type]) # mean
    Agg_bl *= Nzc
    DBL.append(Agg_bl)
    
    Agg_total = (ADR[service_type]['total_kW']*ADR[service_type]['count']).groupby(['time']).mean().rename(
        type_descr[service_type]) # mean
    Agg_total *= Nzc
    TDD.append(Agg_total)
    
    ### weighted avg
    
    Avg_drH = Agg_drH.divide(ADR[service_type]['count'].groupby(['time']).sum().values).rename(
        type_descr[service_type])
    AvgDRH.append(Avg_drH)
    
    Avg_drC = Agg_drC.divide(ADR[service_type]['count'].groupby(['time']).sum().values).rename(
        type_descr[service_type])
    AvgDRC.append(Avg_drC)
    
    Avg_bl = Agg_bl.divide(ADR[service_type]['count'].groupby(['time']).sum().values).rename(
        type_descr[service_type])
    AvgBL.append(Avg_bl)
    
    Avg_total = Agg_total.divide(ADR[service_type]['count'].groupby(['time']).sum().values).rename(
        type_descr[service_type])
    AvgTDD.append(Avg_total)
    
    ### temp
    Agg_temp = (ADR[service_type]['degC']*ADR[service_type]['count']).groupby(['time']).sum().rename(type_descr[service_type])
    Count = ADR[service_type]['count'].groupby(['time']).sum().rename(type_descr[service_type])
    Temp.append(Agg_temp); N.append(Count)
    
    ### no. of accounts
    Nacc = ADR[service_type]['count'].groupby(['zip_code']).max()
    Nacc2016[service_type] = Nacc.sum()

Nacc2016 = pd.DataFrame(Nacc2016, index=[0])
N2016 = Nacc2016.sum(axis=1).loc[0] # total count
print('>> Service class customer count in time horizon, total: {}:\n{}'.format(N2016, Nacc2016))
print('   % customers:\n', Nacc2016.divide(N2016)*100)

N2019 = 3689000 # <------ total no. ComEd res customers 2019 (AMI penetration ~ 100%) EIA form 861 (2019)
Nmult = N2019/N2016 # multipler to get agg to 2019 customer counts

### convert to df (col: service class)       
DRH = pd.concat(DRH, axis=1)*Nmult/1e6 # convert to 2019, kW to GW
DRC = pd.concat(DRC, axis=1)*Nmult/1e6 # convert to 2019, kW to GW
DBL = pd.concat(DBL, axis=1)*Nmult/1e6 # convert to 2019, kW to GW
TDD = pd.concat(TDD, axis=1)*Nmult/1e6 # convert to 2019, kW to GW

AvgDRH = pd.concat(AvgDRH, axis=1)
AvgDRC = pd.concat(AvgDRC, axis=1)
AvgBL = pd.concat(AvgBL, axis=1)
AvgTDD = pd.concat(AvgTDD, axis=1)
Temp = pd.concat(Temp, axis=1) # this is a sum
N = pd.concat(N, axis=1)

## single col df
TempS = Temp.sum(axis=1)/N.sum(axis=1).values # single temp
Temp = Temp/N # temp by service class

### make heating available only in winter and cooling in summer
DRH = get_DR(DRH,'heating')
DRC = get_DR(DRC,'cooling')
DBLH = get_DR(DBL,'heating') # baseload cropped to winter
DBLC = get_DR(DBL,'cooling') # baseload cropped to summer
TDDH = get_DR(TDD,'heating') # total demand cropped to winter
TDDC = get_DR(TDD,'cooling') # total demand cropped to summer

### resample to hourly
DRH = DRH.resample('h').mean()
DRC = DRC.resample('h').mean()
DBLH = DBLH.resample('h').mean()
DBLC = DBLC.resample('h').mean()
DBL = DBL.resample('h').mean()
TDDH = TDDH.resample('h').mean()
TDDC = TDDC.resample('h').mean()
TDD = TDD.resample('h').mean()

print('\n>> Total HVAC demand [GWh] based on 2019 customer count = {}'.format(N2019))
print('   Total heating: {} ({}% total load, {}% winter load)'.format(
    round(DRH.sum().sum(),2), round(DRH.sum().sum()/(DRH.sum().sum()+DBL.sum().sum())*100,2), 
    round(DRH.sum().sum()/(DRH.sum().sum()+DBLH.sum().sum())*100,2) ))
display(pd.concat([DRH.sum(axis=0).rename('GWh'), 
                   (DRH.sum()/(DRH.sum()+DBL.sum().values).values*100).rename('% annual'),
                   (DRH.sum()/(DRH.sum()+DBLH.sum().values).values*100).rename('% winter')], axis=1))
print('   Total cooling: {} ({}% total load, {}% summer load)'.format(
    round(DRC.sum().sum(),2), round(DRC.sum().sum()/(DRC.sum().sum()+DBL.sum().sum())*100,2), 
    round(DRC.sum().sum()/(DRC.sum().sum()+DBLC.sum().sum())*100,2) ))
display(pd.concat([DRC.sum(axis=0).rename('GWh'), 
                   (DRC.sum()/(DRC.sum()+DBL.sum().values).values*100).rename('% annual'),
                   (DRC.sum()/(DRC.sum()+DBLC.sum().values).values*100).rename('% summer')], axis=1))

# combine
ADRHC = pd.concat([DRH, DRC], axis=1)
colors = ['darkred','red','darkmagenta','m', 'royalblue','skyblue','darkcyan','c']
ADRHC.plot.area(stacked=True, color=colors,alpha=1)
plt.ylabel('GW')

##### load ComEd System load
ComEd = pd.read_csv(os.path.join(outdir,'ComEd System Loads','hrl_load_estimated.csv'),
                                 parse_dates=['datetime_beginning_utc']).rename({
                                     'datetime_beginning_utc':'time',
                                     'estimated_load_hourly':'system load'}, 
                                     axis=1)[['time','system load']]

ComEd['time'] = ComEd['time']-DateOffset(hours=6); ComEd.index.freq = 'h'
ComEd.set_index(['time'], inplace=True)
ComEd = ComEd/1e3 # MW to GW

### (1) plot as time series (hourly)
fig, ax = plt.subplots(figsize=(6,3))
Hcolors = ['darkred','red','darkmagenta','m']
Ccolors = ['royalblue','skyblue','darkcyan','c']
DRH.plot.area(stacked=True, color=Hcolors, linewidth=1, ax=ax)
DRC.plot.area(stacked=True, color=Ccolors, linewidth=1, ax=ax)
ax.set_ylim(top=10)
ax.set_ylabel('DR potential (GW)')
ax.margins(x=0)

ax2 = ax.twinx()
ComEd.plot(color='grey', legend=False, linewidth=1, ax=ax2)
ax2.set_ylim(bottom=0)
ax2.set_ylabel('System load (GW)', color='dimgrey')
ax2.tick_params(axis='y', colors='dimgrey')
ax2.spines['right'].set_color('dimgrey')

l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l2+l1[::-1], lab2+lab1[::-1], loc='center left', bbox_to_anchor=(1.125,0.5))
#ax.xaxis.set_major_locator(plt.LinearLocator(5)); 
ax.minorticks_off()
#ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d"))
ax.xaxis.label.set_visible(False)
#plt.setp(ax3b.get_xticklabels(), rotation=0, ha="center")

fig.savefig(os.path.join(outdir_dr,
                                'System_load_DR_ts_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    service_type, fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

### (1)B plot as time series (DR by daily max of system load)
freq = 'd' # <------
fig, ax = plt.subplots(figsize=(6,3))
Hcolors = ['darkred','red','darkmagenta','m']
Ccolors = ['royalblue','skyblue','darkcyan','c']


# retrieve df per daily peak system hour
Tog = pd.concat([DRH, DRC, ComEd], axis=1)
Tog = Tog.loc[Tog.groupby(Tog.index.date)['system load'].idxmax()]
Tog = Tog.groupby(Tog.index.date).max()

Tog.iloc[:,:4].plot.area(stacked=True, color=Hcolors, linewidth=1, ax=ax)
Tog.iloc[:,4:8].plot.area(stacked=True, color=Ccolors, linewidth=1, ax=ax)
ax.set_ylim(top=10)
ax.set_ylabel('DR potential (GW)')
ax.margins(x=0)

ax2 = ax.twinx()
Tog['system load'].plot(color='grey', legend=False, linewidth=1, ax=ax2)
ax2.set_ylim(bottom=0)
ax2.set_ylabel('System load (GW)', color='dimgrey')
ax2.tick_params(axis='y', colors='dimgrey')
ax2.spines['right'].set_color('dimgrey')
ax2.margins(x=0)

l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l2+l1[::-1], lab2+lab1[::-1], loc='center left', bbox_to_anchor=(1.125,0.5))
ax.xaxis.set_major_locator(plt.LinearLocator(5)); 
ax.minorticks_off()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d"))
ax.xaxis.label.set_visible(False)
#plt.setp(ax3b.get_xticklabels(), rotation=0, ha="center")
ax.set_title('Demand at daily system peak')
fig.savefig(os.path.join(outdir_dr,
                                'System_load_DR_daily_peak_hour_ts_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    service_type, fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

###  (2) plot as load duration
ComEd2 = ComEd.sort_values(by='system load', ascending=False)

fig, ax = plt.subplots(figsize=(5,3))
Hcolors = ['darkred','red','darkmagenta','m']
Ccolors = ['royalblue','skyblue','darkcyan','c']
DRH2 = DRH.sum(axis=1).reindex(ComEd2.index).rename('agg. inst. heating DR').reset_index(drop=True).replace({0:np.NAN})
DRC2 = DRC.sum(axis=1).reindex(ComEd2.index).rename('agg. inst. cooling DR').reset_index(drop=True).replace({0:np.NAN})

ax.scatter(DRH2.index, DRH2, color='m', s=1, label=DRH2.name)
ax.scatter(DRC2.index, DRC2, color='c', s=1, label=DRC2.name)

ax.set_ylim(top=10)
ax.set_ylabel('DR potential (GW)')
ax.margins(x=0)

ax2 = ax.twinx()
ComEd2.reset_index(drop=True).plot(color='grey', legend=False, ax=ax2)
ax2.set_ylim(bottom=0)
ax2.set_ylabel('System load (GW)', color='dimgrey')
ax2.tick_params(axis='y', colors='dimgrey')
ax2.spines['right'].set_color('dimgrey')

l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l2+l1, lab2+lab1) #loc='center left', bbox_to_anchor=(1.2,0.5))
ax.set_xlabel('No. hrs above')

fig.savefig(os.path.join(outdir_dr,
                                'System_load_DR_LDR_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                    service_type, fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

ComEd3H = get_DR(ComEd, 'heating').sort_values(by='system load', ascending=False)
DRH3 = DRH.reindex(ComEd3H.index); TDDH3 = (DRH+DBLH.values).reindex(ComEd3H.index)
ComEd3C = get_DR(ComEd, 'cooling').sort_values(by='system load', ascending=False)
DRC3 = DRC.reindex(ComEd3C.index); TDDC3 = (DRC+DBLC.values).reindex(ComEd3C.index)

print('\n Avg top 5 winter peak coincident heating DR [% system total]:')
display((DRH3.head(5)/TDDH3.head(5).values).agg(['mean', 'min', 'max']))
print('\n Avg top 5 summer peak coincident cooling DR [% system total]:')
display((DRC3.head(5)/TDDC3.head(5).values).agg(['mean', 'min', 'max']))

print('\nFor top 5 seasonal system peak hours:')
print('---> Avg total inst. heating DR during winter: {} GW'.format(DRH3.head(5).sum().mean()))
print('     % contribution by service class:')
display(DRH3.head(5).divide(DRH3.head(5).sum(axis=1).values, axis=0).mean()*100)

print('---> Avg total inst. cooling DR during summer: {} GW'.format(DRC3.head(5).sum().mean()))
print('     % contribution by service class:')
display(DRC3.head(5).divide(DRC3.head(5).sum(axis=1).values, axis=0).mean()*100)

# %% system agg diurnal plot
winters = (ComEd.index<heat1e)|(ComEd.index>=heat2s)
summers = (ComEd.index>=cools) & (ComEd.index<coole)
shoulders = ((ComEd.index>=heat1e)&(ComEd.index<cools)) | ((ComEd.index>=coole)&(ComEd.index<heat2s))

TDs = {}
TDs['winter'] = diurnal_box(ComEd[winters]['system load'])
TDs['summer'] = diurnal_box(ComEd[summers]['system load']) 

tds = {}
tds['winter'] = diurnal(ComEd[winters]['system load']) # mean std
tds['summer'] = diurnal(ComEd[summers]['system load']) # mean std

DRs = {}
DRs['winter'] = diurnal_box(DRH.sum(axis=1)[winters]) # heating only in winter
DRs['summer'] = diurnal_box(DRC.sum(axis=1)[summers]) # cooling only in summer

fig5, ([ax53,ax54]) = plt.subplots(1,2, sharex=True, sharey='row', figsize=(7,3))

season = 'winter'; ax = ax53
# bp1 = TDs[season].boxplot(color='grey', grid=False, ax=ax)
bp2 = DRs[season].boxplot(color='m', grid=False, sym='m.', ax=ax); 
ax2 = ax.twinx(); ax2.set_ylabel('system load (GW)')
ax2.plot(tds[season].index+1, tds[season]['mean'], color='grey', label='system load')
# tds[season]['mean'].plot(color='grey', ax=ax2)
ax.autoscale(); ax2.set_ylim(0,None); 
pDR =  DRs[season].sum().sum()/2/(TDs[season].sum().sum()/2)
leg_total = mpatches.Patch(color='grey', label='system load')
leg_dr = mpatches.Patch(color='m', label= 'agg. inst. heating DR');#, round(pDR*100,2)))
ax.legend(handles=[leg_total, leg_dr], loc='upper center', bbox_to_anchor=(0.5, -0.275)); #ax.minorticks_off()
ax.set_ylabel('Demand response (GW)'); ax.margins(x=0); #ax.xaxis.set_major_locator(plt.MultipleLocator(3));
ax.set_title(season)

season = 'summer'; ax = ax54
# bp1 = TDs[season].boxplot(color='grey', grid=False, ax=ax)
bp2 = DRs[season].boxplot(color='c', grid=False, sym='c.', ax=ax); 
ax3 = ax.twinx(); 
ax3.plot(tds[season].index+1, tds[season]['mean'], color='grey',label='system load')
# tds[season]['mean'].plot(color='grey', ax=ax3)
ax3.set_ylim(0,None); ax3.set_ylabel('system load (GW)')
ax.autoscale(); 
pDR =  DRs[season].sum().sum()/2/(TDs[season].sum().sum()/2)
leg_total = mpatches.Patch(color='grey', label='system load')
leg_dr = mpatches.Patch(color='c', label= 'agg. inst. cooling DR');#, round(pDR*100,2)))
ax.legend(handles=[leg_total, leg_dr], loc='upper center', bbox_to_anchor=(0.5, -0.275)); #ax.minorticks_off()
ax.set_ylabel('Demand response (GW)'); ax.margins(x=0); #ax.xaxis.set_major_locator(plt.MultipleLocator(3));
ax.set_title(season)

xticks = np.arange(1,25,3)
labels = np.arange(0,24,3)
ax.set_xticks(xticks)
ax.set_xticklabels(labels)

# reset 2nd axis ylim
ylim_ub = max(ax2.get_ylim()[1], ax3.get_ylim()[1])+1
ax2.set_ylim(0, ylim_ub); ax3.set_ylim(0, ylim_ub) 

ylim_ub = max(ax53.get_ylim()[1], ax54.get_ylim()[1])+1
ax53.set_ylim(0, ylim_ub); ax54.set_ylim(0, ylim_ub) 

fig5.tight_layout()

fig5.savefig(os.path.join(outdir_dr,
                            'System_diurnal_heat_cool_DR_seasons_box_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                service_type, fourier_type, log_ext, pdata_ext, r2_ext,
                                fig_ext)), dpi=300, bbox_inches='tight')

# %% Comparison of service type

### avg permise demand by service type
freq = 'd' # <--------- 'hh', 'h', 'd', 'm'
freq_descr = {'hh': 'half-hourly',
              'h': 'hourly',
              'd': 'daily',
              'm': 'monthly'}

if freq not in ['hh', 'h']:
    ylab = 'kWh'
else:
    ylab = 'kW'
    
    
ylim_ub = 0    
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, sharex=True, sharey='row', figsize=(10,6))
if freq != 'hh':
    TTd = TempS.resample(freq).mean()
TTd.plot(color='green', linewidth=1, ax=ax1)
ax1.axhline(0, linestyle='--', linewidth=1, color='k'); ax1.set_ylabel('degC');
ax1.margins(x=0); ax1.minorticks_off()


TTd.plot(color='green', linewidth=1, ax=ax2)
ax2.axhline(0, linestyle='--', linewidth=1, color='k'); ax2.set_ylabel('degC');
ax2.margins(x=0); ax2.minorticks_off()

### HVAC loads
service_type = 'C23'; ax = ax3 # <<<<<
typ = type_descr[service_type]
ax.set_title('{} ({})'.format(long_type_descr[service_type], freq_descr[freq]))
DRi = pd.concat([AvgBL[typ],AvgDRH[typ], AvgDRC[typ]], axis=1)
DRi.columns = ['baseload', 'heating load','cooling load']
# DRi['baseload'] = AvgTDD[typ]-DRi.sum(axis=1)
# DRi = DRi[['baseload','heating load','cooling load']]

if freq != 'hh':
    DRi = DRi.resample(freq).sum()/2 # half hourly kW -> kWh
    td = AvgTDD[typ].resample(freq).sum()/2 # half hourly kW -> kWh
    
for col in DRi.columns:
    pDR = DRi[col].sum()/DRi.sum().sum()
    DRi = DRi.rename({col:'{} ({}%)'.format(col, round(pDR*100,1))}, axis=1)

DRi.cumsum(axis=1).iloc[:,2].plot.area(color='c', linewidth=1, ax=ax)
DRi.cumsum(axis=1).iloc[:,1].plot.area(color='m', linewidth=1, ax=ax)   
DRi.cumsum(axis=1).iloc[:,0].plot.area(color='tab:blue', linewidth=1, ax=ax)  
td.plot(linewidth=1, color='k', label= 'whole-home demand', ax=ax)
ylim_ub = max(ylim_ub, td.max())

service_type = 'C24'; ax = ax4 # <<<<<
typ = type_descr[service_type]
ax.set_title('{} ({})'.format(long_type_descr[service_type], freq_descr[freq]))
DRi = pd.concat([AvgBL[typ],AvgDRH[typ], AvgDRC[typ]], axis=1)
DRi.columns = ['baseload', 'heating load','cooling load']
# DRi['baseload'] = AvgTDD[typ]-DRi.sum(axis=1)
# DRi = DRi[['baseload','heating load','cooling load']]

if freq != 'hh':
    DRi = DRi.resample(freq).sum()/2 # half hourly kW -> kWh
    td = AvgTDD[typ].resample(freq).sum()/2 # half hourly kW -> kWh
    
for col in DRi.columns:
    pDR = DRi[col].sum()/DRi.sum().sum()
    DRi = DRi.rename({col:'{} ({}%)'.format(col, round(pDR*100,1))}, axis=1)

DRi.cumsum(axis=1).iloc[:,2].plot.area(color='c', linewidth=1, ax=ax)
DRi.cumsum(axis=1).iloc[:,1].plot.area(color='m', linewidth=1, ax=ax)   
DRi.cumsum(axis=1).iloc[:,0].plot.area(color='tab:blue', linewidth=1, ax=ax)  
td.plot(linewidth=1, color='k', label= 'whole-home demand', ax=ax) 
ylim_ub = max(ylim_ub, td.max())

service_type = 'C25'; ax = ax5 # <<<<<
typ = type_descr[service_type]
ax.set_title('{} ({})'.format(long_type_descr[service_type], freq_descr[freq]))
DRi = pd.concat([AvgBL[typ],AvgDRH[typ], AvgDRC[typ]], axis=1)
DRi.columns = ['baseload', 'heating load','cooling load']
# DRi['baseload'] = AvgTDD[typ]-DRi.sum(axis=1)
# DRi = DRi[['baseload','heating load','cooling load']]

if freq != 'hh':
    DRi = DRi.resample(freq).sum()/2 # half hourly kW -> kWh
    td = AvgTDD[typ].resample(freq).sum()/2 # half hourly kW -> kWh
    
for col in DRi.columns:
    pDR = DRi[col].sum()/DRi.sum().sum()
    DRi = DRi.rename({col:'{} ({}%)'.format(col, round(pDR*100,1))}, axis=1)

DRi.cumsum(axis=1).iloc[:,2].plot.area(color='c', linewidth=1, ax=ax)
DRi.cumsum(axis=1).iloc[:,1].plot.area(color='m', linewidth=1, ax=ax)   
DRi.cumsum(axis=1).iloc[:,0].plot.area(color='tab:blue', linewidth=1, ax=ax)  
td.plot(linewidth=1, color='k', label= 'whole-home demand', ax=ax) 
ylim_ub = max(ylim_ub, td.max())

service_type = 'C26'; ax = ax6 # <<<<<
typ = type_descr[service_type]
ax.set_title('{} ({})'.format(long_type_descr[service_type], freq_descr[freq]))
DRi = pd.concat([AvgBL[typ],AvgDRH[typ], AvgDRC[typ]], axis=1)
DRi.columns = ['baseload', 'heating load','cooling load']
# DRi['baseload'] = AvgTDD[typ]-DRi.sum(axis=1)
# DRi = DRi[['baseload','heating load','cooling load']]

if freq != 'hh':
    DRi = DRi.resample(freq).sum()/2 # half hourly kW -> kWh
    td = AvgTDD[typ].resample(freq).sum()/2 # half hourly kW -> kWh
    
for col in DRi.columns:
    pDR = DRi[col].sum()/DRi.sum().sum()
    DRi = DRi.rename({col:'{} ({}%)'.format(col, round(pDR*100,1))}, axis=1)

DRi.cumsum(axis=1).iloc[:,2].plot.area(color='c', linewidth=1, ax=ax)
DRi.cumsum(axis=1).iloc[:,1].plot.area(color='m', linewidth=1, ax=ax)   
DRi.cumsum(axis=1).iloc[:,0].plot.area(color='tab:blue', linewidth=1, ax=ax)  
td.plot(linewidth=1, color='k', label= 'whole-home demand', ax=ax)
ylim_ub = max(ylim_ub, td.max())

ylim_ub = ylim_ub*1.05
for ax in [ax3, ax4, ax5, ax6]:
    ax.set_ylim(0,ylim_ub); 
    ax.set_ylabel(ylab);
    ax.margins(x=0);  ax.minorticks_off()
    ax.legend(prop={'size': 7})
    ax.xaxis.label.set_visible(False)
    # ax.xaxis.set_major_locator(plt.LinearLocator(5)); 
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d"))
    #plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

fig.tight_layout()
fig.savefig(os.path.join(outdir_dr,
                            'All_Heat_cool_demand_{}_{}_{}{}_avg_per_premise{}{}.{}'.format(
                                freq_descr[freq], service_type, fourier_type, log_ext, pdata_ext, r2_ext,
                                fig_ext)), dpi=300, bbox_inches='tight')
    

# %% compare heating and cooling plots
smax = 1; smin = 1
for typ in NACC.keys():
    smin = max(smin, NACC[typ].min())
    smax = max(smax, NACC[typ].max())

# for controlling the size of the bubble plot
bmin = 10
bmax = 250
ms = (bmax-bmin)/(smax-smin)

# (1) AnnDRH, AnnDRC: for storing annual zipcode DR
fig, ax = plt.subplots(figsize=(4,4))
for typ in AnnDRH.keys():
    ax.scatter(AnnDRH[typ], AnnDRC[typ], s=bmin+ms*NACC[typ], alpha=0.5, edgecolors='k', label=type_descr[typ])
ax.set_xlabel('annual heating load (kWh)')
ax.set_ylabel('annual cooling load (kWh)')
ax.axis('square')
ax.plot(ax.get_xlim(),ax.get_xlim(), '--', color='grey')
ax.legend()
#ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.4, -0.3))
fig.tight_layout()
fig.savefig(os.path.join(outdir_dr,
                                'Compare_annual_DR_Model{}{}_avg_per_premise{}{}.{}'.format(
                                    fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

# (2) PDRH, PDRC: for storing annual zipcode DR percents
fig, ax = plt.subplots(figsize=(3,3))
for typ in PDRH.keys():
    ax.scatter(PDRH[typ]*100, PDRC[typ]*100, s=bmin+ms*NACC[typ], alpha=0.5, edgecolors='k', label=type_descr[typ])
ax.set_xlabel('heating as % annual load')
ax.set_ylabel('cooling as % annual load')
ax.axis('square')
ax.plot(ax.get_xlim(),ax.get_xlim(), '--', color='grey')
ax.legend()
#ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.4, -0.3))
fig.tight_layout()
fig.savefig(os.path.join(outdir_dr,
                                'Compare_percent_annual_DR_Model{}{}_avg_per_premise{}{}.{}'.format(
                                    fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

# (3) SPDRH, SPDRC: for storing seasonal zipcode DR percents
fig, ax2 = plt.subplots(figsize=(3,3))
for typ in SPDRH.keys():
    ax2.scatter(SPDRH[typ]*100, SPDRC[typ]*100, s=bmin+ms*NACC[typ], alpha=0.5, edgecolors='k', label=type_descr[typ])
ax2.set_xlabel('heating as % winter load')
ax2.set_ylabel('cooling as % summer load')
ax2.axis('square')
ax2.legend()
ax2.plot(ax2.get_xlim(),ax2.get_xlim(), '--', color='grey')
#ax2.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.4, -0.3))
fig.tight_layout()
fig.savefig(os.path.join(outdir_dr,
                                'Compare_percent_seasonal_DR_Model{}{}_avg_per_premise{}{}.{}'.format(
                                    fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

# (4): (1) + (3)
fig, (ax, ax2) = plt.subplots(1,2, figsize=(7,3.5))
for typ in AnnDRH.keys():
    ax.scatter(AnnDRH[typ], AnnDRC[typ], s=bmin+ms*NACC[typ], alpha=0.5, edgecolors='k', label=type_descr[typ])
ax.set_xlabel('annual heating load (kWh)')
ax.set_ylabel('annual cooling load (kWh)')
ax.axis('square')
ax.plot(ax.get_xlim(),ax.get_xlim(), '--', color='grey')
for typ in SPDRH.keys():
    ax2.scatter(SPDRH[typ]*100, SPDRC[typ]*100, s=bmin+ms*NACC[typ], alpha=0.5, edgecolors='k', label=type_descr[typ])
ax2.set_xlabel('heating as % winter load')
ax2.set_ylabel('cooling as % summer load')
ax2.axis('square')
ax2.plot(ax2.get_xlim(),ax2.get_xlim(), '--', color='grey')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle('Thermostatically controlled loads')
fig.tight_layout()
fig.savefig(os.path.join(outdir_dr,
                                'Compare_annual_percent_seasonal_TCL_Model{}{}_avg_per_premise{}{}.{}'.format(
                                    fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

# (7): (6) + (7)
fig, (ax, ax2) = plt.subplots(1,2, figsize=(7,3.5))
for typ in AnnDRH.keys():
    ax.scatter(PeakDRH[typ], PeakDRC[typ], s=bmin+ms*NACC[typ], alpha=0.5, edgecolors='k', label=type_descr[typ])
ax.set_xlabel('heating DR in winter (kW)')
ax.set_ylabel('cooling DR in summer (kW)')
ax.axis('square')
ax.plot(ax.get_xlim(),ax.get_xlim(), '--', color='grey')
for typ in SPDRH.keys():
    ax2.scatter(SPDRH[typ]*100, SPDRC[typ]*100, s=bmin+ms*NACC[typ], alpha=0.5, edgecolors='k', label=type_descr[typ])
ax2.set_xlabel('heating DR as % winter load')
ax2.set_ylabel('cooling DR as % summer load')
ax2.axis('square')
ax2.plot(ax2.get_xlim(),ax2.get_xlim(), '--', color='grey')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle('Avg Top 5 seasonal peak coincident DR')
fig.tight_layout()
fig.savefig(os.path.join(outdir_dr,
                                'Compare_annual_percent_seasonal_DR_Model{}{}_avg_per_premise{}{}.{}'.format(
                                    fourier_type, '_log' if log_transform else '',
                                    pdata_ext, r2_ext,
                                    fig_ext)), dpi=300, bbox_inches='tight')

# %% [4] Run panel for Aggregating result value tables
fourier_type = '82' # <---------- # '44' or '42'
log_transform = False # <------- # if modeling data as log(data) in [W]

fig_ext = 'png' # <-------

ylog_msg = 'log ' if log_transform else ''
type_descr = {'C23': 'SF-NE',
              'C24': 'MF-NE',
              'C25': 'SF-E',
              'C26': 'MF-E'}
long_type_descr = {'C23': 'Single family non-electric',
                   'C24': 'Multi-family non-electric',
                   'C25': 'Single family electric',
                   'C26': 'Multi-family electric'}

outdir_dr = os.path.join(outdir, 'Model{}{}_DR'.format(fourier_type, '_log' if log_transform else ''))
if not os.path.exists(outdir_dr):
        os.mkdir(outdir_dr)

print('>>> output directory: {}'.format(outdir_dr))

Res = []
for counter, typ in enumerate(['C26','C25','C24','C23'],1):
    
    print('{}. loading data for {}...'.format(counter, typ))
    service_type = typ
    outdir_sub = folder_set_up(service_type, fourier_type, log_transform)
    
    ############### get completed list
    zc_list = list(os.path.splitext(os.path.basename(y))[0][-5:] for y in
        list(filter(lambda x: x.endswith('.pkl'), os.listdir(outdir_sub))))
    zc_list = list(map(int, zc_list))
    
    for zc in zc_list:
        dfV = get_results(zc, service_type, fourier_type, log_transform)
        Res.append(dfV)
    
Res = pd.concat(Res, axis=0)
display(Res)

Res.to_csv(os.path.join(outdir_dr, 'Results_all_Model{}{}.csv'.format(
    fourier_type, '_log' if log_transform else '')), index=False)

print('output saved!')


# %% plot regression paras for all service types

fourier_type = '82' # <---------- # '44' or '42'
log_transform = False # <------- # if modeling data as log(data) in [W]

fig_ext = 'png' # <-------

with_mean_std = True # <-----------
pdata_lb = 95 # <------- # inclusive, query res.pdata >= pdata_lb/100

ylog_msg = 'log ' if log_transform else ''
type_descr = {'C23': 'SF-NE',
              'C24': 'MF-NE',
              'C25': 'SF-E',
              'C26': 'MF-E'}
long_type_descr = {'C23': 'Single family non-electric',
                   'C24': 'Multi-family non-electric',
                   'C25': 'Single family electric',
                   'C26': 'Multi-family electric'}

outdir_dr = os.path.join(outdir, 'Model{}{}_DR'.format(fourier_type, '_log' if log_transform else ''))
if not os.path.exists(outdir_dr):
        os.mkdir(outdir_dr)
        
Res = pd.read_csv(os.path.join(outdir_dr, 'Results_all_Model{}{}.csv'.format(
    fourier_type, '_log' if log_transform else '')))
Res.columns

Res = Res[Res['opt']==1]

#### zip code of highest R2 per service type ####
print('Of the highest R2...')
Res.loc[Res.groupby('type')['r2'].idxmax()]
print('\nOf median R2...')
pd.concat([Res[(Res['r2']>=i-0.000001) & (Res['r2']<=i+0.000001)] for i in Res.groupby(
    'type')['r2'].median()], axis=0)
print('\nOf the lowest R2...')
Res.loc[Res.groupby('type')['r2'].idxmin()]



print('>>> output directory: {}'.format(outdir_dr))

### old code

pdata_ext = 'al_{}pct_pdata'.format(pdata_lb) if pdata_lb > 0 else ''
Res_tags = ['mean','stderr','pval','cilb', 'ciub','sig','hsig']

Nr, Nc = 2,2
for to_plot in ['percent data', 'r-square','slope','intercept','change-point', 'yf']:
    fig = plt.figure(figsize=(3*Nc,2*Nr))
    for n, typ in enumerate(['C23','C24','C25','C26'],1):
        service_type = typ

        df_para = Res.query('(type==@service_type) & (pdata>=@pdata_lb/100)')
        if to_plot in ['slope','intercept','change-point']:
            if to_plot == 'slope':
                varH, varC = 'mh', 'mc'
            elif to_plot == 'intercept':
                varH, varC = 'bh', 'bc'
            elif to_plot == 'change-point':
                varH, varC = 'xh', 'xc' # change-points
            mh, mc, = np.array(df_para[varH].dropna()), np.array(df_para[varC].dropna())
        elif to_plot in ['r-square','yf']:
            if to_plot == 'r-square':
                varH = 'r2' # r-sq
            elif to_plot == 'yf':
                varH = 'yh' # bound between HVAC and non-HVAC regions
            mh = np.array(df_para[varH].dropna())
        elif to_plot == 'percent data':
            varH = 'pdata'
            mh = np.array(df_para[varH].dropna())*100
        #display(mh)
        #display(mc)
        
        ### slopes
        ax = fig.add_subplot(Nr, Nc, n)
        sns.distplot(mh, norm_hist=True, kde=True, hist=True, label='{} hist'.format(varH), 
                     color=sns.color_palette()[3])
        if to_plot in ['slope','intercept','change-point']:
            sns.distplot(mc, norm_hist=True, kde=True, hist=True, label='{} hist'.format(varC), 
                         color=sns.color_palette()[0])
        # to zoom in on the x axis
#            if to_plot == 'slope':
#                ax.set_xlim(-0.1, 0.1)
#                ax.set_xlim(max(ax.get_xlim()[0],-0.1), min(ax.get_xlim()[1],0.1))
        
#        if to_plot == 'slope':
#            # mark positive and negative regions
#            ax.axvspan(ax.get_xlim()[0], 0, facecolor='magenta', alpha=0.1)
#            ax.axvspan(0, ax.get_xlim()[1], facecolor='cyan', alpha=0.1)
        
        if with_mean_std:
            # mean +/- std
            ax.axvspan(mh.mean()-mh.std(), mh.mean()+mh.std(),
                       facecolor=sns.color_palette('pastel')[3], alpha=0.25) # +/- 1 std
            ax.axvline(mh.mean(), linewidth=2, linestyle='--', label='{} mean'.format(varH),
                       color=sns.color_palette()[3])
            print('  >> for {}, mean {} = {}'.format(service_type, to_plot, mh.mean()))
            if to_plot in ['slope','intercept','change-point']:
                ax.axvspan(mc.mean()-mc.std(), mc.mean()+mc.std(),
                           facecolor=sns.color_palette('pastel')[0], alpha=0.25) # +/- 1 std
                ax.axvline(mc.mean(), linewidth=2, linestyle='--', label='{} mean'.format(varC),
                           color=sns.color_palette()[0])
                print('  >> for {}, mean {} = {}'.format(service_type, to_plot, mc.mean()))
                print()
        
        ax.margins(x=0)
        
        if to_plot == 'slope':
            xlab = 'ln(kW)/degC'
        elif to_plot in ['intercept', 'yf']:
            xlab = 'ln(kW)'
        elif to_plot == 'change-point':
            xlab = 'degC' # change-point
        elif to_plot == 'r-square':
            xlab = 'r-sq'
        elif to_plot == 'percent data':
            xlab = '%'
        if n > 2:
            ax.set_xlabel(xlab)
        if n%2 == 1:
            ax.set_ylabel('density')
        ax.set_title('{}: {}'.format(type_descr[service_type], to_plot))
        if (to_plot == 'slope') &( service_type in ['C23', 'C24']) | (to_plot == 'change-point'
                            ) & (service_type in ['C23', 'C24']) | (to_plot in ['r-square','percent data', 'intercept']):
            xpos = 0.02; ha = 'left'
        else:
            xpos = 0.98; ha = 'right'
        ax.text(xpos, 0.95, 'N = {}'.format(len(mh)), 
                horizontalalignment=ha, verticalalignment='top', transform=ax.transAxes)
        
        if n == 4:
            handles, labels = ax.get_legend_handles_labels()
            
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir_dr,'plot_{}_distr_{}_Model{}{}.{}'.format(
            to_plot.replace(' ', '_'), pdata_ext, fourier_type, '_log' if log_transform else '', fig_ext)),
            dpi=300, bbox_inches='tight')

# %%


