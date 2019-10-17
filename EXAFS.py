# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:26:43 2019

@author: Nicolas
"""
import os
os.chdir(r'C:\Users\Nicolas\Dropbox\LUND\KEMXAFS\EXAFS study')
#os.chdir(r'C:\Users\Nicolas\Dropbox\LUND\KEMXAFS\DATA\BALDER_021019')
import matplotlib.pyplot as plt
import scipy.constants as const
import numpy as np
import pandas as pd
import larch
import larch.xafs as xafs
import larch.xray as xray
from numpy import sin,pi
#from scipy import optimize

plt.close('all')

#### DATA READING
dicten={}
for i in range(1,3):
    #print('CoChamp_A8_EXAFS_%03d.dat'%i)
    if i==1:continue
    cols='energy(eV)  Id  If  I0  I1  I2  Is  time log(I0/I1)'
    cols=cols.split()
    df=pd.read_csv('Jens_ferrocene_trans_1_1_%03d.dat'%i,sep='\s+',skiprows=2,header=None,names=cols,skipfooter=1,index_col='energy(eV)')
    #fluo=df['xspress301_ch3_roi1']/(df['albaem01_ch1']+df['albaem01_ch2'])
    trans=-np.log(df['I1']/(df['I0']))
    dicten['%03d'%i]=trans
df=pd.DataFrame(dicten)
def normalize(df,level='pre'):
    if 'pre' in level: #pre-edge
        print('pre')
        normalized_df=(df-df[7100:7150].min())/(df.max()-df[7100:7150].min())
        return normalized_df
    elif 'post' in level:
        print('post')
        normalized_df=(df-df.min())/(df.max()-df.min())
        return normalized_df
data=df.mean(axis=1).to_frame(name='mean')
data['norm']=normalize(df).mean(axis=1)

fig1,ax=plt.subplots(figsize=(8,8))
data.plot(ax=ax)
ax.set_xlabel('energy (eV)')
ax.set_ylabel('mu(E)')
ax.set_title('Non-normalized spectrum Ferrocene')
ax.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()

####LARCH

dat=larch.Group
xafs.autobk(data.index.values,data['norm'].values,group=dat,rbkg=0.55,e0=7112)
data['bkg']=dat.bkg
data['pre_edge']=dat.pre_edge
data['post_edge']=dat.post_edge
data['flat']=(data.loc[dat.e0:,'norm']-data.loc[dat.e0:,'post_edge'])/dat.edge_step+1
data.loc[:dat.e0,'flat']=data.loc[:dat.e0,'norm']-data.loc[:dat.e0:,'pre_edge']

fig2,ax2=plt.subplots(figsize=(8,8))
ax2.plot(data[['flat']], label='flattened')
ax2.set_xlabel('energy (eV)')
ax2.set_ylabel('mu(E)')
ax2.set_title('Normalized spectrum Ferrocene')
ax2.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)

fig3,ax3=plt.subplots(1,4,figsize=(16,8))
ax3[0].plot(dat.k,dat.chi,label='linear')
ax3[1].plot(dat.k,dat.chi*dat.k,label='k')
ax3[2].plot(dat.k,dat.chi*dat.k**2,label=r'k^2')
ax3[3].plot(dat.k,dat.chi*dat.k**3,label=r'k^3')
fig3.suptitle('k-space Ferrocene', fontsize=24)
for i in range(4):
    ax3[i].set_xlim(0,14)
    ax3[i].set_ylim(-4.00*(i+1),4.00*(i+1))
    ax3[i].legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)

####WINDOWS
'''
xafs.xftf(dat.k,dat.chi,kmin=3, kmax=12, dk=3, kweight=0, window='kaiser', group=dat)
fig4,ax4=plt.subplots(figsize=(10,10))
ax4.plot(dat.k, dat.chi*dat.k**2,label=r'k^2, kaiser')
ax4.plot(dat.k, dat.kwin)
ax4.set_title('k-space window')
ax4.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()
'''
####FFT    

xafs.xftf(dat.k,dat.chi,kmin=3, kmax=10, dk=5, dk2=5, kweight=2, window='hanning', group=dat)
fig5,ax5=plt.subplots(2,1,figsize=(13,9))

ax5[0].plot(dat.r, dat.chir_re, color='red', label='Re part')
ax5[0].plot(dat.r, dat.chir_im, color='blue', label='Im part')
ax5[0].plot(dat.r, dat.chir_mag, color='k', label='Magnitude')
ax5[0].set_title('FT + Re + Im')
ax5[0].set_xlabel('R (Å))')
ax5[0].set_ylabel('Amplitude + Magnitude')
ax5[1].plot(dat.r, dat.chir_mag, color='k', label='Magnitude')
ax5[1].set_title('R space/3-10/dk5/hanning/k**2')
ax5[1].set_xlabel('R (Å))')
ax5[1].set_ylabel('Magnitude')
fig5.suptitle('Fourier transform Ferrocene', fontsize=16)
ax5[0].legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()
ax5[1].legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()

####BFT
xafs.xftr(dat.r, dat.chir, rweight=2, rmin=4.5, rmax=7.5, dr=5, dr2=5, window='hanning', qmax_out=14 , group=dat)
fig6,ax6=plt.subplots(figsize=(9,9))
ax6.set_title('q space')
ax6.set_xlabel('q (Å^-1))')
ax6.set_ylabel('q (k) (Å^-2)')
ax6.plot(dat.q, dat.chiq_re,alpha=0.1, color='red', label='Re part')
ax6.plot(dat.q, dat.chiq_im,alpha=0.1, color='blue', label='Im part')
ax6.plot(dat.q, dat.chiq_mag, color='k', label='bFT')
ax6.plot(dat.k,dat.chi*dat.k**2, color='green', label=r'k^2')
ax6.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)


'''
def xafs(fname):
    data = np.genfromtxt(fname, dtype = float, delimiter = '  ',invalid_raise = False, skip_header=2)
    return data
data1 = xafs('normalized_ferro.txt')

a = np.array([row[0] for row in data1])
b = np.array([row[1] for row in data1])

E0 = 7112 #dissociation energy in eV
E=np.arange(6910, 8105, 0.1) #energy ranges in eV not exp
distance=np.array([2.14610243973E-10, 1.56E-10])
variance = 0.0050E-10 #debye waller in Å
N = 20 #degeneracy in ferrocene
S = 1.02 # electron overlap

def E_J_exp(a):
    return a*const.electron_volt

def E_k_exp(a,E0):
    return (1/const.hbar) * (1/1) * np.sqrt(2. * const.m_e * (E_J_exp(a) - E_J_exp(E0)))

def E_J(E):
    return E*const.electron_volt

def E_k(E,E0):
    return (1/const.hbar) * (1/1) * np.sqrt(2. * const.m_e * (E_J(E) - E_J(E0)))


k = E_k(E,E0)
k_exp = E_k_exp(a,E0)*1E-10
lambd = (2.*pi)/k
F=0.25E-10#scaling factor
for dist in distance:
    oscill=sin(2*k*dist)
    debye=np.exp(-2*((k**2)*(variance**2)))
    mfp=np.exp(-2*(dist/lambd))#mean free path

y=(S**2*N*oscill*(mfp/(k*(dist**2)))*debye)*F
x=k*1E-10
c=b

fig = plt.figure(figsize=(7,9))
ax = fig.add_subplot()
#ax.plot(a,b, 'b-')
ax.plot(x,(-y*x**2)-.1, 'b-', label='fit')
ax.plot(k_exp-1.1, (c-1)*k_exp**2, '.', color='red', label='Data')
#ax.set_ylim([-0.5, 0.06])
ax.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()
ax.set_ylabel(r'$\mu$(E)',fontsize=16) 
ax.set_xlabel('k [Å$^{-1}$]',fontsize=16)
ax.set_xlim(0,12)

#enumerate generate second list with degeneracies, think of using names, path 1, path 2, use dcitionary as name container, have factor degeneracy

'''