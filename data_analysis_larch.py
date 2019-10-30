import pandas as pd;import numpy as np;import os;import matplotlib.pyplot as plt;
import matplotlib.cm as cm
os.chdir(r'C:\Users\Nicolas\Dropbox\LUND\KEMXAFS\DATA\BALDER_021019')

import larch
import larch.xafs as xafs
import larch.xray as xray

plt.close('all')

dicten={}
for i in range(1,22):
    #print('CoChamp_A8_EXAFS_%03d.dat'%i)
    if i==1:continue
    cols='Pt_No  mono1_energy  albaem01_ti  albaem01_ch1  albaem01_ch2  albaem01_ch3  albaem01_ch4  xspress301_counter  xspress301_ch1_roi1  xspress301_ch2_roi1  xspress301_ch3_roi1  xspress301_ch4_roi1  xspress301_ch5_roi1  xspress301_ch6_roi1  xspress301_ch7_roi1  dt'
    cols=cols.split()
    df=pd.read_csv('CoChamp_A8_EXAFS_%03d.dat'%i,sep='\s+',skiprows=5,header=None,names=cols,skipfooter=1,index_col='mono1_energy')
    fluo=df['xspress301_ch3_roi1']/(df['albaem01_ch1']+df['albaem01_ch2'])
    trans=-np.log(df['albaem01_ch3']/(df['albaem01_ch1']+df['albaem01_ch2']))
    dicten['%03d'%i]=trans

df=pd.DataFrame(dicten)

def normalize(df,level='post'):
    if 'pre' in level: #pre-edge
        print('pre')
        normalized_df=(df-df[7700:7750].min())/(df.max()-df[7700:7750].min())
        return normalized_df
    elif 'post' in level:
        print('post')
        normalized_df=(df-df.min())/(df.max()-df.min())
        return normalized_df

data=df.mean(axis=1).to_frame(name='mean')

data['norm']=normalize(df).mean(axis=1)

fig,ax=plt.subplots()
ax.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()
data.plot(ax=ax)

#larch

dat=larch.Group
xafs.autobk(data.index.values,data['norm'].values,group=dat,rbkg=0.55,e0=7715)
data['bkg']=dat.bkg
data['pre_edge']=dat.pre_edge
data['post_edge']=dat.post_edge
data['flat']=(data.loc[dat.e0:,'norm']-data.loc[dat.e0:,'post_edge'])/dat.edge_step+1
data.loc[:dat.e0,'flat']=data.loc[:dat.e0,'norm']-data.loc[:dat.e0:,'pre_edge']

fig,ax=plt.subplots(figsize=(10,10))
data[['flat']].plot(ax=ax)

#data[['norm','bkg','pre_edge','post_edge']].plot(ax=ax)
def k_plots(k,chi):
    fig2,ax2=plt.subplots(1,4,figsize=(16,8))
    ax2[0].plot(k,chi,label='lin')
    ax2[1].plot(k,chi*k,label='k')
    ax2[2].plot(k,chi*k**2,label=r'k^2')
    ax2[3].plot(k,chi*k**3,label=r'k^3')
    #ax2.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()
    for i in range(4):
        ax2[i].set_xlim(0,14)
        ax2[i].set_ylim(-4.00*(i+1),4*(i+1))
        ax2[i].legend()
    return fig2,ax2
#ax2.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()
#fig3,ax3=plt.subplots(figsize=(10,10))
#xafs.pre_edge(data.index.values,data['norm'].values,group=dat.flat)
#ax3.plot(data.index.values,dat.flat)
#fig,ax=plt.subplots()
#xafs.xftf_fast(dat.chi, nfft=4096, kstep=0.05)
#x = np.linspace(0, 13, endpoint=True)
#xafs.ftwindow(x, xmin=0, xmax=13, dk=1)
#ax.plot(x,color='red', label='dk 1')
#ax.plot(dat.dat.k, color='b', label='dk 5')

####WINDOWS

#xafs.xftf(dat.k,dat.chi,kmin=3, kmax=6, dk=1, kweight=2, window='hanning', group=dat)
#fig4,ax4=plt.subplots(figsize=(10,10))
#ax4.plot(dat.k, dat.chi*dat.k**2,label=r'k^2')
#ax4.plot(dat.k, dat.kwin)
#ax4.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()

#xafs.xftf(dat.k,dat.chi,kmin=3, kmax=6, dk=5, kweight=2, window='hanning', group=dat)
#fig5,ax5=plt.subplots(figsize=(10,10))
#ax5.plot(dat.k, dat.chi*dat.k**2,label=r'k^2')
#ax5.plot(dat.k, dat.kwin)
#ax5.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()

####FFT    

#xafs.xftf(dat.k,dat.chi,kmin=3, kmax=6, dk=5, kweight=2, window='hanning', group=dat)
#fig,ax=plt.subplots()

#ax.plot(dat.r, dat.chir_re, color='red', label='chir_re')
#ax.plot(dat.r, dat.chir_im, color='blue', label='chir_im')
#ax.plot(dat.r, dat.chir_mag, color='k', label='chir_mag')
#ax.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()

####wavelet



kopts = {'xlabel': r'$k \,(\AA^{-1})$','ylabel': r'$k^2\chi(k) \, (\AA^{-2})$','linewidth': 3, 'title': 'Cobalt sample A8', 'show_legend':True}

fig7,ax7=plt.subplots()
ax7.plot(dat.k, dat.chi*dat.k**2, label='original data')#, **kopts)
ax7.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()

xafs.cauchy_wavelet(dat.k, dat.chi, group=dat, kweight=0)#, rmax_out=10)

x = dat.k
y = dat.r

imopts = {'x', 'y'}
fig10,ax10=plt.subplots()
ax10.imshow(dat.wcauchy_mag*dat.k*dat.r[0:320],  label='Wavelet Transform: Magnitude')

xafs.cauchy_wavelet(dat.k, dat.chi, group=dat, kweight=0)
fig11,ax11=plt.subplots()
ax11.imshow(dat.wcauchy_re*dat.k*dat.r[0:320], label='Wavelet Transform: Real Part')

#fig12,ax12=plt.subplots()
#ax12.plot(dat.k, dat.r[0:320], label='wavelet')#, **kopts)
#ax12.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()
ropts=kopts
fig9,ax9=plt.subplots()
ax9.plot(dat.r, dat.wcauchy_mag.sum(axis=1), label='projected wavelet')
ax9.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()

fig8,ax8=plt.subplots()
ax8.plot(dat.k,dat.wcauchy_re.sum(axis=0),label='projected wavelet')
ax8.legend(loc=4,fancybox=True,shadow=False,prop ={'size':16},numpoints=1,ncol=1)#.draggable()