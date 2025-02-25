
import matplotlib.pyplot as plt 
from sklearn.preprocessing import normalize 
from matplotlib.patches import Patch
import numpy as np
import paxplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import seaborn as sns 
import pandas as pd 


import paxplot
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from statannotations.Annotator import Annotator


def pax_plot_data(data,cols,labels,pallete,savepath, save= False):
    # # Import data

    scaler_min = MinMaxScaler()
    scaler_std = StandardScaler()
    # scaler_norm = normalize()
    data_to_plot = data
    cols = cols
    labels =  labels
    # Create figure
    paxfig = paxplot.pax_parallel(n_axes=len(cols))
    paxfig.set_figheight(10)
    paxfig.set_figwidth(30)

    scaled_min = scaler_min.fit_transform(data_to_plot.to_numpy())

    scaled_std = scaler_std.fit_transform(data_to_plot.to_numpy())

    scaled_norm = normalize(data_to_plot.to_numpy(),axis=0)

    for i in list(set(labels)):
        idx = np.where(labels==i)
        paxfig.plot(
            scaled_min[idx[0],:], 
            line_kwargs={'alpha': 0.1, 'color': pallete[i], 'zorder': 1}
        )

        paxfig.plot(
            [np.mean(scaled_min[idx[0],:],axis=0)], 
            line_kwargs={'alpha': 1., 'color': pallete[i],'linewidth':2, 'zorder': 1}
        )

    for  ax in paxfig.axes:   
        ax.set_yticks([0,0.5,1])
        ax.tick_params(axis='x', labelsize=20,rotation=90)
        ax.tick_params(axis='y', labelsize=10)
    # # Add labels
    paxfig.set_labels(cols)
    # paxfig.legend()
    if save: 
        plt.savefig(savepath,dpi=300)

    plt.show()

def return_confusion_matrix_(df1,df2,label1_name,label2_name,vmin=0,vmax=100,figsize =[12,5],shuffle = False,save=False,savepath=None,cmap='BrBG_r',annot_kws=None):
    np.random.seed(42)
    if shuffle:
        fig,ax1 = plt.subplots(figsize =figsize )
        df = pd.DataFrame(columns=['label1','label2'])
        # df['exp_name1'] = df1.exp_name
        # df['exp_name2'] = df2.exp_name
        
        label1 = list(df1[label1_name])
        np.random.shuffle(label1)
        label2 = list(df2[label2_name])
        np.random.shuffle(label2)

        df['label1_sh'] = label1
        df['label2_sh'] = label2

        df['label1'] = np.array(df1[label1_name]) 
        df['label2'] = np.array(df2[label2_name])

        mat_orig = np.zeros((len(set(df1[label1_name])),len(set(df2[label2_name]))))
        for i in np.unique(df.label1):
            data_ = np.unique(df[df.label1==i]['label2'],return_counts=True)
            mat_orig[i,data_[0]] =(data_[1]/np.sum(data_[1]))

        mat_sh = np.zeros((len(set(df['label1_sh'])),len(set(df['label2_sh']))))

        for i in np.unique(df.label1_sh):
            data_ = np.unique(df[df.label1_sh==i]['label2_sh'],return_counts=True)
            mat_sh[i,data_[0]] =(data_[1]/np.sum(data_[1]))


        sns.heatmap(mat_orig-mat_sh,cmap=cmap,annot=True,ax=ax1,vmin=vmin,vmax=vmax,annot_kws=annot_kws) 
        if save:
            plt.savefig(savepath,dpi=300)
        else:
            plt.show()

    else:
        fig,ax1 = plt.subplots(figsize =figsize )

        df = pd.DataFrame(columns=['label1','label2'])
        # df['exp_name1'] = df1.exp_name
        # df['exp_name2'] = df2.exp_name
        df['label1'] = np.array(df1[label1_name])
        df['label2'] = np.array(df2[label2_name])

        mat = np.zeros((len(set(df1[label1_name])),len(set(df2[label2_name]))))

        for i in np.unique(df.label1):
            data_ = np.unique(df[df.label1==i]['label2'],return_counts=True)
            mat[i,data_[0]] =(data_[1]/np.sum(data_[1]))

        sns.heatmap(mat,cmap=cmap,annot=True,vmin=vmin,vmax=vmax,annot_kws=annot_kws,ax=ax1) 

        if save:
            plt.savefig(savepath,dpi=300)
        else:
            plt.show()

def plot_cosine_mat(data1, data2, label1, label2,figsize =[12,5], save = False, savepath = None,annot_kws=None):
    cosine_mat = np.zeros((len(set(label1)),len(set(label2))))
    print(figsize)
    sim_data = cosine_similarity(data1,data2)
    fig,ax1 = plt.subplots(figsize =figsize )

    for i in set(label1):
        for j in set(label2):
            idx_FN = label1==i
            idx_SH = label2==j
            cosine_mat[i,j] = np.mean(sim_data[:,idx_FN][idx_SH])
    sns.heatmap(np.round(cosine_mat,decimals=2),ax=ax1,annot=True,vmax=1,vmin=-1,annot_kws=annot_kws)

    if save:
        plt.savefig(savepath,dpi=200)
        plt.show()
    else:
        plt.show()

def plot_significance_new(data,var,hue,ax,palette='mako',drug=False,test ='Mann-Whitney'):


    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    boxes = sns.violinplot(data=data,
                            x=hue,
                            y=var,
                            width=.6, 
                            palette=palette,
                            ax=ax)

    # sns.stripplot(x=hue, y=var , data=data,           
    #             size=3, color=".4", linewidth=0,ax=ax) 
    ax.set_xlabel('class',fontdict={'fontsize':20})
    ax.set_ylabel(var,fontdict={'fontsize':20})

    for box,col in zip(boxes.patches,['blue','crimson','teal']):
        mybox1 = box

        # Change the appearance of that box
        if drug:
            mybox1.set_facecolor('white')
            mybox1.set_edgecolor(col)
        else:
            mybox1.set_facecolor(col)
            mybox1.set_edgecolor('black')

        mybox1.set_linewidth(3)

    pairs = np.unique(data[hue])
    pairs = [i for i in combinations(pairs,2)]

    annotator = Annotator(ax, 
                          pairs, 
                          data=data, 
                          x=hue, 
                          palette=palette, 
                          y=var)
    
    annotator.configure(test=test, text_format='star', loc='inside')
    annotator.apply_and_annotate()  
    plt.show()

def plot_waveforms(df, save=False, savepath=None,c_list=None):

    for i in set(df.labels_wave):
        print(i)
        t = np.arange(0,6,1/20)
        fig,ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        # data_wave_FN_acsf[data_wave_FN_acsf.labels_wave==i]['half_width']
        wave_i = np.vstack(df[df.labels_wave==i]['waveform']).T
        plt.plot(t,wave_i[-int(8*20):int(-2*20),:],c=c_list[i],alpha=0.2)
        plt.plot(t,np.mean(wave_i,axis=1)[-int(8*20):int(-2*20)],c='black',linewidth=2,label=i)
        plt.xticks([])
        plt.yticks([])
        if save:
            plt.savefig(savepath+'wave_'+str(i)+'.pdf',dpi=200)
        plt.show()

def plot_radar(data,cols,labels,figsize=(6, 6),lims=None,palette=None,logscale=True,save=False,savepath=None,label_font_size=12,linewidth=0.5):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.yaxis.grid(color='lightgray', linestyle='--')
    zero_line_color = 'red'

    cols_ = cols
    data_norm = normalize(data[cols_].to_numpy(),axis=0)
    # Data
    categories = cols_
    for i in list(set(labels)):
        idx = labels==i
        values_1 = data_norm[idx,:]

        # Number of variables we're plotting.
        num_vars = len(cols_)
        # Compute angle of each axis.
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop".
        # Append the start value to the end.
        values_1 = np.insert(values_1,[values_1.shape[1]],values_1[:,:1],axis=1)
        angles += angles[:1]

        # Create the figure.

        # Draw one axe per variable and add labels.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories,fontsize=label_font_size,)
        for label, angle in zip(ax.get_xticklabels(), angles):
            x, y = label.get_position()
            label.set_position((x, y-0.2 ))  # Adjust the value to move labels further out
        # Draw ylabels.
        if logscale:
            ax.set_rscale('log')
            ax.plot(np.linspace(0, 2*np.pi, 100), np.zeros(100), color=zero_line_color, linewidth=linewidth, zorder=3)
            ax.plot(angles, values_1.T, linewidth=1,c=palette[i], linestyle='solid',alpha=0.1 )
            ax.plot(angles, np.mean(values_1.T,axis=1), linewidth=3,c=palette[i], linestyle='solid',alpha=.8 )
            ax.set_ylim(lims)

        else:
            # ax.set_rscale('log')
            ax.plot(np.linspace(0, 2*np.pi, 100), np.zeros(100), color=zero_line_color, linewidth=linewidth, zorder=3)
            ax.plot(angles, values_1.T, linewidth=1,c=palette[i], linestyle='solid',alpha=0.1 )
            ax.plot(angles, np.mean(values_1.T,axis=1), linewidth=3,c=palette[i], linestyle='solid',alpha=.8 )
            ax.set_ylim(lims)

    if save:
        plt.savefig(savepath,dpi=200)
    else:
        plt.show()

def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1), np.var(d2)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s        