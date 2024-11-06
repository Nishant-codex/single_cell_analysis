
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

def return_confusion_matrix_(df1,df2,label1_name,label2_name,vmin=0,vmax=100,figsize =[12,5],shuffle = False,save=False,savepath=None,cmap='BrBG_r'):
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
            mat_orig[i,data_[0]] =(data_[1]/np.sum(data_[1]))*100 

        mat_sh = np.zeros((len(set(df['label1_sh'])),len(set(df['label2_sh']))))

        for i in np.unique(df.label1_sh):
            data_ = np.unique(df[df.label1_sh==i]['label2_sh'],return_counts=True)
            mat_sh[i,data_[0]] =(data_[1]/np.sum(data_[1]))*100 


        sns.heatmap(mat_orig-mat_sh,cmap=cmap,annot=True,ax=ax1,vmin=vmin,vmax=vmax) 
        if save:
            plt.savefig(savepath,dpi=300)
        else:
            plt.show()

    else:
        df = pd.DataFrame(columns=['label1','label2'])
        # df['exp_name1'] = df1.exp_name
        # df['exp_name2'] = df2.exp_name
        df['label1'] = np.array(df1[label1_name])
        df['label2'] = np.array(df2[label2_name])

        mat = np.zeros((len(set(df1[label1_name])),len(set(df2[label2_name]))))

        for i in np.unique(df.label1):
            data_ = np.unique(df[df.label1==i]['label2'],return_counts=True)
            mat[i,data_[0]] =(data_[1]/np.sum(data_[1]))*100 

        sns.heatmap(mat,cmap=cmap,annot=True,vmin=vmin,vmax=vmax) 

def plot_cosine_mat(data1,data2,label1,label2):
    cosine_mat = np.zeros((len(set(label1)),len(set(label2))))
    sim_data = cosine_similarity(data1,data2)
    for i in set(label1):
        for j in set(label2):
            idx_FN = label1==i
            idx_SH = label2==j
            cosine_mat[i,j] = np.mean(sim_data[:,idx_FN][idx_SH])
    return cosine_mat

def plot_radar(data,cols,labels,figsize=(6, 6),lims=None,palette=None,logscale=True,save=False,savepath=None,):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

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
        plt.xticks(angles[:-1], categories,fontsize=12,)
        for label, angle in zip(ax.get_xticklabels(), angles):
            x, y = label.get_position()
            label.set_position((x, y-0.2 ))  # Adjust the value to move labels further out
        # Draw ylabels.
        if logscale:
            ax.set_rscale('log')
            ax.plot(angles, values_1.T, linewidth=1,c=palette[i], linestyle='solid',alpha=0.1 )
            ax.plot(angles, np.mean(values_1.T,axis=1), linewidth=3,c=palette[i], linestyle='solid',alpha=.8 )
            ax.set_ylim(lims)

        else:
            # ax.set_rscale('log')

            ax.plot(angles, values_1.T, linewidth=1,c=palette[i], linestyle='solid',alpha=0.1 )

            ax.plot(angles, np.mean(values_1.T,axis=1), linewidth=3,c=palette[i], linestyle='solid',alpha=.8 )
            ax.set_ylim(lims)

    if save:
        plt.savefig(savepath,dpi=200)
    else:
        plt.show()