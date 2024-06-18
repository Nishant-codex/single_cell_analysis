import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
from sknetwork.clustering import Louvain,get_modularity
import umap

def return_confusion_matrix(df1,df2,label1_name,label2_name,figsize =[12,5] ,shuffle = False,save=False,savepath=None,cmap='BrBG_r'):
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


        sns.heatmap(mat_orig-mat_sh,cmap=cmap,annot=True,ax=ax1,vmin=0,vmax=100) 
        # sns.heatmap(mat_sh,cmap=cmap,annot=True,ax=ax2,vmin=0,vmax=100) 
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

        sns.heatmap(mat,cmap=cmap,annot=True,vmin=0,vmax=100) 

def plot_clust_res(all_data):
    f, ax1 = plt.subplots(figsize=[6,5])
    for feat in all_data.keys():
        modularity_dict, n_clusts_dict = all_data[feat].values()
        resolution_list = np.linspace(0,5,11)

        avg_n_clusts = []
        for k in list(n_clusts_dict.keys()):
            avg_n_clusts.append(np.mean(n_clusts_dict[k]))
            
        std_n_clusts = []
        for k in list(n_clusts_dict.keys()):
            std_n_clusts.append(np.std(n_clusts_dict[k]))
            
        std_modularity = []
        for k in list(modularity_dict.keys()):
            std_modularity.append(np.std(modularity_dict[k]))
            
        avg_modularity = []
        for k in list(modularity_dict.keys()):
            avg_modularity.append(np.mean(modularity_dict[k]))



        ax1.errorbar(resolution_list,avg_modularity,yerr=std_modularity,
                    marker='o', fillstyle='full', markerfacecolor='w', 
                    linewidth=1, markeredgewidth=1)
        ax1.set_ylabel('Modularity Score')
        ax1.set_xlabel('Resolution Parameter',fontsize=12)
        ax1.set_xlim([0,5])
        ax1.set_xticks([0,2,4,6])
        ax1.yaxis.label.set_color('#5c95ff')
        ax1.tick_params(axis='y',colors='#5c95ff')
        ax1.set_ylim(0,1.0)
        ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
        # ax1.set_yticklabels([0.0,'',0.2,'',0.4,'',0.6,'',0.8,'',1.0],fontsize=12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_color('#f87575')
        ax1.spines['left'].set_color('#5c95ff')

        ax2 = ax1.twinx()
        ax2.errorbar(resolution_list[1:],avg_n_clusts[1:],yerr=std_n_clusts[1:],c='r', marker='o', fillstyle='full', markerfacecolor='w', linewidth=1, markeredgewidth=1)
        ax2.set_ylabel('Number of Clusters',fontsize=12,c='#f87575')
        # ax2.spines['left'].set_color('b')
        ax2.tick_params(axis='y',colors='#f87575')
        ax2.set_ylim([0,22])
        ax2.set_yticks([0,4,8,12,16,18,20]);
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_color('#f87575')
        ax2.spines['left'].set_color('#5c95ff')
    plt.show()

def find_optimum_res(data,save=False,savepath=None):
    resolution_list = np.linspace(0,5,11)
    modularity_dict = {}
    n_clusts_dict = {}
    #Louvain Clustering Parameters
    random_state = 42
    full_data = data
    # BLUE COLOR
    BlueCol = '\033[94m'
    subsets=[0.8]
    import random 
    for res in resolution_list:
        print("\n" + BlueCol + str(res))
        for frac in subsets:
            rand_list = []
            n_clusts = []
            for i in list(range(1,25)):
                reducer_rand_test = umap.UMAP(n_neighbors = 20, 
                                        min_dist=0.1, 
                                        # random_state=random.randint(1,100000),
                                        n_jobs=-1
                                        )
                
                idx1 = np.zeros(len(full_data),dtype=bool)
                rows_to_keep = np.random.randint(0,int(len(full_data)),int(len(full_data)*frac))
                idx1[rows_to_keep] = True
                part_data = full_data[idx1,:]

                rand_data = np.vstack(np.random.permutation(part_data))
                mapper = reducer_rand_test.fit(rand_data)
                embedding_rand_test = reducer_rand_test.transform(rand_data)

                umap_df_rand_test = pd.DataFrame(embedding_rand_test, columns=('x', 'y'))
                louvain = Louvain(resolution=res,random_state=random_state)
                adjacency = mapper.graph_
                labels_exc = louvain.fit_predict(adjacency)
                clustering_solution = labels_exc
                modularity= get_modularity(adjacency,labels_exc)
                rand_list.append(modularity)
                n_clusts.append(len(set(clustering_solution)))
            modularity_dict.update({str(res): rand_list})
            n_clusts_dict.update({str(res): n_clusts})


    resolution_list = np.linspace(0,5,11)

    avg_n_clusts = []
    for k in list(n_clusts_dict.keys()):
        avg_n_clusts.append(np.mean(n_clusts_dict[k]))
        
    std_n_clusts = []
    for k in list(n_clusts_dict.keys()):
        std_n_clusts.append(np.std(n_clusts_dict[k]))
        
    std_modularity = []
    for k in list(modularity_dict.keys()):
        std_modularity.append(np.std(modularity_dict[k]))
        
    avg_modularity = []
    for k in list(modularity_dict.keys()):
        avg_modularity.append(np.mean(modularity_dict[k]))

    f, ax1 = plt.subplots(figsize=[3,2.5])

    ax1.errorbar(resolution_list,avg_modularity,yerr=std_modularity,
                c = '#5c95ff', marker='o', fillstyle='full', markerfacecolor='w', 
                linewidth=1, markeredgewidth=1)
    ax1.set_ylabel('Modularity Score')
    ax1.set_xlabel('Resolution Parameter',fontsize=12)
    ax1.set_xlim([0,6])
    ax1.set_xticks([0,2,4,6])
    ax1.yaxis.label.set_color('#5c95ff')
    ax1.tick_params(axis='y',colors='#5c95ff')
    ax1.set_ylim(0,1.0)
    ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    # ax1.set_yticklabels([0.0,'',0.2,'',0.4,'',0.6,'',0.8,'',1.0],fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_color('#f87575')
    ax1.spines['left'].set_color('#5c95ff')

    ax2 = ax1.twinx()
    ax2.errorbar(resolution_list[1:],avg_n_clusts[1:],yerr=std_n_clusts[1:],c = '#f87575', marker='o', fillstyle='full', markerfacecolor='w', linewidth=1, markeredgewidth=1)
    ax2.set_ylabel('Number of Clusters',fontsize=12,c='#f87575')
    # ax2.spines['left'].set_color('b')
    ax2.tick_params(axis='y',colors='#f87575')
    ax2.set_ylim([0,18])
    ax2.set_yticks([0,4,8,12,16]);
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('#f87575')
    ax2.spines['left'].set_color('#5c95ff')
    if save:
        plt.savefig(savepath,dpi=300)
    else:
        plt.show()

def find_optimum_res_with_cols(data:np.ndarray,cols:list):
    
    dict_with_cols_excluded = {}
    for i in range(len(cols)):
        idx2 = ~np.zeros_like(cols,dtype=bool)
        idx2[i] =0 
        cols_  = np.array(cols)[~idx2][0]  
        print(cols_)
        resolution_list = np.linspace(0,5,11)
        modularity_dict = {}
        n_clusts_dict = {}
        #Louvain Clustering Parameters
        random_state = 42
        full_data = data
        # BLUE COLOR
        BlueCol = '\033[94m'
        subsets=[100]
        import random 
        for res in resolution_list:
            print("\n" + BlueCol + str(res))
            for frac in subsets:
                rand_list = []
                n_clusts = []
                for i in list(range(1,25)):
                    reducer_rand_test = umap.UMAP(n_neighbors = 20, 
                                            min_dist=0.1, 
                                            # random_state=random.randint(1,100000),
                                            n_jobs=8
                                            )
                    idx1 = np.zeros(len(full_data),dtype=bool)
                    rows_to_keep = np.random.randint(0,len(full_data),int(len(full_data)*frac))
                    idx1[rows_to_keep] = True
                    part_data = full_data[idx1,:]
                    part_data = part_data[:,idx2]
                    
                    rand_data = np.vstack(np.random.permutation(part_data))
                    mapper = reducer_rand_test.fit(rand_data)
                    embedding_rand_test = reducer_rand_test.transform(rand_data)

                    umap_df_rand_test = pd.DataFrame(embedding_rand_test, columns=('x', 'y'))
                    louvain = Louvain(resolution=res,random_state=random_state)
                    adjacency = mapper.graph_
                    labels_exc = louvain.fit_predict(adjacency)
                    clustering_solution = labels_exc
                    modularity= get_modularity(adjacency,labels_exc)
                    rand_list.append(modularity)
                    n_clusts.append(len(set(clustering_solution)))
                modularity_dict.update({str(res): rand_list})
                n_clusts_dict.update({str(res): n_clusts})

        dict_with_cols_excluded[cols_] = {'mod_dict':modularity_dict,'n_clust_dict':n_clusts_dict}

    return dict_with_cols_excluded

