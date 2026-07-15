from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from multicor_fa import mcfa_model
import seaborn as sns
import scipy.stats as stats

'''
Utility functions for MCFA analysis.

'''
# Columns for the ephys features dataframe
cols_ephys = ['current_at_first_spike', 'ap_count', 'fr', 'inst_fr',
       'time_to_first_spike', 'mean_isi', 'median_isi', 'max_isi', 'min_isi',
       'first_thr', 'mean_thr', 'median_thr', 'min_thr', 'max_thr',
       'mean_width', 'median_width', 'max_width', 'min_width',
       'mean_amplitude', 'median_amplitude', 'min_amplitude', 'max_amplitude',]
# Columns for the biological features dataframe
cols_bio = ['tau_m (ms)', 'C (nF)', 'gl (nS)', 'El (mV)', 'Vr (mV)', 'Vt* (mV)', 'DV (mV)',]

def return_neuromod_df(df:pd.DataFrame,cond:list,df_type:str,cols=None,sample=None):
    '''
    Function to return a dataframe based on the condition and type of data requested.
    parameters:
    -----------
    df : pd.DataFrame
        Dataframe from which to select data.
    cond : list
        List of conditions to filter the data.
    df_type : str
        Type of dataframe to return ('sta', 'bio', 'eta', 'ephys').
    cols : list, optional
        List of columns to include in the returned dataframe.
    sample : int, optional
        Number of samples to include in the returned dataframe.

    Returns:
    --------
    pd.DataFrame
        Filtered and normalized dataframe based on the specified parameters.
    '''
    if df_type=='sta':
        df_sta = pd.DataFrame(columns=np.arange(np.vstack(df['sta'].to_numpy()).shape[1]))
        df_sta[df_sta.columns] = normalize(np.vstack(df[df.cond.isin(cond)]['sta'].to_numpy()))
        if sample is not None:
            df_sta = df_sta.sample(sample)
            return df_sta
        else:
            return df_sta
    
    elif df_type=='bio':
        df_bio = pd.DataFrame(columns=cols)
        df_bio[cols] = df[df.cond.isin(cond)][cols].to_numpy(dtype=np.float32)
        if sample is not None:
            df_bio = df_bio.sample(sample)
            return df_bio
        else:        
            return df_bio
    
    elif df_type=='eta':    
        df_eta_exc_all = pd.DataFrame(columns=np.arange(10759))
        df_eta_exc_all[df_eta_exc_all.columns]  = normalize(np.float32(np.vstack(df[df.cond.isin(cond)]['eta'].to_numpy())))
        if sample is not None:
            df_eta_exc_all = df_eta_exc_all.sample(sample)
            return df_eta_exc_all
        else:
            return df_eta_exc_all
             
    elif df_type=='ephys':    
        df_ephys = pd.DataFrame(columns=cols)
        df_ephys[cols] = df[df.cond.isin(cond)][cols].to_numpy(dtype=np.float32)
        if sample is not None:
            df_ephys = df_ephys.sample(sample)
            return df_ephys
        else:
            return df_ephys
        
def remove_duplicates(df:pd.DataFrame):
    '''
    Function to remove duplicate entries from a dataframe based on the 'exp_name' column.
    parameters:
    -----------
    df : pd.DataFrame
        Dataframe from which duplicates need to be removed.
    Returns:
    --------
    pd.DataFrame
        Dataframe with duplicates removed based on 'exp_name' column.   
    
    '''
    return df[~df.exp_name.duplicated()]

def bootstrap_MCFA(datarame, iters = 100, sample_size = 10,verbose=False):
    """     
    Function to run bootstrap MCFA on the data.
    Parameters  
    ----------      
    datarame : pd.DataFrame
        Dataframe containing the data to run bootstrap on.
    iters : int, optional       
        Number of iterations to run the bootstrap. The default is 100.
    sample_size : int, optional
        Number of samples to take from the data. The default is 10.
    verbose : bool, optional
        If True, prints the shape of the data at each iteration. The default is False.
    Returns
    -------                     
    all_pds : list
        List of dataframes containing the variance explained for each iteration.
    """
    np.random.seed(0)

    all_pds = []
    for i in range(iters):
        try:
            rand_inds = np.random.randint(0,len(datarame[0]),sample_size)

            Y_inh = {
                "ephys": datarame[0][cols_ephys].reset_index(drop='index').loc[rand_inds,:].reset_index(drop='index'), 
                "bio": datarame[1][cols_bio].reset_index(drop='index').loc[rand_inds,:].reset_index(drop='index'), 
                "sta": datarame[2].reset_index(drop='index').loc[rand_inds,:].reset_index(drop='index'),
                'eta': datarame[3].reset_index(drop='index').loc[rand_inds,:].reset_index(drop='index')
                }
            for key in Y_inh.keys():
                  print(Y_inh[key].shape)
            mcfa_res_inh = mcfa_model.fit(Y_inh,k=[1,1,1,1],n_pcs= [2,2,2,2],verbose=verbose,center=True) 

            mcfa_res_df_inh = pd.concat([mcfa_res_inh.Z] + list(mcfa_res_inh.X.values()), axis=1)



            var_exp_private_inh = {mode: ve_X.sum() for mode, ve_X in mcfa_res_inh.var_exp_X.items()}
            var_exp_shared_inh = mcfa_res_inh.var_exp_Z.sum()
            var_exp_totals_inh = pd.DataFrame({'Shared': var_exp_shared_inh, 'Specific': var_exp_private_inh})
            var_exp_totals_inh['Total'] = var_exp_totals_inh['Shared'] + var_exp_totals_inh['Specific']
            var_exp_normed_inh = (mcfa_res_inh.var_exp_Z/var_exp_totals_inh['Total']).T
            var_exp_totals_inh['Residual'] = 1 - var_exp_totals_inh['Total']
            var_exp_totals_inh = var_exp_totals_inh.drop('Total', axis=1)
            all_pds.append(var_exp_totals_inh)
        except:
            print(f"Error in iteration {i}")
    return all_pds

def perform_MCFA(cond=None,save=False, savepath = None):
    """_summary_

    Args:
        cond (_type_, optional): _description_. Defaults to None.
        save (bool, optional): _description_. Defaults to False.
        savepath (_type_, optional): _description_. Defaults to None.
    """
    if cond==None:
        raise 'Provide a drug condition'
    drug = [cond]

    ephys_drug_exc = return_neuromod_df(ephys_exc_all,cond=drug,cols=cols_ephys,df_type='ephys')
    print('exc', len(ephys_drug_exc))
    sta_drug_exc   = return_neuromod_df(sta_exc_all,cond=drug,df_type='sta')
    bio_drug_exc   = return_neuromod_df(bio_exc_all,cond=drug,df_type='bio',cols=cols_bio)
    eta_drug_exc   = return_neuromod_df(bio_exc_all,cond=drug,df_type='eta')

    ephys_drug_inh = return_neuromod_df(ephys_inh_all,cond=drug,cols=cols_ephys,df_type='ephys')
    print('inh', len(ephys_drug_inh))
    sta_drug_inh   = return_neuromod_df(sta_inh_all,cond=drug,df_type='sta')
    bio_drug_inh   = return_neuromod_df(bio_inh_all,cond=drug,df_type='bio',cols=cols_bio)
    eta_drug_inh   = return_neuromod_df(bio_inh_all,cond=drug,df_type='eta')




    Y_inh = {"ephys": ephys_drug_inh[cols_ephys].reset_index(drop='index'), 
            "bio"  : bio_drug_inh[cols_bio].reset_index(drop='index'), 
            "sta"  : sta_drug_inh,
            'eta'  : eta_drug_inh}

    Y_exc = {"ephys": ephys_drug_exc[cols_ephys].reset_index(drop='index'), 
            "bio"  : bio_drug_exc[cols_bio].reset_index(drop='index'), 
            "sta"  : sta_drug_exc,
            'eta'  : eta_drug_exc}
            



    mcfa_res_inh_d1 = mcfa_model.fit(Y_inh,center =True, k=[1,1,1,1],n_pcs= [2,2,2,2],verbose=False) 
    mcfa_res_exc_d1 = mcfa_model.fit(Y_exc,center =True, k=[1,1,1,1],n_pcs= [2,2,2,2],verbose=False) 



    var_exp_private_exc = {mode: ve_X.sum() for mode, ve_X in mcfa_res_exc_d1.var_exp_X.items()}
    var_exp_shared_exc = mcfa_res_exc_d1.var_exp_Z.sum()
    var_exp_totals_exc_d1 = pd.DataFrame({'Shared': var_exp_shared_exc, 'Specific': var_exp_private_exc})
    var_exp_totals_exc_d1['Total'] = var_exp_totals_exc_d1['Shared'] + var_exp_totals_exc_d1['Specific']
    var_exp_normed_exc = (mcfa_res_exc_d1.var_exp_Z/var_exp_totals_exc_d1['Total']).T
    var_exp_totals_exc_d1['Residual'] = 1 - var_exp_totals_exc_d1['Total']
    var_exp_totals_exc_d1 = var_exp_totals_exc_d1.drop('Total', axis=1)


    var_exp_private_inh = {mode: ve_X.sum() for mode, ve_X in mcfa_res_inh_d1.var_exp_X.items()}
    var_exp_shared_inh = mcfa_res_inh_d1.var_exp_Z.sum()
    var_exp_totals_inh_d1 = pd.DataFrame({'Shared': var_exp_shared_inh, 'Specific': var_exp_private_inh})
    var_exp_totals_inh_d1['Total'] = var_exp_totals_inh_d1['Shared'] + var_exp_totals_inh_d1['Specific']
    var_exp_normed_inh = (mcfa_res_inh_d1.var_exp_Z/var_exp_totals_inh_d1['Total']).T
    var_exp_totals_inh_d1['Residual'] = 1 - var_exp_totals_inh_d1['Total']
    var_exp_totals_inh_d1 = var_exp_totals_inh_d1.drop('Total', axis=1)


    var_exp_totals_exc_d1['dataset'] = var_exp_totals_exc_d1.index
    var_exp_totals_exc_d1 = var_exp_totals_exc_d1.melt(id_vars=['dataset'], var_name='Space', value_name='Variance explained')

    var_exp_totals_inh_d1['dataset'] = var_exp_totals_inh_d1.index
    var_exp_totals_inh_d1 = var_exp_totals_inh_d1.melt(id_vars=['dataset'], var_name='Space', value_name='Variance explained')

#     var_exp_totals_exc_d1 = var_exp_totals_exc_d1.reindex(['ephys','bio','eta','sta'])
#     var_exp_totals_inh_d1 = var_exp_totals_inh_d1.reindex(['ephys','bio','eta','sta'])

    var_exp_totals_exc_d1 = var_exp_totals_exc_d1.reindex([0,1,3,2,4,5,7,6,8,9,11,10])
    var_exp_totals_inh_d1 = var_exp_totals_inh_d1.reindex([0,1,3,2,4,5,7,6,8,9,11,10])
    print(var_exp_totals_exc_d1)
    print(var_exp_totals_inh_d1)
    # sns.set_context('talk')
    var_exp_totals_exc_d1.to_csv(savepath+cond+'_var_exp_exc.csv')
    var_exp_totals_inh_d1.to_csv(savepath+cond+'_var_exp_inh.csv')
    fig,ax_exc = plt.subplots(figsize=(7, 5))
    var_exp_totals_exc_d1['Space'] = pd.Categorical(var_exp_totals_exc_d1['Space'], ['Residual', 'Specific', 'Shared'])
    sns.histplot(var_exp_totals_exc_d1, y='dataset', hue='Space', weights='Variance explained',
                multiple='stack', palette='icefire',ax =ax_exc)
    ax_exc.set(title='Percent variance explained', xlabel=None, ylabel=None)
    ax_exc.set_xlim(0, 1)
    ax_exc.tick_params(left=False)
    ax_exc.set_yticklabels=['ephys','bio','eta','sta']
    if save:
        plt.savefig(savepath+cond+'_exc.pdf',dpi=200)
    plt.show() 

    # sns.set_context('talk')
    
    fig,ax_inh = plt.subplots(figsize=(7, 5))    
    var_exp_totals_inh_d1['Space'] = pd.Categorical(var_exp_totals_inh_d1['Space'], ['Residual', 'Specific', 'Shared'])
    sns.histplot(var_exp_totals_inh_d1, y='dataset', hue='Space', weights='Variance explained',
                multiple='stack', palette='icefire',ax = ax_inh)
    ax_inh.set(title='Percent variance explained', xlabel=None, ylabel=None)
    ax_inh.set_xlim(0, 1)
    ax_inh.tick_params(left=False)
    ax_inh.set_yticklabels=['ephys', 'bio','eta','sta']
    if save:
        plt.savefig(savepath+cond+'_inh.pdf',dpi=200)
    plt.show() 
    return var_exp_totals_exc_d1,var_exp_totals_inh_d1

def plot_bootstraped_data(data,savepath=None,cond=None,save=False,ei_type='exc'):
    '''
    Function to plot the bootstraped data.
    parameters: 
    ----------- 
    data : pd.DataFrame
        Dataframe containing the bootstraped data.
    savepath : str, optional
        Path where the plot will be saved.
    cond : str, optional
        Condition for saving the plot.
    save : bool, optional
        Whether to save the plot.
    ei_type : str, optional
        Type of data to plot ('exc' or 'inh').
    returns:
    --------
    None
    '''
    fig,ax = plt.subplots(figsize=(7, 5))       
    sns.histplot(data, y='dataset', hue='Space',hue_order= ['Residual','Specific','Shared'], weights='Variance explained',
                multiple='stack', palette='icefire',ax =ax)
    ax.set(title='Percent variance explained', xlabel=None, ylabel=None)
    ax.set_xlim(0, 1)
    ax.tick_params(left=False)
    ax.set_yticklabels=['ephys','bio','eta','sta']
    if save:
        plt.savefig(savepath+cond+'_'+ei_type+'.pdf',dpi=200)
    plt.show()


def add_fscore(data_ago,data_acsf,n_agonist,n_control):
    data_ago['F'] = np.nan
    data_ago['p_value'] = np.nan
    for space in set(data_ago.Space):
        for dataset in set(data_ago.dataset):
            var_d1   = data_ago[data_ago.dataset==dataset][data_ago.Space==space]['Variance explained'].to_numpy()
            var_acsf = data_acsf[data_acsf.dataset==dataset][data_acsf.Space==space]['Variance explained'].to_numpy()
            F,p_val = F_test(var_d1,var_acsf,n_agonist,n_control)

            data_ago.loc[(data_ago.dataset==dataset) & (data_ago.Space==space),'F'] = F
            data_ago.loc[(data_ago.dataset==dataset) & (data_ago.Space==space),'p_value'] = p_val
    return data_ago

def F_test(variance_agonist,variance_control,n_agonist,n_control):

    # Determine F-statistic
    F = max(variance_control, variance_agonist) / min(variance_control, variance_agonist)

    # Degrees of freedom
    df1 = n_control - 1
    df2 = n_agonist - 1

    # Compute p-value
    p_value = 2 * (1 - stats.f.cdf(F, df1, df2))

    # Output results
    # print(f"F-statistic: {F}")
    # print(f"P-value: {p_value}")
    return F,p_value