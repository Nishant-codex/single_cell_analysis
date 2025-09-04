
# parameters{1} = 'Current at first spike'; @
# parameters{3} = 'AP Count - Start'; @
# parameters{5} = 'Absolute Firing Rate - Start'; @
# parameters{7} = 'Inst Firing Rate - Start'; @
# parameters{9} = 'AP Latency First - Start'; @
# parameters{11} = 'AP Latency Last - Start'; @
# parameters{13} = 'AP Latency Window - Start';
# parameters{15} = 'AP Latency Compression - Start';
# parameters{17} = 'ISI Min - Start'; @
# parameters{19} = 'ISI Max - Start'; @
# parameters{21} = 'ISI Mean - Start'; @
# parameters{23} = 'ISI Median - Start'; @
# parameters{25} = 'ISI Adapt Rate - Start'; 
# parameters{27} = 'AP Threshold First - Start'; @
# parameters{29} = 'AP Threshold Last - Start'; @ 
# parameters{31} = 'AP Threshold Min - Start'; @ 
# parameters{33} = 'AP Threshold Max - Start'; @
# parameters{35} = 'AP Threshold Mean - Start'; @
# parameters{37} = 'AP Threshold Median - Start'; @
# parameters{39} = 'AP Threshold AdaptationRate - Start';
# parameters{41} = 'AP HalfWidth First - Start'; @
# parameters{43} = 'AP HalfWidth Last - Start'; @
# parameters{45} = 'AP HalfWidth Min - Start'; @
# parameters{47} = 'AP HalfWidth Max - Start'; @
# parameters{49} = 'AP HalfWidth Mean - Start'; @
# parameters{51} = 'AP HalfWidth Median - Start'; @
# parameters{53} = 'AP HalfWidth AdaptationRate - Start';
# parameters{55} = 'AP Amplitude First - Start'; @
# parameters{57} = 'AP Amplitude Last - Start'; @
# parameters{59} = 'AP Amplitude Min - Start'; @
# parameters{61} = 'AP Amplitude Max - Start'; @
# parameters{63} = 'AP Amplitude Mean - Start'; @
# parameters{65} = 'AP Amplitude Median - Start'; @
# parameters{67} = 'AP Amplitude AdaptationRate - Start';
#%%
import neo
from quantities import *
from elephant import statistics
from elephant.kernels import GaussianKernel
from elephant.statistics import isi, cv
from elephant.statistics import time_histogram, instantaneous_rate
from elephant import sta

class EphysSet:

    def __init__(self,data,cond,exp_name,trialnr):

        self.data = data
        self.cond = cond
        self.exp_name = exp_name
        self.trialnr = trialnr
        self.V = self.data['membrane_potential']
        self.I = self.data['input_current']
        self.dt = 1/20 

    def remove_nan(self,data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(data)
        data_ = data[ind]
        return data_

    def rolling_avg(self,data):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        start = 0
        length = len(data)//10
        end = len(data)
        finish = False
        avg = []

        if length == 0:
            return np.mean(data[start:end])
        else:
            while finish != True:
                if start+length < end:
                    avg.append(np.mean(data[start:start+length]))
                    start += length
                else:
                    avg.append(np.mean(data[start:end]))
                    finish = True

            return avg

    def return_quant_divided_by_time(self, divisions, quant):
        total_duration = len(self.data['membrane_potential'])/20
        quantity = quant
        time_ranges = np.arange(0,total_duration+1,total_duration//divisions)
        spike_times = self.data['spikeindices']/20
        time_divided_spiketimes = [np.array(spike_times[np.where(np.logical_and(spike_times>time_ranges[i],spike_times<=time_ranges[i+1] ))[0]]*20,dtype=np.int32)
        for i in range(len(time_ranges)-1)]
        last = 0
        vals_divided_by_time = []
        for i in time_divided_spiketimes:
            vals_divided_by_time.append(quantity[last:last+len(i)])
            last = len(i)
        return vals_divided_by_time  

    def fano_factor(self,divisions):
        total_duration = len(self.data['membrane_potential'])/20
        time_ranges = np.arange(0,total_duration+1,total_duration//divisions)
        spike_times = self.data['spikeindices']/20
        spike_counts = [len(np.array(spike_times[np.where(np.logical_and(spike_times>time_ranges[i],spike_times<=time_ranges[i+1] ))[0]]*20,dtype=np.int32))
        for i in range(len(time_ranges)-1)]

        return np.var(spike_counts)/np.mean(spike_counts)  

    def cv(self,data):
        return np.std(data)/np.mean(data)
    
    def return_variance_shift(self, shift, window):
        shift = shift*20 
        window = window*20
        run =True
        start = 0
        means = []
        var = [] 
        while run:
            if start+window>len(self.I):
                run=False
            else:
                I_samp = self.I[start:(start+window)]
                means.append(np.mean(I_samp))
                var.append(np.var(I_samp))
                start+=shift
        return means,var 
    
    #spike shape
    def current_at_first_spike(self):

        firstspike_ind = self.data['spikeindices'][0]
        return self.I[firstspike_ind]
    
    def get_Vm(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        Vm = []
        V = self.data['membrane_potential']
        thr = self.data['thresholds']
        thr_ind = self.data['thresholdindices']
        spikes = self.data['spikeindices']
        ind = ~np.isnan(thr)
        spikes = spikes[ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]

        for i, j in zip(thr_ind, thr):
            Vm.append(V[int(i)-70:int(i)+100])
        if return_mean:
            return np.mean(Vm), Vm, np.mean(V)
        else:
            return Vm
    
    def get_dvdt(self,data,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        dvdt_p = []
        dvdt_n = []


        for i in data:
            dv_ = np.diff(i)
            posp = np.where(dv_ > 0)
            dvdt_ind = np.zeros_like(dv_, dtype=bool)
            dvdt_ind[posp] = True
            posp = dvdt_ind
            posn = ~posp
            dvdt_p.append(np.max(dv_[posp]))
            dvdt_n.append(np.min(dv_[posn]))
                      
        if return_mean:
            return np.mean(dvdt_p), np.mean(dvdt_n), np.median(dvdt_p), np.median(dvdt_n),np.max(dvdt_p), np.max(dvdt_n),np.min(dvdt_p), np.min(dvdt_n),np.std(dvdt_p), np.std(dvdt_n)
                
        else:
            return dvdt_p, dvdt_n
    
    def get_AP_width(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        thr_ind = self.data['thresholdindices']
        thr = self.data['thresholds']
        ind = ~np.isnan(thr_ind)
        spks = self.data['spikeindices'][ind]
        thr = thr[ind]
        thr_ind = thr_ind[ind]
        V = self.data['membrane_potential']
        peak = 0
        width = []
        for i, j in zip(spks, thr_ind):
            try:
                spike_wf = V[int(j):int(j)+100]
                # plt.plot(spike_wf)
                left = V[int(j):i]
                right_ind = i-int(j)
                right = spike_wf[right_ind:]
                half_height = V[i]/2
                left_first = np.where(left <= half_height)
                right_first = np.where(right <= half_height)
                width.append(((int(i-j)+right_first[0][0]+1)-(left_first[0][-1]))/40)
            except:
                pass
        if return_mean:
            return np.mean(width),np.median(width),np.max(width),np.min(width),np.std(width),np.min(width)/np.max(width),return_quant_divided_by_time(10,width)
        else:
            return width
        
    def get_AP_peak(self,spike_waves,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_v = []
        for i in spike_waves:
            max_v.append(np.max(i))
        if return_mean:
            return np.mean(max_v),np.median(max_v),np.max(max_v),np.min(max_v),np.std(max_v),np.min(max_v)/np.max(max_v),return_quant_divided_by_time(10,max_v)
        else:
            return max_v

    
    #time related

    def time_to_first_spike(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.data['thresholdindices'][0]*self.dt

    def get_isi(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        isi =np.diff(self.data['spikeindices'][ind]*dt)
        if return_mean:
            return np.mean(isi),np.median(isi),np.max(isi),np.min(isi),np.min(isi)/np.max(isi),np.std(isi),cv(isi),fano_factor(10)
        else:
            return np.diff(self.data['spikeindices'][ind])*dt
    
    def get_firing_rate(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        # def plot_firing_rate(spks,V,sampling_rate,sampling_period,gaussian_kernel_std):
        sampling_rate = 1/20000
        sampling_period = 1
        gaussian_kernel_std = 1
        spiketrain = neo.SpikeTrain(spks, t_stop=len(V)*(sampling_rate), units='s')
        rate = instantaneous_rate(spiketrain, sampling_period=sampling_period*s, kernel=GaussianKernel(gaussian_kernel_std*s))


        if return_mean:        
            return np.mean(self.data['firing_rate'])
        else:
            return self.data['firing_rate']

    def sub_threshold_resistance(self):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        spikes = self.remove_nan(self.data['thresholdindices'])

        V = self.data['membrane_potential'][:int(spikes[0])-100]
        I = self.data['input_current'][:int(spikes[0])-100].reshape((-1, 1))
        
        model = LinearRegression()
        model.fit(I, V)

        R = model.coef_

        return R

    def get_thresholds(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])


        if return_mean:
            return self.data['thresholds'][ind][0], np.mean(self.data['thresholds'][ind]),np.median(self.data['thresholds'][ind]),np.min(self.data['thresholds'][ind]),np.max(self.data['thresholds'][ind])
        else:
            return self.data['thresholds'][ind]
              
    def get_threshold_adaptation(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        ind = ~np.isnan(self.data['thresholds'])
        thrs_avg = [np.nanmean(i) for i in return_quant_divided_by_time(36,self.data['thresholds'])]
        mean,var = return_variance_shift(10000,10000)
        #%todo measure change in variance compared to change in thresholds per s 
        if return_mean:
            return np.mean(np.diff(self.data['thresholds'][ind]))
        else: 
            return np.diff(self.data['thresholds'][ind])

    def hyperpolarized_value(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.min(self.data['membrane_potential'])

    def get_up_down_ratio(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(self.data['Analysis']) > 1 and type(self.data['Analysis']) == list:
            avg_up = []
            avg_down = []
            for i in self.data['Analysis']:
                avg_up.append(i['nup'])
                avg_down.append(i['ndown'])
        else:
            avg_up = self.data['Analysis']['nup']
            avg_down = self.data['Analysis']['ndown']
        return np.nanmean(np.array(avg_up)/np.array(avg_down))

    #biophysics 

    def subthreshold(self, subthreshold=False, plot=False,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_
            subthreshold (bool, optional): _description_. Defaults to False.
            plot (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        V = self.data['membrane_potential']
        I = self.data['input_current']
        spikes = self.data['spikeindices']
        if self.data['input_generation_settings']['tau'] == 250:
            left = 20
            right = 30
        else:
            left = 20
            right = 20
        empty_cell = []
        for i in spikes:
            empty_cell.append([np.arange(i-left, i+right)])
            V1 = V[i-left]
            V2 = V[i+right]
            if V1 != V2:
                div = (V2-V1)/(left+right)
                V[i-left:i+right] = np.arange(V1, V2, div)[:left+right]
            else:
                V[i-left:i+right] = np.ones((1, left+right))*V1
            I1 = I[i-left]
            I2 = I[i+right]
            if I1 != I2:
                divI = (I2-I1)/(left+right)
                I[i-left:i+right] = np.arange(I1, I2, divI)[:left+right]
            else:
                I[i-left:i+right] = np.ones((1, left+right))*I1
        V_ = V
        I_ = I
        tailored_spikes = np.ndarray.flatten(np.array(empty_cell))
        zero_spikes = np.zeros(np.size(V))
        zero_spikes[tailored_spikes] = True
        if subthreshold == True:
            zero_spikes = ~zero_spikes
        if return_mean:
            return np.mean(V_),np.median(V_),np.max(V_),np.min(V_)
        else:
            return V_

    def psth(self,return_mean=True):
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        V_zero = np.zeros_like(self.data['membrane_potential'])
        thr = self.data['thresholdindices']
        ind = ~np.isnan(thr)
        spks = self.data['spikeindices'][ind]
        V_zero[spks] = 1
        count_spk = []
        start = 0
        width = 50000
        end = len(V_zero)
        run = True
        while run:
            if start+width > end:
                count_spk.append(sum(V_zero[start:end]))
                run = False
            else:
                count_spk.append(sum(V_zero[start:start+width]))
                start = start+width
        if return_mean:        
            return np.mean(count_spk)
        else:
            return count_spk

    def get_MI(self,return_mean=True):
        if type(self.data['Analysis']) == list:
            return np.mean([i['FI'] for i in self.data['Analysis']])
        else:
            return self.data['Analysis']['FI']
    
    def get_impedence(self,return_mean=True):
        """_summary_

        Args:
            data (list): _description_

        Returns:
            float: _description_
        """
        I_acsf = self.data['input_current']
        V_acsf = self.data['membrane_potential']
        spk_acsf, V_acsf, I_acsf = return_stiched_spike_train(self.data)
        imp = overdracht_wytse(0.01, I_acsf, V_acsf, 20001, 20001, 1)
        if return_mean:
            return np.mean(imp),np.max(imp)
        else:
            return imp
    
    def sta(self):        

        spks = self.data['spikeindices']*(1/20000)
        V = self.V
        I = self.I
        sampling_rate = 1/20

        spiketrain = neo.SpikeTrain(spks, t_stop=len(V)*(sampling_rate), units='ms')
        signal = neo.AnalogSignal(np.array([I]).T, units='pA',
                                    sampling_rate=20/ms) 
        sta = sta.spike_triggered_average(signal, spiketrain, (-10 * ms, 0 * ms))
        return sta.magnitude   
     
    def capacitance(self):
        V = self.V*1e-3 # V
        I = self.I*1e-12 #pA
        spks = self.data['spikeindices']
        spks_tr =list(spks[(np.where(np.diff(spks)>4000)[0])])
        spks_tr.append(spks[(np.where(np.diff(spks)>4000)[0][-1])+1])
        V_zero  = np.zeros_like(V,dtype=bool)
        dt  = 1/20000
        t_window = .2/dt
        indices = np.int64([np.hstack([np.arange(i,i+(t_window)) for i in spks_tr ])])
        dvdt = np.diff(V)/(dt)
        V_zero[indices]=1
        V_dyn = V[~V_zero]
        I_dyn = I[~V_zero]
        i = -0.075
        V_wind = [i+0.001,i-0.001] #max-min
        V_ =V_dyn[np.where(np.logical_and(V_dyn>=V_wind[1], V_dyn<=V_wind[0]))[0]]
        dv_dt = np.diff(V_)/(dt)
        I_ =I_dyn[np.where(np.logical_and(V_dyn>=V_wind[1], V_dyn<=V_wind[0]))[0]]
        Cms = np.logspace(-1,3,4000)*1e-12 
        Var = [np.var(I_[:-1]/(cm) - dv_dt) for cm in Cms]
        min_Cm = Cms[np.argmin(Var)] 
        return min_Cm
    
    #______________________________________

    def get_ephys_vals(self):
        """_summary_

        Args:
            data_i (dict): _description_

        Returns:
            _type_: _description_
        """

        Vm_avg, Vm, avg_V = self.get_Vm()
        dvdt_p, dvdt_n = self.get_dvdt(Vm)
        resistance = self.sub_threshold_resistance()
        thr = self.get_thresholds()
        adaptation = self.get_threshold_adaptation()
        isi = self.get_isi()
        peak = self.get_AP_peak(Vm)
        peak_adaptation = self.get_AP_peak_adaptation(Vm)
        ap_width = self.get_AP_width()
        hyp_value = self.hyperpolarized_value()
        fist_spike = self.first_spike()
        up_down_ratio = self.get_up_down_ratio()
        isi_adaptation = self.isi_adaptation_index()
        thr_adp_ind = self.threshold_adaptation_index()
        psth = self.psth()
        mi = self.get_MI()
        int_fr = self.get_inst_fr()
        fr = self.get_firing_rate()
        sub_thr = self.subthreshold()
        spk_fr_adp = self.spike_frequency_adaptation()
        imp = self.get_impedence()

        ephys_data =      [Vm, 
                           Vm_avg, #
                            dvdt_p,
                            dvdt_n,
                            avg_V,
                            resistance, #
                            thr,#
                            adaptation,#
                            isi,#
                            peak,
                            peak_adaptation,
                            ap_width, #
                            hyp_value,
                            fist_spike,
                            up_down_ratio,
                            isi_adaptation,
                            thr_adp_ind,
                            psth,
                            int_fr,
                            fr, #
                            sub_thr,
                            mi,
                            spk_fr_adp,
                            imp,
                            self.exp_name,
                            self.cond,
                            self.trialnr] #
        return ephys_data

#%%

