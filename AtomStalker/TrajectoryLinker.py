from scipy.stats import binom
import itertools, collections, copy, time
import numpy as np
from scipy import optimize
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcess
from scipy.interpolate import RegularGridInterpolator
from matplotlib import patches

class TrajectoryAnalyzer(object):
    """
    
    This class is first handed an initial set of positions and intensities and links these. 
    After this is done it expects continuous input of positions and intensities frame by frame 
    and links them as those arrive.
    
    """
    
    def __init__(self,p_r={'type':'laguerre','paras':{'deg':5}},p_phi={'type':'legendre','paras':{'deg':5}},
                    p_dI={'type':'normal','paras':{'cov':.1,'mu':0.}},eps=0.05,explore=True):
        """
        Inference stuff needs to be set initially. p_r => P(r), p_phi => P(phi), p_dI => P(intensity difference)
        """
        
        self.p_phi = p_phi
        
        #probability related
        self.p_r_implemented = ['laguerre','normal']
        assert p_r['type'] in self.p_r_implemented, "Assertion failed - functional type {} not one of the implemented type for r: {}".format(p_r['type'],self.p_r_implemented)
        self.p_r = p_r
        
        self.p_phi_implemented = ['legendre','uniform']
        assert p_phi['type'] in self.p_phi_implemented, "Assertion failed - functional type {} not one of the implemented type for phi: {}".format(p_phi['type'],self.p_phi_implemented)
        self.p_phi = p_phi
        
        self.p_dI_implemented = ['normal','uniform']
        assert p_dI['type'] in self.p_dI_implemented, "Assertion failed - functional type {} not one of the implemented type for intensity difference: {}".format(p_dI['type'],self.p_dI_implemented)
        self.p_dI = p_dI
        
        self.p_r_fun = None #will be set after fitting
        self.p_phi_fun = None #will be set after fitting
        self.p_dI_fun = None #will be set after fitting
        self.p_nd = None #probability of non-disappearance from frame to frame
        self.p_nondisappearance_fun = None #will be set after fitting using Binomial distribution and self.p_nd the probability of non-disappearance from frame to frame
        
        self.rs_accumulated = None #will contain the rs for the linked trajectories
        self.phis_accumnulated = None #will contain the phis for the linked trajectories
        self.dIs_accumnulated = None #will contain the dIs for the linked trajectories
        
        self.r_mean = None
        
        #particle motion related
        self.rs_storage = []
        self.phis_storage = []
        self.dIs_storage = []
        self.rs_range = [] #min and max value for pdf
        self.num_pos_vs_t = None #continuously updated number of particles observed each frame
        self.avg_num_pos = None
        self.expected_num_disappearing = None #continously updated value of atoms disappearing from frame to frame
        self.positions = None
        self.intensities = None
        self.dim = None
        self.Nposmax = None
        self.Nt = None
        self.Nt_new = None #number of time steps added due to update of positions
        self.LM = None #row corresponds to time step into the future (row 0 = 1 step into the future), columns correspond ot the indices of particle position entries in self.positions
        self.LM_traj = None #keeps the dense form of the trajectories (row = trajectory, column = timestep, entries = int referring to entry in self.positions)
        self.LL = None #locked list - length of max number of particles in any frame, entries integers indicating for how many steps a trajectory shall not be updated
        
        #trajectory linking mode related
        self.eps = eps
        self.explore = explore
    
    def _determine_p_nondisappearance(self,verbose=False):
        self.num_pos_vs_t = np.array([self.Nposmax - len(np.where(np.isnan(v[0,:]))[0]) for v in self.positions])
        self.avg_num_pos = np.mean(self.num_pos_vs_t)
        self.expected_num_disappearing = self.num_pos_vs_t[1:] - self.num_pos_vs_t[:-1]
        self.expected_num_disappearing = np.mean([v for v in self.expected_num_disappearing if v<=0])
        
        self.p_nd = abs(self.expected_num_disappearing) / self.avg_num_pos
        
        if verbose:
            print("self.num_pos_vs_t {}".format(self.num_pos_vs_t))
            print("self.expected_num_disappearing {}".format(self.expected_num_disappearing))
            print("self.expected_num_disappearing {}".format(self.expected_num_disappearing))
            print("self.p_nd {}".format(self.p_nd))
        
        def fun_wrapper(p):
            def fun(dt):
                return binom.pmf(dt-1,dt,p)
            return fun
        
        self.p_nondisappearance_fun = fun_wrapper(self.p_nd)
        
        
    def set_initial(self,positions,intensities):
        """
        Sets initial observations for positions and intensities. Also sets initial probability funciton
        for the disappearance of particles.
        """
        
        print("Setting up the trajectory analyzer with initial positions and intensities...")
        self.positions = positions
        self.intensities = intensities
        self.Nposmax = positions.shape[2]
        self.dim = positions.shape[1]
        self.Nt = self.positions.shape[0]
        self.LL = [0]*self.Nposmax
        self.LM = np.zeros((self.Nposmax,self.Nt))
        self.LM[:] = np.nan
        
        self._determine_p_nondisappearance()
        
    def update(self,positions,intensities):
        
        self.Nt_new = positions.shape[0]
        
        if positions.shape[2] == self.Nposmax:
            self.positions = np.vstack((self.positions,positions))
            self.intensities = np.vstack((self.intensities,intensities))    
        elif positions.shape[2] < self.Nposmax:
            Nposnew = positions.shape[2]
            new_positions = np.zeros((self.Nt_new,positions.shape[1],self.Nposmax))
            new_positions[:] = np.nan
            new_positions[:,:,:Nposnew] = positions
            self.positions = np.vstack((self.positions,new_positions))
            new_intensities = np.zeros((self.Nt_new,self.Nposmax))
            new_intensities[:] = np.nan
            new_intensities[:,:Nposnew] = intensities
            self.intensities = np.vstack((self.intensities,new_intensities))
        else:
            Nposnew = positions.shape[2]
            new_positions = np.zeros((self.Nt,positions.shape[1],Nposnew))
            new_positions[:] = np.nan
            new_positions[:,:,:self.Nposmax] = self.positions
            self.positions = np.vstack((new_positions,positions))
            new_intensities = np.zeros((self.Nt,Nposnew))
            new_intensities[:] = np.nan
            new_intensities[:,:self.Nposmax] = self.intensities
            self.intensities = np.vstack((new_intensities,intensities))
            self.Nposmax = Nposnew
        
        self.Nt += self.Nt_new
        
    def _fit_p_phi(self,phis,P_phi):
        if self.p_phi['type'] == 'legendre':
            coef_Le = np.polynomial.legendre.Legendre.fit(phis,P_phi,self.p_phi['paras']['deg'])
            coef_Le = tuple(list(coef_Le))
            Le = np.polynomial.legendre.Legendre(coef_Le,domain=[-np.pi,np.pi])
            def Le_wrapper(Le):
                def Le_fun(x):
                    return [np.abs(Le(v)) for v in x]
                return Le_fun
            self.p_phi_fun = Le_wrapper(Le)
        
        elif self.p_phi['type'] == 'uniform':
            domain_factor = 1/(2.*np.pi)
            def uniform_wrapper(domain_factor):
                def uni_fun(x):
                    x = np.array(x)
                    s = x.shape
                    f = np.zeros(s)
                    f[:] = domain_factor
                    return f
                return uni_fun
            
            self.p_phi_fun = uniform_wrapper(domain_factor)
            
        else:
            raise ValueError('Error - unknown functional type {} choice for phi!'.format(self.p_r['phi']))
    
    def _fit_p_r(self,rs,P_r):
        if self.p_r['type'] == 'laguerre':
            coef_La = np.polynomial.laguerre.lagfit(rs,P_r,self.p_r['paras']['deg'])
            La = np.polynomial.laguerre.Laguerre(coef_La)
            
            self.p_r_fun = La
        
        elif self.p_r['type'] == 'normal':
            def normal_fun(x,sig,mu):
                
                sig2 = sig**2
                x = np.array(x)
                return 1./np.sqrt(2.*np.pi*sig2)*np.exp(-(x-mu)**2/(2.*sig2))
            
            fit = optimize.curve_fit(normal_fun,rs,P_r,p0=[1,0])
            
            def normal_wrapper(fit):
                def fun(x):
                    return normal_fun(x,fit[0][0],fit[0][1])
                return fun
            self.p_r_fun = normal_wrapper(fit)
        else:
            raise ValueError('Error - unknown functional type {} choice for r!'.format(self.p_r['type']))
    
    def _fit_p_dI(self,dIs,P_dI):
        
        if self.p_dI['type'] == 'normal':
            def normal_fun(x,sig,mu):
                sig2 = sig**2
                x = np.array(x)
                return 1./np.sqrt(2.*np.pi*sig2)*np.exp(-(x-mu)**2/(2.*sig2))
            
            fit = optimize.curve_fit(normal_fun,dIs,P_dI,p0=[1,0])
            
            def normal_wrapper(fit):
                def fun(x):
                    return normal_fun(x,fit[0][0],fit[0][1])
                return fun
            self.p_dI_fun = normal_wrapper(fit)
        
        elif self.p_dI['type'] == 'uniform':
            def uniform_fun(x,ab):
                x = np.array(x)
                s = np.zeros(x.shape)
                s[:] = ab
                return s
            
            fit = optimize.curve_fit(uniform_fun,dIs,P_dI,p0=[1])
            
            def uniform_wrapper(fit):
                def fun(x):
                    return uniform_fun(x,fit[0][0])
                return fun
            self.p_dI_fun = uniform_wrapper(fit)
        
        else:
            raise ValueError('Error - unknown functional type {} choice for dI!'.format(self.p_dI['type']))
    
    def show_current_probability_funs(self,r_lim=(0,5),phi_lim=(-np.pi,np.pi),dI_lim=(-.1,.1),dt_lim=(1,4)):
        fig = plt.figure(figsize=(20,10))
        
        plt.subplot(221)
        rs = np.linspace(r_lim[0],r_lim[1],100)
        P_r = self.p_r_fun(rs)
        plt.plot(rs,P_r,'-')
        plt.ylabel('p(r)')
        plt.xlabel('r')
        
        plt.subplot(222)
        phis = np.linspace(phi_lim[0],phi_lim[1],100)
        P_phi = self.p_phi_fun(phis)
        plt.plot(phis,P_phi,'-')
        plt.ylabel('p(phi)')
        plt.xlabel('phi')
        
        plt.subplot(223)
        dIs = np.linspace(dI_lim[0],dI_lim[1],100)
        P_dI = self.p_dI_fun(dIs)
        plt.plot(dIs,P_dI,'-')
        plt.ylabel('p(dI)')
        plt.xlabel('dIs')
        
        plt.subplot(224)
        dts = np.array(range(dt_lim[0],dt_lim[1]))
        P_dt = self.p_nondisappearance_fun(dts)
        plt.plot(dts,P_dt,'-')
        plt.ylabel('p_non-disappearance')
        plt.xlabel('dt')
        plt.show()       
           
    def initialize_probabilities(self,rs,P_r,phis,P_phi,dIs,P_dI):
        
        self._fit_p_r(rs,P_r)
        self.r_mean = np.sum(rs * P_r)
        self._fit_p_phi(phis,P_phi)
        self._fit_p_dI(dIs,P_dI)
        
    def _update_LL(self,t):
        """
        self.LL: 1-d list with integers for number of steps the position with the same index is still untouchable
        """
        if t==0:
            self.LL = [0]*self.Nposmax
        self.LL = [v-1 if v>0 else v for v in self.LL]
    
    def _update_LM(self,t,Ntf):
        """
        self.LM: ndarray with shape (Ntf,self.Nposmax) if t + Ntf <= last t
            each column corresponds to a position at the current timestep
            each row corresponds to a timestep
            entries are integers indicating the related linked position at the current rows timestep
            entries can also be nan indicating no linking - updates make all steps furthest in the future nan
        """
        if t == 0:
            self.LM = np.array([[np.nan]*self.Nposmax]*Ntf)
        else:
            self.LM = np.roll(self.LM,-1,axis=0)
            self.LM[-1,:] = np.nan
        
    def _get_observables(self,t,Ntf,verbose=False):
        """
        Gets radius r, angle phi, intensity difference dI for delta t = 1,Ntf, where Ntf is the number of time steps into the future to consider.
        
        Input:
            t - int, index to first dimension of the self.positions array
            Ntf - int, number of additional array entries along the first dimension to consider except the one specified by t
        Returns:
            rs - pairwise distances, stack of square arrays n x n where is the number of particles
            phis - angles for pairwise distances, stack of square arrays n x n where is the number of particles
            dIs - pairwise intensity differences, stack of square arrays n x n where is the number of particles
        """
        
        #checking how many steps actually are left -> tf = final future step
        if t+Ntf+1 <= self.positions.shape[0]:
            tf = t+Ntf+1
        else:
            tf = self.positions.shape[0]
                
        pos2proc = self.positions[t:tf,:,:]
        int2proc = self.intensities[t:tf,:]
        dt = tf-t-1
        
        distance_vecs = np.zeros((self.dim,self.Nposmax,self.Nposmax))
        
        rs = np.zeros((Ntf,self.Nposmax,self.Nposmax)) #row = timestep, column = position
        rs[:] = np.nan
        phis = np.zeros((Ntf,self.Nposmax,self.Nposmax))
        phis[:] = np.nan
        dIs = np.zeros((Ntf,self.Nposmax,self.Nposmax))
        dIs[:] = np.nan
        
        if verbose:
            print("self.positions {} t {} tf {} dt {}".format(self.positions.shape,t,tf,dt))
            print("pos2proc {} {}".format(pos2proc.shape,pos2proc))
        for dt_tmp in xrange(0,dt):
            distance_vecs[:] = np.nan
            if verbose:
                print("\ndt_tmp {} Nposmax {} dim {}".format(dt_tmp,self.Nposmax,self.dim))
            for i in xrange(self.Nposmax):
                #current position = column, possible next position = row for next time step
                current = np.reshape(pos2proc[0,:,i],(self.dim,1))
                targets = pos2proc[dt_tmp+1,:,:]
                differences = np.add(targets,-current)
                distances = np.linalg.norm(differences,axis=0)
                
                distance_vecs[:,:,i] = differences
                rs[dt_tmp,:,i] = distances
                dIs[dt_tmp,:,i] = np.roll(int2proc[dt_tmp+1,:],i,axis=0) - int2proc[0,i]
            
            x1 = distance_vecs[1,:,:]
            x2 = distance_vecs[0,:,:]
            phis[dt_tmp,:,:] = np.arctan2(x1,x2)
            
        return rs, phis, dIs
        
    def _get_joint_log_probabilities(self,rs,phis,dIs,verbose=False):
        """
        Processes rs, phis and dIs as produced by self._get_observables().
        
        Returns:
            logP_joint - ndarray shape (num frames to link, self.Nposmax, self.Nposmax) log(P_joint)
        """
        if verbose:
            print("rs {} phis {} dIs {}".format(rs.shape,phis.shape,dIs.shape))
        
        P_r = np.zeros(rs.shape)
        P_r[:] = np.nan
        P_phi = np.zeros(rs.shape)
        P_phi[:] = np.nan
        P_dI = np.zeros(rs.shape)
        P_dI[:] = np.nan
        
        P_r[0] = self.p_r_fun(rs[0])
        if rs.shape[0]>1: #decompose hyothetical tracjetories for Ntf>1
            rmean = self.r_mean
            P_rmean = self.p_r_fun(rmean)
            
            #generate pseudo trajectories each timestep
            signs = np.random.choice([-1,1],size=len(rs.shape)-1)
            rmeans = np.array([rmean]*(len(rs.shape)-1))
            sim_rs = rmeans * signs
            P_sim = np.array([P_rmean]*rmeans.shape[0])
            r_cumulative = np.cumsum(rmeans)
            if verbose:
                print("rmean {} signs {} rmeans {} sim_rs {} P_sim {}".format(rmean,signs,rmeans,sim_rs,P_sim))
                print("r_cumulative {}".format(r_cumulative))
            
            #calculate cummulative distances
            for i in xrange(1,rs.shape[0]):                
                #for each timestep for each possible link calculate simulated final displacements using
                #the appropriate simulated trajectory in sim_rs. the obtained final displacement and the simulated ones
                #are then used for the calculation of the probability of that link only based on distance
                # r_{tn-1->tn} = ||(r_tn-r_t0)| - |sum_i=0^(n-2) r_ti+1 - r_ti||
                rs_tmp = np.abs(rs[i] - r_cumulative[i-1])
                P_r[i] = self.p_r_fun(rs_tmp) * P_sim[i-1]
                if verbose:
                    print("\ni: {}\nrs_tmp = {} P_r[i] = {}".format(i,rs_tmp,P_r[i]))
            
        P_phi = self.p_phi_fun(phis)
                
        P_dI = self.p_dI_fun(dIs)
        dts = np.array(range(1,rs.shape[0]+1))
        P_dt = self.p_nondisappearance_fun(dts)
        if verbose:
            print("rs {}".format(rs))
            print("P_r {}".format(P_r))
            print("phis {}".format(phis))
            print("P_phi {}".format(P_phi))
            print("dIs {}".format(dIs))
            print("P_dI {}".format(P_dI))
            print("P_dt {}".format(P_dt))
        
        logP_joint = np.log(P_r) + np.log(P_phi) + np.log(P_dI)
        for i in xrange(logP_joint.shape[0]):
            logP_joint[i,:,:] += np.log(P_dt[i])
        
        return logP_joint
            
    def _merge_logP_joint(self,logP_joint,verbose=False):
        
        #time_index = list of integers from 0 to log_P_joint.shape[0]-1 relating positions to future time steps (time_index = 0 means 1 step into the future and os on)
        time_index = list(itertools.chain(*[[v]*self.Nposmax for v in xrange(1,logP_joint.shape[0]+1)]))
        
        merged_logP_joint = logP_joint[0]
        for i in xrange(1,logP_joint.shape[0]):
            merged_logP_joint = np.vstack((merged_logP_joint,logP_joint[i]))
        if verbose:
            print("log_P_joint {}".format(logP_joint.shape))
            print("merged_log_P_joint {}".format(merged_logP_joint.shape))
        return merged_logP_joint, time_index
    
    
    def _update_LL_new_assignments(self,assignments,duplicates,time_index):
        for i,a in enumerate(assignments):
            if self.LL[i] == 0 and not (a in duplicates):
                self.LL[i] = time_index[a]
    
    def _assign_to_trajectories_via_mcmc(self,merged_logP_joint,time_index,t,verbose=False):
                
        #create subset of merged_logP_joint for current and target positions which are not locked by previous processing
        #column indices which can be used for linking based on 0 self.LL entries and not all nan columns in merged_pogP_joint
        idx_tails = [iv for iv,v in enumerate(merged_logP_joint.T) if not all(np.isnan(v))] #not only nan-columns
        idx_tails = [v for v in idx_tails if self.LL[v]==0]
        idx_heads_blocked = list(itertools.chain(*[[iv*self.Nposmax + v2 for v2 in self.LM[iv,:]] for iv in xrange(self.LM.shape[0])])) #entry = timestep (starting at 1) * entry in self.LM
        
        idx_heads  = list(set(range(merged_logP_joint.shape[0])).difference(idx_heads_blocked))
        
        sub_merged_logP_joint = np.array([[row[ix] for ix in idx_tails] for row in merged_logP_joint]) #subset of merged_logP_joint contain no all-NaN columns
        sub_merged_logP_joint = sub_merged_logP_joint[(idx_heads,)]
        if verbose:
            print("time_index {} idx_heads_blocked {}".format(time_index,idx_heads_blocked))
            print("idx_tails {} {}".format(idx_tails,len(idx_tails)))
            print("idx_heads {} {}".format(idx_heads,len(idx_heads)))
            print("merged_logP {} sub_merged_logP {}".format(merged_logP_joint.shape,sub_merged_logP_joint.shape))
            print("merged_logP {}".format(merged_logP_joint))
        #get ix for sub_merged_logP_joint which are assignable
          
        #calculate 1d array of assignments - entry is position in sub_merged_logP_point
        assignments = np.zeros(sub_merged_logP_joint.shape[1])
        assignments[:] = np.nan
        if verbose:
            print("assignments {}".format(assignments))
        
        if len(sub_merged_logP_joint)>0: #case that some positions are still left to assign
            unresolved = True
        else: #no positions left to assign to trajectories -> finished here
            unresolved = False
        
        if verbose:
            print("sub_merged_logP_joint {}".format(sub_merged_logP_joint))
        
        #begin loop here until there are no more duplicates in the updateable positions
        if unresolved:
            
            #get all greedy choices and explore random choices
            greedy_assignment = np.nanargmax(sub_merged_logP_joint,axis=0)
            if verbose: print("greedy_assignment {}".format(greedy_assignment))
            coin_tosses = np.random.uniform(low=0,high=1,size=sub_merged_logP_joint.shape[1])
            ix_greedy = np.where(coin_tosses>self.eps)
            ix_explore = np.where(coin_tosses<self.eps)
            if verbose: print("ix_greedy {}".format(ix_greedy))
            if verbose: print("ix_explore {}".format(ix_explore))
            
            assignments[ix_greedy] = greedy_assignment[ix_greedy]
            if verbose: print("assignments[ix_greedy] {}".format(assignments[ix_greedy]))
            choices_left = list(set(range(sub_merged_logP_joint.shape[0])).difference(list([int(v) for v in assignments if not np.isnan(v)])))
            if verbose: print("choices left {}".format(choices_left))
            random_assignments = np.random.choice(choices_left,replace=False,size=len(ix_explore[0]))
            if verbose: print("random assignemnts {}".format(random_assignments))
            assignments[ix_explore] = random_assignments
            if verbose: print("assignments {}".format(assignments))
            
            #sort out duplicates - because the above random selection was done to prevent creation of duplicates the presenet duplicates can be sorted using max probs
            duplicates = [v for v,c in collections.Counter([int(v) for v in assignments]).items() if c>1]
            ix_dupes = [iv for iv,v in enumerate(assignments) if v in duplicates]
            
            choices_left = list(set(choices_left+list(duplicates)).difference(list(random_assignments)))
            if verbose: print("choices left {}".format(choices_left))
            options_for_duplicates = [[(row,ix_d,sub_merged_logP_joint[row,ix_d]) for row in choices_left if not np.isnan(sub_merged_logP_joint[row,ix_d])] for ix_d in ix_dupes]
            options_for_duplicates = [sorted(v,key = lambda x: x[2], reverse = True) for v in options_for_duplicates]
            idx_sorted_options = [v[0][2] for v in options_for_duplicates]
            if verbose: print("idx_sorted_options {}".format(idx_sorted_options))
            idx_sorted_options = [v[0] for v in sorted(list(enumerate(idx_sorted_options)),key = lambda x: x[1], reverse=True)]
            if verbose: print("idx_sorted_options {}".format(idx_sorted_options))
            options_for_duplicates = [options_for_duplicates[v] for v in idx_sorted_options]
            ix_dupes = [ix_dupes[v] for v in idx_sorted_options]
            if verbose: print("options for duplicates (head,tail,logp) {}".format(options_for_duplicates))
            for i in xrange(len(options_for_duplicates)):
                if verbose: 
                    print("i {}".format(i))
                    print("options_for_duplicates[i] {}".format(options_for_duplicates[i]))
                if len(options_for_duplicates[i])>0:
                    row = options_for_duplicates[i][0][0]
                    assignments[ix_dupes[i]] = row
                    for j in xrange(i+1,len(ix_dupes)):
                        options_for_duplicates[j] = [v for v in options_for_duplicates[j] if v[0]!=row]
                else:
                    assignments[ix_dupes[i]] = np.nan
            if verbose: 
                print("assignments {}".format(assignments))
                
        #we made it! now converting stuff for output yay...
        #since the assignments array was used for the sub matrix of the log joint probability matrix we need to convert
        #assignments = [int(v) for v in assignments]
        tmp_LM_traj = [np.nan]*self.Nposmax
        for i,ass in zip(idx_tails,assignments):
            
            if not np.isnan(ass):
                blocked_time = time_index[idx_heads[int(ass)]]
                future_time = t + blocked_time
                future_pos = idx_heads[int(ass)]%self.Nposmax
                if verbose: print("future_pos = idx_heads[ass]%self.Nposmax = idx_heads[{}]%self.Nposmax = {}%{} = {}".format(int(ass),idx_heads[int(ass)],self.Nposmax,future_pos))
                #LL
                self.LL[i] = blocked_time
                #LM
                self.LM[blocked_time-1,i] = future_pos
                #LM_traj
                tmp_LM_traj[i] = (future_time,future_pos)
          
        self.LM_traj += [tmp_LM_traj]
        if verbose: 
            print("LL linked {}".format(self.LL))
            print("LM linked {}".format(self.LM))
            print("LM_traj linked {}".format(self.LM_traj))
        
        self.time_index = time_index
        self.assignments = assignments
        self.idx_heads = idx_heads
        self.idx_tails = idx_tails
        #return LMtmp, assignments 
    
    def _assign_to_trajectories(self,merged_logP_joint, time_index,t):
        #assigns atoms to trajectories, either by exploration/exploitation using mcmc (explore=True) or by minimum cost (explore=False)
        
        if self.explore:
            self._assign_to_trajectories_via_mcmc(merged_logP_joint, time_index,t)
        
    
    def _fit_probabilities(self,phis,rs,dIs,P_phi,P_r,P_dI):
        
        self._fit_p_r(rs,P_r)
        self._fit_p_phi(phis,P_phi)
        self._fit_p_dI(dIs,P_dI)
        
    def get_LM_traj_matrix(self,verbose=False):
        """
        The present self.LM_traj is a list of lists where each entry is an edge pointing from the 
        current position and time to a future position and time denoted by (future_time,future_pos)
        
        This is transformed into an array where rows correspond to trajectories and columns time steps.
        Entries are integer or nan indicating which position is linked to the trajectory next.
        """
        
        LM = np.zeros((self.positions.shape[2],self.positions.shape[0]))
        LM[:] = np.nan
        for i,pos in enumerate(self.positions[0,0,:]):
            #if not np.isnan(pos):
            LM[i,0] = i
        
        #loop over individual trajectories - if nan encountered as head value assume next head value is for the same head index at a later point in time
        num_trajs = len(self.LM_traj[0])
        num_t = len(self.LM_traj)
        
        for i in xrange(num_trajs):
            
            t = 0
            old_idx = i
            while t<num_t and not np.isnan(old_idx):
                try:
                    t, new_idx = self.LM_traj[t][old_idx]
                except:
                    break
                LM[i,t] = new_idx
                old_idx = copy.deepcopy(new_idx)            
        return LM
        
    def _update_probabilities(self,rs,phis,dIs,r_bins=10,phi_bins=10,dI_bins=10,verbose=False):
        """
        Called after the last linking is completed to update the pool of rs, phis and dIs
        and update the associated probabilities.
        """
        if verbose: 
            print("rs {} phis {}, dIs {}".format(rs,phis,dIs))
            print("assignments {}".format(self.assignments))
            print("idx_heads {}".format(self.idx_heads))
            print("idx_tails {}".format(self.idx_tails))
            print("time_index {}".format(self.time_index))
        #select values related to high probability values / the current linking
        
        assignments = [int(ass) for ass in self.assignments if not np.isnan(ass)]
        idx_tails = [v for v,v2 in zip(self.idx_tails,self.assignments) if not np.isnan(v2)]
        if verbose: 
            print("assignments {}".format(assignments))
            print("idx_tails {}".format(idx_tails))
        ts = [self.time_index[self.idx_heads[ass]]-1 for i,ass in zip(idx_tails,assignments)]
        atom = idx_tails
        neigh = [self.idx_heads[ass]%self.Nposmax for ass in assignments]
        if verbose: 
            print("ts {}, atom {} neigh {}".format(ts,atom,neigh))

        rs_new = rs[(ts,neigh,atom)]
        phis_new = phis[(ts,neigh,atom)]
        dIs_new = dIs[(ts,neigh,atom)]
        
        if verbose: print("new rs {} phis {}, dIs {}".format(rs_new,phis_new,dIs_new))
        self.rs_storage += list(rs_new)
        self.rs_range = [0,np.amax(self.rs_storage)]
        self.phis_storage += list(phis_new)
        self.dIs_storage += list(dIs_new)
        
        #fitting probabilities to new distributions
        P_r, xr = np.histogram(self.rs_storage,bins=r_bins,normed=True)
        rplot = (xr[1:]+xr[:-1])*.5
        self.r_mean = np.sum(rplot*P_r)
        P_phi, xphi = np.histogram(self.phis_storage,bins=phi_bins,normed=True)
        phiplot = (xphi[1:]+xphi[:-1])*.5
        P_dI, xdI = np.histogram(self.dIs_storage,bins=dI_bins,normed=True)
        dIplot = (xdI[1:]+xdI[:-1])*.5
        
        self._fit_probabilities(phiplot,rplot,dIplot,P_phi,P_r,P_dI)
    
    def link_all(self,Ntf=1,re_evaluate=False,show=False):
        """
        Main function, does the algorithm step 2) described above. Interface function which 
        links the positions for all known frames
        
        Input:
            Ntf - int (optional), number of steps into the future to consider, hence Ntf >= 1. Example if Ntf = 1 then
                the current and the next step in the future will be considered for linking with the 
                last linked step.
            re_evaluate - boolean (optional), if True the trajectories will be processed again using the obtained
                probability distributions
        
        """
        assert isinstance(Ntf,int), "Assertion failed - expected Ntf of type 'int', got {} instead.".format(type(Ntf))
        assert Ntf >= 1, "Assertion failed - expected Ntf >= 1, got {} instead.".format(Ntf)
        self.LM_traj = []
        
        for t in xrange(self.positions.shape[0]-1):
            print("\nframe {}".format(t))
            #2.1) reduce non-zero entries in LL by one
            self._update_LL(t)
            self._update_LM(t,Ntf)
            
            #2.2) get r, phi, dI delta t=1,Ntf, where Ntf is the number of time steps into the future to consider
            rs, phis, dIs = self._get_observables(t,Ntf)
            
            #2.3) get the corresponding probabilities P(r,phi,dI,dt) 
            logP_joint = self._get_joint_log_probabilities(rs,phis,dIs)
            
            #2.4) join matrices vertically and create time related indexing
            merged_logP_joint, time_index = self._merge_logP_joint(logP_joint)
            
            #2.5) assign atoms to trajectories
            self._assign_to_trajectories(merged_logP_joint,time_index,t)
            
            #2.6) update probability distributions
            self._update_probabilities(rs,phis,dIs)
            if show:
                self.show_current_probability_funs(r_lim=(0,5),phi_lim=(-np.pi,np.pi),dI_lim=(-.1,.1),dt_lim=(1,4))
            
        if re_evaluate:
            self.LM_traj = []
        
            for t in xrange(self.positions.shape[0]-1):
                self._update_LL(t)
                self._update_LM(t,Ntf)
                
                #2.2) get r, phi, dI delta t=1,Ntf
                rs, phis, dIs = self._get_observables(t,Ntf)

                #2.3) get the corresponding probabilities P(r,phi,delta I,delta t)
                logP_joint = self._get_joint_log_probabilities(rs,phis,dIs)

                #2.4) join matrices vertically and create time related indexing
                merged_logP_joint, time_index = self._merge_logP_joint(logP_joint)

                #2.5) assign atoms to trajectories
                self._assign_to_trajectories(merged_logP_joint,time_index,t)
            
        
    def link_new(self):
        #interface function for online linkage of arriving frames - to be called after self.update()
        print("blub - not implemented yet")
        
def load_coordinates(path):
    
    with open(path,'r') as f:
        lines = map(lambda x: x.rstrip('\n'), f.readlines())
    
    positions = []
    
    for line in lines:
        if 'frame' in line:
            if 'tmp_positions' in locals():
                positions += [np.array(tmp_positions).T]
            tmp_positions = []
        else:
            tmp_positions += [map(float,line.split())]
    
    max_num_pos = max([v.shape[1] for v in positions])
    out_positions = np.zeros((len(positions),2,max_num_pos))
    out_positions[:] = np.nan
    
    for i,pos in enumerate(positions):
        out_positions[i,:,:pos.shape[1]] = pos
        
    print("loaded {} frames".format(len(out_positions)))
    
    return out_positions

def show_trajectories(positions,LM_traj,intensities,LM_traj2=None,t_lim=None,xlim=None,
                      ylim=None,show_tuple=True,title=None,legend=True):
    
    win_width = .1
    patch_w = 7
    patch_h = 5
    
    fig = plt.figure(figsize=(10,10))
    if LM_traj2 is not None:
        ax1 = plt.subplot(121)
        plt.grid()
    else:
        ax1 = plt.subplot(111)
        plt.grid()
    plt.hold(True)
    
    if t_lim is None:
        lim_positions = positions
    else:
        lim_positions = positions[t_lim[0]:t_lim[1]]
        
    c = 0
    for t,t_pos in enumerate(lim_positions):
        x,y = t_pos[0,:], t_pos[1,:]
        plt.plot(x,y,'bo',alpha=0.4)
        
        if show_tuple:
            for i, (vx, vy) in enumerate(zip(x,y)):
                if vx==vx:
                    p = patches.Rectangle((vx+win_width+.5,vy), patch_w, patch_h,fill=True,color='w')
                    ax1.add_patch(p)
                    ax1.text(vx+win_width+.5,vy,str(c),color='r',fontsize=12)
                    c += 1
        
    if LM_traj is not None:
        for i,idx in enumerate(LM_traj):
            if t_lim is None:
                pos = [positions[t,:,int(ix)] for t,ix in enumerate(idx) if ix==ix]
            else:
                pos = [positions[t,:,int(ix)] for t,ix in enumerate(idx) if ix==ix if t_lim[0]<=t<t_lim[1]]
            x, y = [v[0] for v in pos], [v[1] for v in pos]
            plt.plot(x,y,'-',linewidth=2.,alpha=0.4,label=str(i))
        
    plt.hold(False)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if legend: plt.legend(loc=0)
    
    ######
    
    if LM_traj2 is not None:
        plt.title('original trajectory')
        ax2 = plt.subplot(122)
        plt.grid()
        plt.hold(True)
        
        if t_lim is None:
            lim_positions = positions
        else:
            lim_positions = positions[t_lim[0]:t_lim[1]]

        for t,t_pos in enumerate(lim_positions):
            x,y = t_pos[0,:], t_pos[1,:]
            plt.plot(x,y,'bo',alpha=0.4)

            if show_tuple:
                for i, (vx, vy) in enumerate(zip(x,y)):
                    if vx==vx:
                        p = patches.Rectangle((vx+win_width+.5,vy), patch_w, patch_h,fill=True,color='w')
                        ax2.add_patch(p)
                        ax2.text(vx+win_width+.5,vy,str((t,i)),color='r',fontsize=12)

        for i,idx in enumerate(LM_traj2):
            if t_lim is None:
                pos = [positions[t,:,int(ix)] for t,ix in enumerate(idx) if ix==ix]
            else:
                pos = [positions[t,:,int(ix)] for t,ix in enumerate(idx) if ix==ix if t_lim[0]<=t<t_lim[1]]
            x, y = [v[0] for v in pos], [v[1] for v in pos]
            plt.plot(x,y,'-',linewidth=2.,alpha=0.4,label=str(i))

        
        plt.hold(False)
        if legend: plt.legend(loc=0)
        plt.title('Guessed trajectory')
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
    
    plt.show() 

def convert_man2links(man,positions):
    """
    Input:
        man - list of lists, each lowest level list contains integers which refer to 
            a position in time and space observed (result of a merging of time steps).
        positions - ndarray (Nt,2,Nposmax) 
    """
    flat_positions = positions[0].T

    num_not_nan = len(flat_positions) - sum(np.isnan(flat_positions[0,:]))
    print("num_not_nan {}".format(num_not_nan))
    idx_t = [0]*num_not_nan #time index
    idx_p = range(num_not_nan)
    for i,pos in enumerate(positions[1:]):
        num_not_nan = len(pos.T) - sum(np.isnan(pos.T[0,:]))
        
        for j,p in enumerate(pos.T):
            if p[0]==p[0]:
                idx_t += [i+1]
                idx_p += [j]
        
    print("idx_t {} idx_p {}".format(len(idx_t),len(idx_p)))
    print("max {}".format(idx_p))
    print("man {}".format(man))
    links = [[(idx_t[m2],idx_p[m2]) for m2 in m] for m in man]
    return links

def links2LMtraj(links):
    num_t = max([max([v[0] for v in v2]) for v2 in links])+1
    num_traj = len(links)

    LM_traj = np.zeros((num_traj,num_t))
    LM_traj[:] = np.nan
    for i,link in enumerate(links):
        for (t,j) in link:
            LM_traj[i,t] = j
    return LM_traj
        
def LMtraj2links(LM_traj,t_max=3):
    links = [[(t,p) for t,p in enumerate(tr) if p==p and t<=t_max] for tr in LM_traj]
    return links

class SigmoidConstructor(object):
    
    def __init__(self,space):
        """
        Input:
            space - dict, defines the space the sigmoid is in, 
                    e.g. {'r':np.linspace(0,50,100),'dt':np.linspace(0,5)}
            
        """
        self.space = space
        self.logistic_sigmoid = None
        self.axis_order = ['r','dt']
        
        self.X = None #X = [R,T,...]
        self.logistic_sigmoid = None 
        
    def _time_function(self,T):
        
        f = np.exp(-(T-1.5)**2/10.)
        f[np.where(f<=0)] = 0
        f[np.where(f>1)] = 1
        return f
    
    def _distance_function(self,R):
        
        b = 1/10.
        f = 1./(1. + np.exp(R-10))
        f[np.where(f<=0)] = 0
        f[np.where(f>1)] = 1
        return f
    
    def _sigmoid_function(self,X):
        #X = [R,T,...]
        return self._distance_function(X[0])*self._time_function(X[1])
    
    def create(self):
        """
        Function to create the sigmoid based on self.space.
        """
        X = np.meshgrid(*[self.space[v] for v in self.axis_order],indexing='ij')
        self.X = X
        self.logistic_sigmoid = self._sigmoid_function(X)
    
    def show_sigmoid(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(self.X[0],self.X[1],self.logistic_sigmoid,alpha=.8,rstride=1,cstride=1)
        ax.set_xlabel(self.axis_order[0])
        ax.set_ylabel(self.axis_order[1])
        plt.show()
        
    def get_splined_sigmoid(self):
        values = np.array([self.space[v] for v in self.axis_order])
        rgi = RegularGridInterpolator(values,self.logistic_sigmoid,bounds_error=False,fill_value=0)
        
        return rgi

def flatten(some_list):
    return list(itertools.chain(*some_list))

class PositionProcessing(object):
    """
    Processing positions. Assumes that the entries under index 0 of the zeroth axis is the current /
    the reference for the linking. Hence for the application to construction of trajectories new positions
    variables need to be processed where the 0th entries of the 0th axis correspond to the current configuration
    and all other positions to possible future positions for the linking.
    """
    
    def __init__(self):
        
        #data to be set - originating from observations or simulations
        self.positions = None #(Nt,2,Npos)
        self.intensities = None #(Nt,Npos)
        
        #properties for linking: time distance, spatial distance, angle (rel. to cartesian x), intensity
        self.combinations = None #list: contains tuples of two tuples ((t_s,i_s), (t_f,i_f)) where t and i indicate time index and position index (axes 0 and 2 in self.positions repsectively), and s = start and f = finish
        self.combi_info = {'r':[],'dI':[],'theta':[],'dt':[]} #dict: keys are properties for linking, values are lists corresponding in their order to the order in self.combinations
        self.implemented_info = ['r','dt']
    
    def set_data(self,positions,intensities):
        self.positions = positions
        self.intensities = intensities

    def _generate_combinations(self,neighbors=5,leafsize=10):        
        #generation of kd trees, one for each time step
        from scipy.spatial import KDTree
        
        idx_non_nan = [[ip for ip,p in enumerate(pos.T) if p[0]==p[0]] for it,pos in enumerate(self.positions)]
        
        trees = []
        for i,idx in enumerate(idx_non_nan):
            if i == 0:
                kdt = None
            else:
                obs = np.array(self.positions[i,:,tuple(idx)])
                kdt = KDTree(obs,leafsize=leafsize)
            trees += [kdt]
            
        #using the kd trees to generate tailored edges for spatially correlated positions
        idx_pos_t0 = np.array(idx_non_nan[0])
        idx_pos_t = np.array(idx_non_nan[1:])
        edges = []
                
        for i, pos in zip(idx_pos_t0,self.positions[0,:,tuple(idx_pos_t0)]):
            for t,kdt in enumerate(trees):
                if t>0:
                    _, idx_neighbors = kdt.query(pos, k=neighbors)
                    x,y = self.positions[t,0,tuple(idx_neighbors)], self.positions[t,1,tuple(idx_neighbors)]
                    edges += [((0,i),(t,v)) for v in idx_neighbors]
            
        self.combinations = edges
        
    def _calc_distance(self,x0,x1):
        return np.linalg.norm(x0 - x1)
    
    def _calc_distance_time(self,t0,t1):
        return t1 - t0
        
    def _calculate_metrics(self):
        """
        Calculating distances and such for the pairs specified in self.combinations.
        """
        #spatial distance
        self.combi_info['r'] = [self._calc_distance(self.positions[v[0][0],:,v[0][1]],self.positions[v[1][0],:,v[1][1]]) for v in self.combinations]
        
        #time distance
        self.combi_info['dt'] = [self._calc_distance_time(v[0][0],v[1][0]) for v in self.combinations]
        
    def process(self):
        self.combi_info = {'r':[],'dI':[],'theta':[],'dt':[]}
        self._generate_combinations()
        self._calculate_metrics()
   
class RGI_wrap(object):
    def __init__(self,rgi=None):
        self.rgi = rgi
    def fit(self,X,y):
        self.rgi = RegularGridInterpolator(X,y,bounds_error=False,fill_value=0)
    def predict(self,X):
        return self.rgi(X)

def softmax(x,tau):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(-(x - np.max(x))/float(tau))
    return e_x / e_x.sum()

class SigmoidManager(PositionProcessing):
        
    def __init__(self,bin_spec={'r':10,'dI':10,'theta':10,'dt':3}):
        #data to be set - originating from observations or simulations
        self.positions = None #(Nt,2,Npos)
        self.intensities = None #(Nt,Npos)
        
        #properties for linking: time distance, spatial distance, angle (rel. to cartesian x), intensity
        self.combinations = None #list: tuples of tuples ((t_s,i_s), (t_f,i_f)) where t and i indicate time index and position index (axes 0 and 2 in self.positions repsectively), and s = start and f = finish
        self.combi_info = {'r':[],'dI':[],'theta':[],'dt':[]} #dict: keys are properties for linking, values are lists corresponding in their order to the order in self.combinations
        self.implemented_info = ['r','dt']
        self.num_info_types = len(self.implemented_info)
            
        #links
        self.links = None #original links created manually
        self.links_updated = None #updated links based on examples obtained with self.links
        self.manual_combinations = None
        self.link_values = None #"values" in terms of reinforcement learning, here sigmoid value for a link
        
        #classification class related
        self.bin_spec = bin_spec
        self.idx_positive = None
        self.idx_negative = None
        self.positive_examples = None #ndarray (Nsamples positive,num info types)
        self.negative_examples = None #ndarray (Nsamples negative,num info types)
        self.num_positive = None
        self.num_negative = None
        self.probability_distributions = None #dict: 'positive': ... 'negative': ...
        self.edges = None #list of arrays
        self.logistic_sigmoid = None
        self.predictor = None #will contain the GaussianProcess class from sklearn      
        
    def _generate_classes(self,variant='joint',show=False):
        """
        Generates positives and negative classes.
        """
        #self.positive_combinations = flatten([flatten([[tuple([ps[i],p1]) for p1 in ps[i+1:]] for i in range(len(ps)-1)]) for ps in self.links])
        self.positive_combinations = []
        for traj in self.links:
            self.positive_combinations += [(v,v2) for v,v2 in zip(traj[:-1],traj[1:])]
        self.idx_positive = [iv for iv,v in enumerate(self.combinations) if v in self.positive_combinations]
        self.idx_negative = list(set(range(len(self.combinations))).difference(self.idx_positive))
        self.negative_combinations = [self.combinations[v] for v in self.idx_negative]
        self.num_positive = len(self.idx_positive)
        self.num_negative = len(self.idx_negative)
        
        self.positive_examples = np.zeros((self.num_positive,self.num_info_types))
        for i,typ in enumerate(self.implemented_info):
            self.positive_examples[:,i] = [self.combi_info[typ][v] for v in self.idx_positive]

        self.negative_examples = np.zeros((self.num_negative,self.num_info_types))
        for i,typ in enumerate(self.implemented_info):
            self.negative_examples[:,i] = [self.combi_info[typ][v] for v in self.idx_negative]

        #bins = [np.linspace(0,max(self.combi_info[v])*1.5,self.num_bins[v]) for v in self.implemented_info]
        ranges = [[0,max(self.combi_info[v])] for v in self.implemented_info]
        bins = [int(r[1]/float(self.bin_spec[v]))+1 for r,v in zip(ranges,self.implemented_info)]
        
        p_positive, edges = np.histogramdd(self.positive_examples,bins=bins,range=ranges,normed=True)
        p_negative, edges = np.histogramdd(self.negative_examples,bins=bins,range=ranges,normed=True)

        self.probability_distributions = {'positive':p_positive,'negative':p_negative}
        self.edges = [.5*(v[1:]+v[:-1]) for v in edges]   
        if show:
            fig = plt.figure()
            X, Y = np.meshgrid(*self.edges,indexing='ij')

            ax0 = plt.subplot(121,projection='3d')
            ax0.plot_surface(X,Y,p_positive)
            plt.title('Positive distribution')

            ax1 = plt.subplot(122,projection='3d')
            ax1.plot_surface(X,Y,p_negative)
            plt.title('Negative distribution')
            plt.show()
            
    def _calculate_sigmoid(self,show=False):
        #calculate logistic sigmoid
        #assuming uniform class priors
        
        pd_pos = self.probability_distributions['positive']
        pd_neg = self.probability_distributions['negative']
        a = np.log(pd_pos) - np.log(pd_neg)
        self.logistic_sigmoid = 1./(1+np.exp(-a))

        #no negative samples observed
        sigmoid_one = np.where(pd_neg == 0,1,0) + np.where(pd_pos !=0 ,1,0)
        sigmoid_one = np.where(sigmoid_one == 2)
        self.logistic_sigmoid[sigmoid_one] = 1

        #no negative and positive samples observed / np positive samples observed
        sigmoid_zero = np.where(pd_neg == 0,1,0) + np.where(pd_pos==0,1,0)
        sigmoid_zero = np.where(sigmoid_zero==2,1,0)
        sigmoid_zero += np.where(pd_pos==0,1,0)
        self.logistic_sigmoid[np.where(sigmoid_zero==2)] = 0

        if show:
            x = self.edges[0]
            y = self.edges[1]

            X,Y = np.meshgrid(x,y,indexing='ij')

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X,Y,self.logistic_sigmoid,rstride=1,cstride=1,alpha=0.6)
            plt.title('sigmoid')
            ax.set_xlabel('r')
            ax.set_ylabel('dt')
            plt.show()
        
            
    def fit(self,show=False):
        #RegularGridInterpolator
        self.predictor = RGI_wrap()
        self.predictor.fit(self.edges,self.logistic_sigmoid)
        
        X = np.meshgrid(*self.edges,indexing='ij')
        X = np.array(zip(*[np.ravel(v) for v in X]))
        
        if show:
            y_pred = self.predictor.predict(X)

            x = self.edges[0]
            y = self.edges[1]

            X,Y = np.meshgrid(x,y,indexing='ij')
            y_pred = np.reshape(y_pred,X.shape)

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X,Y,y_pred,rstride=1,cstride=1,alpha=0.6)
            plt.title('Predictor')
            ax.set_xlabel('r')
            ax.set_ylabel('dt')
            plt.show()
        
            X = np.array([np.linspace(0,100,100), [1]*100])
            
    def set_sigmoid(self,sigmoid):
        """
        Function to get the process started. The variable 'sigmoid' is a RegularGridInterpolator object
        and can be created with the class SigmoidConstructor.
        """
        self.predictor = RGI_wrap(sigmoid)
           
    def set_manual_links(self,links,positions,intensities):
        self.positions = positions
        self.intensities = intensities
        self.process()
        
        #sort links for increasing time
        links = [sorted(li,key = lambda x: x[0]) for li in links]
        self.links = links
        
    def calculate_classes_and_sigmoid(self):
        self.process()
        self._generate_classes()
        self._calculate_sigmoid()
        self.fit(show=False)
    
    
    def update_positions_and_intensities(self,positions,intensities):
        if self.positions != None:
            self.positions = np.vstack((self.positions,positions))
        else:
            self.positions = positions
        
        if self.intensities != None:
            self.intensities = np.vstack((self.intensities,intensities))
        else:
            self.intensities = intensities    
        
    def get_sigmoid_values(self,new_pos,new_intensities):
        
        PP = PositionProcessing()
        PP.set_data(new_pos,new_intensities)
        
        #calculate combinations
        PP.process()
        
        #calculate sigmoid values for combinations
        X = np.array(zip(*[PP.combi_info[typ] for typ in PP.implemented_info]))
        s = self.predictor.predict(X)
        
        return PP, s
    
class LinkAnalyzer(object):
    
    """
    The purpose of this class is to find the positive links given overlapping time windows and
    corresponding combinations and sigmoid values.
    """
    def __init__(self,sm):
        
        self.sm = sm
        self.targets = None #dict: keys (idx_t_f,idx_pos_f), values = dict keys (idx_t_s,idx_pos_s) values sigmoid val
        self.targets_for_linking = None #subset of self.targets for the current time step which will be removed from self.target
        self.already_fix_heads = None
        self.already_fix_tails = None
              
    def update_targets(self,positions,intensities,t_curr):
        """
        Takes positions, intensities and the current time 't_curr' and adds to 'self.targets'.
        """
        t0 = time.time()
        pp, sigs = self.sm.get_sigmoid_values(positions,intensities)
        print("calc sigmoids and combinations {}s...".format(time.time()-t0))
        t0 = time.time()
        self.sm.update_positions_and_intensities(positions,intensities)
        print("update pos {}s...".format(time.time()-t0))
        
        if self.targets is None:
            self.targets = {}
        
        #adding possible links to self.targets
        t0 = time.time()
        for c,s in zip(pp.combinations,sigs):
            sou = tuple([c[0][0]+t_curr,c[0][1]])
            tar = tuple([c[1][0]+t_curr,c[1][1]])
            if s>.5:
                if not tar in self.targets:
                    self.targets[tar] = {sou:s}
                else:
                    self.targets[tar][sou] = s
        print("updating targets {}s...".format(time.time()-t0))
        
        #splitting all targets into targets for linking now and for next steps
        t0 = time.time()
        self.targets_for_linking = {k: v for k,v in self.targets.items() if t_curr == k[0]} #heads which correspond to the current timestep
        self.targets = {k: v for k,v in self.targets.items() if t_curr < k[0]} #heads which correspond to future timesteps
        print("splitting targets {}s...".format(time.time()-t0))
        
    def solidify_links(self,t_curr,dt=3,link_type='greedy',link_paras={'eps':.1,'tau':-1}):
        """
        Input:
            dt - int (optional), limits the number of trajectories being checked for linking by restricting the 
                latest time stamp they can have to t_curr - dt
        """
        dt = int(dt)
        #solidifies link in the sense that all targets with the current timestamp t_curr
        #are assigned a source and added to self.sm.links
        
        #check for relevant targets
        links = copy.deepcopy(self.sm.links)
        link_values = copy.deepcopy(self.sm.link_values)
        if links is None:
            links = []
            link_values = []
            
        if len(self.targets_for_linking) > 0:
            to_solidify = self.targets_for_linking
            
            #make sure the targets are not already in self.links so that the manual linking 
            #is not overwritten
            
            t0 = time.time()
            already_fix_heads = {k: False for k in itertools.chain(*links)}
            already_fix_tails = {k: False for k in itertools.chain(*[v[:-1] for v in links if v[:-1][0] >= t_curr - dt])}
            print("finding fixed {}s...".format(time.time()-t0))
            
            def heads_wrap(k):
                try:
                    already_fix_heads[k]
                    return False
                except:
                    return True
                
            def tails_wrap(k):
                try:
                    already_fix_tails[k]
                    return False
                except:
                    return True
            t0 = time.time()
            #make sure no targets positions are already fixed
            to_solidify = {k: v for k,v in to_solidify.items() if heads_wrap(k)}
            #make sure no origin positions are already fixed
            to_solidify = {k:{k2:v2 for k2,v2 in v.items() if tails_wrap(k2)} for k,v in to_solidify.items()}
            print("to_solidify {}s...".format(time.time()-t0))
            
            #make a list of all possible links and assign them after sorting for maximum sigmoid value for sigmoids > .5
            t0 = time.time()
            possible_new_links = []
            for head, data in to_solidify.items():
                for tail, s in data.items():
                    if s > .5:
                        possible_new_links += [[(tail,head),s]]
            possible_new_links = sorted(possible_new_links,key = lambda x:[1],reverse=True)            
            print("possible new links {}s...".format(time.time()-t0))
            
            #assign edges to update links while removing assigned heads until possible_new_links is empty
            t0 = time.time()
            previous_heads = [v[-1] for v in links]

            if link_type == 'eps-greedy':
                coin = np.random.uniform(size=len(possible_new_links))
                
            c = 0
            while len(possible_new_links)>0:
                if link_type == 'greedy':
                    link, s = possible_new_links.pop(0) #greedy choice since the list is sorted in descending sigmoid values
                elif link_type == 'eps-greedy':
                    if coin[c] < link_paras['eps']:
                        i = np.random.choice(range(len(possible_new_links)))
                        link, s = possible_new_links.pop(i)
                    else:
                        link, s = possible_new_links.pop(0)
                elif link_type == 'softmax':
                    probs_softmax = softmax([v[1] for v in possible_new_links],link_paras['tau'])
                    i = np.random.choice(range(len(possible_new_links)),p=probs_softmax)
                    link, s = possible_new_links.pop(i)
                
                else:
                    raise "Error - unknown linking type {}, expected: 'greedy', 'eps-greedy' or 'softmax'.".format(link_type)
                tail, head = link
                
                try:
                    idx = previous_heads.index(tail)
                    links[idx] += [head]
                    link_values[idx] += [s]
                except:
                    links += [[tail,head]]
                    link_values += [[s]]

                possible_new_links = [v for v in possible_new_links if v[0][1]!=head and v[0][0]!=tail]
                c += 1 #related to eps-greedy
                
            print("possible new links {}s...".format(time.time()-t0))
            self.sm.links = copy.deepcopy(links)
            self.sm.link_values = copy.deepcopy(link_values)
    
    def get_trajectories(self):
        """
        "values" again in the sense of reinforcement learning's value of an action. Here the sigmoid value for a link.
        
        Returns:
            LM_traj - ndarray shape (Ntraj,Nt), contains index as integer to position at current time step
                        otherwise NaN
            mat_values - ndarray shape (Ntraj,Nt), same form as LM_traj but contains the sigmoid value written 
                        to the tails of the respective links.
        """
        links = self.sm.links
        link_values = self.sm.link_values
                
        num_t = max([max([v[0] for v in v2]) for v2 in links])+1
        num_traj = len(links)
        
        LM_traj = np.zeros((num_traj,num_t))
        mat_values = np.zeros(LM_traj.shape)
        LM_traj[:] = np.nan
        mat_values[:] = np.nan
        for i,(link,values) in enumerate(zip(links,link_values)):
            for (t,j),s in zip(link[:-1],values): #since two positions form one link "link" is one entry longer than "values"
                LM_traj[i,t] = j
                mat_values[i,t] = s
            t,j = link[-1]
            LM_traj[i,t] = j
        return LM_traj, mat_values
        
    def update_sigmoid(self):
        self.sm.calculate_classes_and_sigmoid()
        
def link_positions(loaded_positions,loaded_intensities,splined_sigmoid,dt=3,t_update=5,t_max=None,
                   link_type='greedy',link_paras={'eps':.1,'tau':.1},bin_spec={'r':2,'dI':10,'theta':10,'dt':1},
                   final_eval=True):
    """Convenience function to do the linking.
    
    This function calls SigmoidManager and LinkAnalyzer to construct a sigmoid
    and link positions.
    
    """
    SM = SigmoidManager(bin_spec=bin_spec)
    SM.set_sigmoid(splined_sigmoid)
    LA = LinkAnalyzer(SM)
    ts = time.time()
    if t_max is not None:
        t_max = int(t_max)
    
    for t in range(len(loaded_positions)):
        if t == t_max:
            break
        print("\nt {}".format(t))
        pos = loaded_positions[0+t:dt+t]
        intensity = loaded_intensities[0+t:dt+t]
        tsl = time.time()
        LA.update_targets(pos,intensity,t)
        LA.solidify_links(t,dt=dt,link_type=link_type,link_paras=link_paras)
        print("linking time {}s...".format(time.time() - tsl))
        if t>0 and t%t_update == 0:
            LA.update_sigmoid()
    
    #re-evaluation using the fitted sigmoid
    if final_eval:
        print("Re-evaluting the links using the fitted sigmoid...")
        LA2 = LinkAnalyzer(SM)
        LA2.sm.predictor = LA.sm.predictor
        for t in range(len(loaded_positions)):
            if t == t_max:
                break
            pos = loaded_positions[0+t:dt+t]
            intensity = loaded_intensities[0+t:dt+t]
            tsl = time.time()
            LA2.update_targets(pos,intensity,t)
            LA2.solidify_links(t,dt=dt,link_type=link_type,link_paras=link_paras)
        LM_traj2, link_values = LA2.get_trajectories() 
    else:
        LM_traj2, link_values = LA.get_trajectories() 
    print("total time {}s...".format(time.time() - tsl))
    return LM_traj2, link_values