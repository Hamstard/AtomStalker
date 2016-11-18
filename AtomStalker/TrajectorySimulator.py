import numpy as np
from scipy import optimize
import matplotlib.pylab as plt
import collections, copy, itertools

class TrajectorySource(object):
    """
    Class to generate initial trajectories for linkage inference as well as for continued production
    of trajectories feeding into a "live" linking process.
    """
    
    def __init__(self,p_nd,n,ran_dis_spec,dim=2,intensity_choices=[1.],ran_int_spec={'paras':{'delta':0.1},'type':'uniform'}):
        """
        Input:
            p_nd - float, within [0,1] = probability of non-disappearance for an atom of a trajectroy from one step to the next
            n - int, maximum number of positions per step
            ran_dis_spec - dict, defines the generation of spatial displacements. 
                expected structure: {'paras':{'mu':0,'sig':1.},'type':'gaussian'}
            dim - int (optional), defines the spatial dimensionality of the problem
        """
        
        #trajectory generation related
        self.implemented_ran_dis = ['gaussian','cubic grid']
        assert isinstance(p_nd,(int,float)) and (0. <= p_nd <= 1.), "Assertion failed - p_nd is not within [0,1]. p_nd = {}".format(p_nd)
        assert isinstance(n,(int,float)) and n>0, "Assertion failed - expected int or float value (to convert to int) for 'n' greater than 0, got {}".format(n)
        assert ('paras' in ran_dis_spec) and ('type' in ran_dis_spec) and isinstance(ran_dis_spec['paras'],dict) and isinstance(ran_dis_spec['type'],str), "Assertion failed - expected structure {'paras':[1.,0.],'type':'gaussian'} for ran_dis_spec with type any of {}, got {} instead.".format(self.implemented_ran_dis,ran_dis_spec)
        self.p_nd = p_nd
        self.n = n
        self.ran_dis_spec = ran_dis_spec
        self.dim = dim
        self.intensity_choices = intensity_choices
        self.ran_int_spec = ran_int_spec
        
        #trajectory related
        self.positions = None
        self.LM_traj = None
        self.intensities = None
        
    def generate_intensity_deviation(self):
        ran_int_spec={'paras':{'delta':0.1},'type':'uniform'}
        
        if self.ran_int_spec['type'] == 'uniform':
            dx = self.ran_int_spec['paras']['delta']*.5
            deviation = np.random.uniform(low=-dx,high=dx,size=self.n)
        else:
            raise ValueError("Error - got unexpected random generator type for intensity noise  {}".format(self.ran_int_spec['type']))
        return deviation
    
    def _get_initial_frame(self,bounds):
        
        pos = np.array([np.random.uniform(low=bounds[v][0],high=bounds[v][1],size=self.n) for v in xrange(self.dim)])
        to_skip = np.random.random(self.n)    
        pos[:,np.where(to_skip>self.p_nd)] = np.nan
        intensities = np.random.choice(self.intensity_choices,size=self.n) + self.generate_intensity_deviation()
        return pos, intensities
    
    def generate_displacements(self):
        
        if self.ran_dis_spec['type'] == 'gaussian':
            deviation = np.random.normal(scale=self.ran_dis_spec['paras']['sig'],loc=self.ran_dis_spec['paras']['mu'],size=(self.dim,))
            signs = np.random.choice(np.array([-1,1]))
            deviation *= signs
        elif self.ran_dis_spec['type'] == 'cubic grid':
            deviation = np.random.choice(np.array([0,1]),size=(self.dim,))
            signs = np.random.choice(np.array([-1,1]))
            deviation *= signs
            
        else:
            raise ValueError("Error - got unexpected random generator type for displacements {}".format(self.ran_dis_spec['type']))
        return deviation
        
    def update_positions(self,bounds):
        """
        Expects ndarray of shape (Nt,dim,nmax) and extends along the first dimension of the ndarray.
        
        Input:
            pos - ndarray of shape (Nt,dim,nmax) containing particle positions
            
        Returns:
            new_pos - ndarray of shape (Nt+1,dim,nmax) containing particle positions
        """
        pos = self.positions
        intensities = self.intensities
        
        #call generate_displacements using the supplied info in ran_dis_spec
        last_steps = [None]*self.n #index to timestep which is the last non nan value for the given trajectory
        for i in xrange(self.n):
            not_nan = np.where([not v for v in np.isnan(pos[:,0,i])])[0]
                        
            #if most recent step is not nan or any of the other write down last timestep with non nan value
            #is only nan then last_step entry is 'None'
            if len(not_nan) > 0:
                last_steps[i] = not_nan[-1]
        
        #update all recent positions which are not skipped (i.e. pos val == nan)
        new_pos = np.zeros((pos.shape[0]+1,pos.shape[1],pos.shape[2]))
        new_pos[:-1] = pos
        new_pos[-1,:,:] = np.array([pos[val,:,i] if val!=None else [np.nan,np.nan] for i,val in enumerate(last_steps)]).T
        new_int = np.zeros((intensities.shape[0]+1,intensities.shape[1]))
        new_int[:-1] = intensities
        
        
        Nt = pos.shape[0] #new timestep
        if None in last_steps:
            new_initial, _ = self._get_initial_frame(bounds)
        deviations = self.generate_displacements()
        intensity_deviations = self.generate_intensity_deviation()
        new_int[-1,:] = intensities[-1,:] + intensity_deviations
        
        for i,val in enumerate(last_steps):
            if val==None: #case that no initial position exists yet
                new_pos[-1,:,i] = new_initial[:,i]
            else: #case that previous position exists but is more than step ago
                
                if np.random.uniform() > self.p_nd: #throwing the dice whether or not the position in the next move will be known
                    nan_array = np.zeros((self.dim,))
                    nan_array[:] = np.nan
                    new_pos[-1,:,i] = nan_array
                else:
                    n_dis = Nt - val #is 1 if position at previous timestep present or larger if steps were skipped
                    dis = [self.generate_displacements() for v in xrange(n_dis)]
                    dis = np.array(reduce(lambda x,y: x+y,dis))
                    new_pos[-1,:,i] += dis
                     
        return new_pos, new_int
            
            
    def generate_initial(self,Nt,bounds=[(0,1),(0,1)]):
        """
        Input:
            Nt - int, number of steps to generate including the initial
            bounds - list of tuples of floats or ints (optional), the bounds for the initial frame where each tuple corresponds to one dimension in space
            
        """
        print("Simulating {} initial steps...".format(Nt))
        #initial positions
        x0, I0 = self._get_initial_frame(bounds)
        self.positions = np.array([x0])
        self.intensities = np.array([I0])
        
        for i in xrange(Nt-1):
            self.positions, self.intensities = self.update_positions(bounds)
            
    def generate_more_steps(self,Nt,bounds=[(0,1),(0,1)]):
        print("Simulating {} more steps...".format(Nt))
        for i in xrange(Nt):
            self.positions, self.intensities = self.update_positions(bounds)
    
    def generate_LM_traj(self):
        
        LM_traj = np.zeros((self.n,self.positions.shape[0]))
        for i in xrange(self.n):
            LM_traj[i,:] = i
            for t in xrange(self.positions.shape[0]):
                if np.isnan(self.positions[t,0,i]):
                    LM_traj[i,t] = np.nan
        self.LM_traj = LM_traj
    
    def get_positions(self,shuffle=False):
        """
        Return positions as well as original Linkage Matrix (LM).
        """
        self.generate_LM_traj()
        if shuffle:
            for i in xrange(self.positions.shape[0]):
                idx_shuffle = np.arange(self.positions.shape[2])
                np.random.shuffle(idx_shuffle)
                self.positions[i,:,:] = self.positions[i,:,(idx_shuffle)].T
                self.intensities[i] = self.intensities[i,(idx_shuffle)]
                self.LM_traj[:,i] = self.LM_traj[(idx_shuffle),i]
            new_LM_traj = np.zeros(self.LM_traj.shape)
            new_LM_traj[:] = np.nan
            for j,config in enumerate(self.LM_traj.T):
                for i,ix in enumerate(config):
                    if ix==ix:
                        new_LM_traj[ix,j] = i
                    
            self.LM_traj = new_LM_traj
        return self.positions, self.LM_traj, self.intensities
        
def coordinates_interpreter(path):
    """
    Reads the coordinates file produced by ImagePeakClassifier.
    """
    with open(path,'r') as f:
        lines = map(lambda x: x.rstrip('\n'),f.readlines())
    positions = []
    for i,line in enumerate(lines):
        if 'frame' in line and i>0:
            positions += [pos]
            pos = []
        elif i==0 and 'frame' in line:
            pos = []
        else:
            pos += [map(int,line.split())]
    print("num frames {} num positions each frame {}".format(len(positions),[len(v) for v in positions]))
    max_num_pos = max([len(v) for v in positions])
    num_t = len(positions)
    arr_positions = np.zeros((num_t,2,max_num_pos))
    arr_positions[:] = np.nan
    for i,pos in enumerate(positions):
        for j,particle in enumerate(pos):
            arr_positions[i,:,j] = np.array(particle)
    intensities = np.ones((num_t,max_num_pos))
    return arr_positions, intensities