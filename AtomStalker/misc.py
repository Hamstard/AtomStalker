import matplotlib.pylab as plt
import itertools
import numpy as np

def show_trajectories(positions,LM_traj,intensities,LM_traj2=None,legend=True):
    fig = plt.figure()
    if LM_traj2!=None:
        plt.subplot(121)
    plt.hold(True)
    x,y = list(itertools.chain(*[v[0,:] for v in positions])), list(itertools.chain(*[v[1,:] for v in positions]))
    plt.scatter(x,y,marker='o',edgecolor='b',linestyle='dashed',alpha=0.3)
    for i,idx in enumerate(LM_traj):
        pos = [positions[t,:,int(ix)] for t,ix in enumerate(idx) if ix==ix]
        x, y = [v[0] for v in pos], [v[1] for v in pos]
        plt.plot(x,y,'-',linewidth=2.,alpha=0.4,label=str(i))
    plt.hold(False)
    if legend: plt.legend(loc=0)
    if LM_traj2!=None:
        plt.title('traj1')
    
    if LM_traj2!=None:
        
        plt.subplot(122)
        plt.hold(True)
        x,y = list(itertools.chain(*[v[0,:] for v in positions])), list(itertools.chain(*[v[1,:] for v in positions]))
        plt.scatter(x,y,marker='o',edgecolor='b',linestyle='dashed',alpha=0.3)
        for i,idx in enumerate(LM_traj2):
            pos = [positions[t,:,int(ix)] for t,ix in enumerate(idx) if ix==ix]
            x, y = [v[0] for v in pos], [v[1] for v in pos]
            plt.plot(x,y,'-',linewidth=2.,alpha=0.4,label=str(i))
        plt.hold(False)
        if legend: plt.legend(loc=0)
        plt.title('traj2')
    plt.show() 
    
def show_values_vs_t(cumulative=True,legend=True,**kwargs):
    fig = plt.figure()
    plt.hold(True)
    for link_type, link_values in kwargs.items():
        ts = range(link_values.shape[1])
        q = []
        
        for t in ts:
            non_nan = link_values[:,t]
            non_nan = [v for v in non_nan if v==v]
            q += [sum(non_nan)]
        if cumulative:
            plt.plot(ts,np.cumsum(q),'-',label=link_type)
        else:
            plt.plot(ts,q,'-',label=link_type)
        print("{}: q {},\nt {}".format(link_type,q,ts))
    plt.hold(False)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('Cumulative q')
    if legend:
        plt.legend(loc=0)
    plt.title("Cumulative q for current and all past steps")
    plt.show()
    
def write_trajectories(path,LM_traj,positions,intensities):
    """
    Format:
    trajectory 0
    frame x y I
    0 39.964 58.77 1
    """
    with open(path,'w') as f:
        for i, traj_idx in enumerate(LM_traj):
            
            f.write("trajectory {}\n".format(i))
            f.write("frame x y I\n")
            for j, ix in enumerate(traj_idx):
                if ix==ix:
                    ix = int(ix)
                    x = positions[j][0][ix]
                    y = positions[j][1][ix]
                    I = intensities[j][ix]
                    f.write("{} {} {} {}\n".format(j,x,y,I))