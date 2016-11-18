from __future__ import print_function

import sys, cPickle, copy
sys.path.append("D:/Anaconda/envs/python2/Lib/site-packages") #you may have to edit this
import cPickle, time, h5py, pywt
import numpy as np
from scipy import fftpack, cluster
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse, Circle, Rectangle
import matplotlib.image as mpimg

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

class ImagePeakClassifier(object):
    """Class to classify windows in frames.
    
    Parameters
    ----------
    
    wavelet : str
        By default set to 'haar' selecting the Haar wavelet for feature calculation.
    """
    
    def __init__(self,wavelet='haar'):
        self.classifier = None
        self.img = None
        self.win_width = None
        self.win_height = None
        self.samples = None
        self.xys = None
        self.classifications = None
        self.intensities = None
        self.centers = None
        self.corners = None
        self.cluster_centers = None
        self.cluster_corners = None
        self.sequence = None
        self.cluster_centers_storage = None
        self.cluster_corners_storage = None
        
        if wavelet != 'haar':
            allowed_wavelets = pywt.wavelist()
            assert wavelet in allowed_wavelets, "Assertion failed - provided wavelet {} not understood, please chose one of the following: {}".format(wavelet,allowed_wavelets)
        self.wavelet = wavelet
                        
    def load_classifier(self,path):
        with open(path,'r') as f:
            self.classifier = cPickle.load(f)
        
    def set_trained_classifier(self,classifier):
        self.classifier = classifier
        
    def set_image_and_windows(self,img,width,height):
        self.img = copy.deepcopy(img)
        self.win_width = width
        self.win_height = height
        
    def sample_to_feature_vec(self,sample,intensity):
        
        feature_vec = list(np.ravel(pywt.dwt(sample,self.wavelet)[0])) + [intensity]
        feature_vec = np.array(feature_vec)
        return feature_vec
    
    def get_predictions(self):
        nsamples = len(self.samples)
        num_features = len(self.sample_to_feature_vec(self.samples[0],self.intensities[0]))
        features = np.zeros((nsamples,num_features))
        
        for i in xrange(nsamples):# sample,intensity in zip(self.samples,self.intensities):
            feature_vec = np.array(list(np.ravel(pywt.dwt(self.samples[i],self.wavelet)[0]))+[self.intensities[i]])
            features[i] = self.sample_to_feature_vec(self.samples[i],self.intensities[i])
        
        print("features {}".format(features.shape))
        return self.classifier.predict(features)
    
    def window(self,xstep,ystep,i,j):
        xi, yj = i*xstep, j*ystep
        win = np.array(self.img[yj:yj+self.win_height-1,xi:xi+self.win_width-1])
        return win,[xi,yj]
    
    def show_window(self,xy=[0,0]):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        plt.imshow(self.img,origin='lower')
        ax = plt.gca()        
        poly = Rectangle(xy,self.win_width,self.win_height,alpha=1,facecolor='none',
                        edgecolor='m',lw=2)
        ax.add_artist(poly)
        plt.show()
        
    def show_window_for_single_img(self,i):
        #i - index for image in stored img sequence (self.sequence)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        plt.imshow(self.sequence[i],origin='lower')
        plt.hold(True)
        for xy in self.cluster_corners_storage[i]:
            ax = plt.gca()        
            poly = Rectangle(xy,self.win_width,self.win_height,alpha=1,facecolor='none',
                            edgecolor='m',lw=2)
            ax.add_artist(poly)
        plt.hold(False)
        plt.show()

    def sliding_window_analysis(self,show=False,xstep=1,ystep=1,iclassy=[0]):
        """
        iclassy - list:
            contains as integers the classes of interest which will be shown if "show" is True
        """
        print('\nAnalyzing...')
        x_wins = int((self.img.shape[1]-self.win_width)/float(xstep))
        y_wins = int((self.img.shape[0]-self.win_height)/float(ystep))
        print('{} frames'.format((x_wins+1)*(y_wins+1)))
        img_count = 0
        self.samples = []
        self.xys = []
        self.classifications = []
        self.intensities = []
        self.centers = []
        
        for i in xrange(x_wins+1):
            for j in xrange(y_wins+1):
                win, xy = self.window(xstep,ystep,i,j)
                
                if win.shape[0]==self.win_height-1 and win.shape[1]==self.win_width-1:
                    self.intensities += [np.sum(win)]
                    self.samples += [win/self.intensities[-1]]
                    self.xys += [xy]
                    img_count += 1
                else:
                    print('xi ',xy[0],' yj ',xy[1])
                    print('sub_img ',self.sub_img.shape,self.sub_img[xy[0]+self.win_height-1,xy[1]+self.win_width-1])
                    print('win ',win.shape,win)
                    print(' xstep ',xstep,' ystep ',ystep)
                    print('i ',i,' j ',j)
                    raise ValueError
        
        self.classifications = self.get_predictions()
        
        positive = 0
        if any([any([v2==v for v in self.classifications]) for v2 in iclassy]):
            for i,(c,xy,win) in enumerate(zip(self.classifications,self.xys,self.samples)):
                if c in iclassy:
                    
                    if show:
                        self.show_window(xy=xy)
                        fig = plt.figure()
                        plt.imshow(win,origin='lower')
                        plt.show()
                    positive += 1
                    self.centers += [[xy[1]+self.win_height*.5,xy[0]+self.win_width*.5]]
            print('{} positive classifications in total'.format(positive))
            
    def get_clusters(self,t):
        
        fdata = cluster.hierarchy.fclusterdata(self.centers,t,criterion='distance',metric='euclidean',method='single')
        cluster_centers = [map(int,np.mean(np.array([vX for vX,f in zip(self.centers,fdata) if f==v]),axis=0)) for v in list(set(fdata))]
        cluster_corners = [[int(j-.5*self.win_width)+1,int(i-.5*self.win_height)+1] for (i,j) in cluster_centers]
        self.cluster_centers = np.array(cluster_centers,dtype=int)
        self.cluster_corners = np.array(cluster_corners,dtype=int)
        
    def get_list_of_cluster_centers(self,separator=' ',c=None):
        if c==None:
            c = self.cluster_centers
        out = [separator.join(map(str,map(int,v))) for v in c]
        return out
    
    def load_TEM_sequence(self,path,h5=True,key='dataset'):
        if h5:
            f = h5py.File(path, 'r')
            imagesequence = f[key]
            return imagesequence
        else:
            raise ValueError('Error - file type does not equal h5!')
    
    def clear_tmp(self):
        """
        Removes data.
        """
        self.img = None
        self.samples = None
        self.xys = None
        self.classifications = None
        self.intensities = None
        self.centers = None
        self.corners = None
        self.cluster_centers = None
        self.cluster_corners = None
                
    def _store_memory_usage(self):
        variables = ['classifier','img','win_width','win_height','samples',
                     'xys','classifications','intensities','centers','corners',
                     'cluster_centers','cluster_corners','sequence','cluster_centers_storage',
                     'cluster_corners_storage']
        if hasattr(self,'memory_usage'):
            for var in variables:
                self.memory_usage[var] += [total_size(getattr(self,var))]
        else:
            self.memory_usage = {}
            for var in variables:
                self.memory_usage[var] = [total_size(getattr(self,var))]
    
    def _print_memory_usage(self):
        variables = ['classifier','img','win_width','win_height','samples',
                     'xys','classifications','intensities','centers','corners',
                     'cluster_centers','cluster_corners','sequence','cluster_centers_storage',
                     'cluster_corners_storage']
        print("\nmemory usage")
        for var in variables:
            print("var {}: {}".format(var,self.memory_usage[var]))
        
    def process_single_TEM_frame(self,frame,width,height,xstep=5,ystep=5,iclassy=[1],
                            t=5):
        t1 = time.time()
        self.clear_tmp()
        if self.cluster_centers_storage == None: 
            self.cluster_centers_storage = []
        if self.cluster_corners_storage == None:
            self.cluster_corners_storage = []
        print('\nProcessing frame {}...'.format(frame))
        self.set_image_and_windows(self.sequence[frame,:,:],width,height)
        self.sliding_window_analysis(show=False,xstep=xstep,ystep=ystep,iclassy=iclassy)
        self.get_clusters(t)
        print("\n {} clusters".format(len(self.cluster_centers)))
        self.cluster_centers_storage += [self.cluster_centers]
        self.cluster_corners_storage += [self.cluster_corners]
        t2 = time.time()
        
        self._store_memory_usage()
    
    
    def process_TEM_sequence(self,path,width,height,xstep=5,ystep=5,iclassy=[1],
                            t=5,num_frames='all',key='/Experiments/__unnamed__/data'):
        print('\nProcessing TEM image sequence...')
        self.sequence = self.load_TEM_sequence(path,key=key)
        
        if num_frames=='all':
            frames = self.sequence.shape[0]
        elif type(num_frames)==int:
            frames = num_frames
        else:
            raise ValueError('Error - {} not understood please either supply "all" or an integer value'.format(num_frames))
                
        for frame in range(frames):
            self.process_single_TEM_frame(frame,width,height,xstep=xstep,ystep=ystep,iclassy=iclassy,t=t)

        self._print_memory_usage()
        
    def save_ClusterCenters2disk(self,path,separator=' '):
        """
        The coordinates written to file should be read as row and column indices of an
        image matrix.
        """
        print('\nWriting cluster centers to disk {}...'.format(path))
        with open(path,'w') as f:
            for i,cluster_centers in enumerate(self.cluster_centers_storage):
                str_cluster_centers = self.get_list_of_cluster_centers(separator=' ',c=cluster_centers)
                f.write('frame {}\n'.format(i))
                for cc in str_cluster_centers:
                    f.write(cc+'\n')
                    
                    
def Process_TEM(img_path,classifier_path,width,height,xstep=5,ystep=5,iclassy=[1],t=5,num_frames='all',key='/Experiments/__unnamed__/data',
                write_path=None,bool_return=True,wavelet='haar'):
                
    Ipc = ImagePeakClassifier(wavelet=wavelet)
    #set classifier
    Ipc.load_classifier(classifier_path)
    classifier = Ipc.classifier
    sequence = Ipc.load_TEM_sequence(img_path,key='/Experiments/__unnamed__/data')
    
    if num_frames=='all':
        idx_frames = range(sequence.shape[0])
    elif type(num_frames)==int:
        idx_frames = range(num_frames)
    elif isinstance(num_frames,list) and all([isinstance(v,int) for v in num_frames]):
        idx_frames = num_frames
    else:
        raise ValueError('Error - {} not understood please either supply "all" or an integer value'.format(num_frames))
            
    centers, corners = [], []
    for frame in idx_frames:
        Ipc_tmp = ImagePeakClassifier(wavelet=wavelet)
        Ipc_tmp.classifier = classifier
        Ipc_tmp.sequence = sequence
        Ipc_tmp.process_single_TEM_frame(frame,width,height,xstep=xstep,ystep=ystep,iclassy=iclassy,t=t)
        centers += [Ipc_tmp.cluster_centers]
        corners += [Ipc_tmp.cluster_corners]
        del Ipc_tmp
    
    if write_path != None:
        Ipc.cluster_centers_storage = centers
        Ipc.save_ClusterCenters2disk(write_path,separator=' ')
    if bool_return:
        return centers, corners