import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import itertools, collections
from sklearn import mixture, decomposition
from sklearn import cluster as sk_cluster
from matplotlib.patches import Ellipse, Circle, Rectangle
from scipy import fftpack, cluster
import h5py, copy
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

def show_all_subimgs(samp,clas,target_classification=1):
    
    c = 0
    for i,sample in enumerate(samp):

        if clas[i] == target_classification:
            fig = plt.figure()
            plt.imshow(sample,origin='lower', interpolation='none', cmap=plt.cm.gray)
            c += 1
            plt.title("sample {} - num {}".format(i,c))

def load_TEM_image(path,h5=False,frame=0,key='dataset'):
    if h5:
        frame = int(frame)
        f = h5py.File(path, 'r')
        imagesequence = f[key]
        return imagesequence[frame,:,:]
    else:
        return mpimg.imread(path)

def Show_group_assignment_windows(img,xys,classifications,width,height,colors=['r','g','y','m'],patch_w=7,patch_h=5):
    # select individual peaks and their distributions

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.imshow(img,origin='lower', interpolation='none', cmap=plt.cm.gray)
    plt.hold(True)
    assert len(classifications)==len(xys), "Assertion failed - unequal number of classifications {} != {} xys".format(len(classifications),len(xys))
    for i,(c,ji) in enumerate(zip(classifications,xys)):

        if ax is None:
            ax = plt.gca()
        xy = [ji[1],ji[0]]
        poly = Rectangle(xy,width,height,alpha=1,facecolor='none',
                        edgecolor=colors[int(c%len(colors))],lw=2.,label=i)
        p = patches.Rectangle((xy[0]+width+.5,xy[1]), patch_w, patch_h,fill=True,color='w')
        ax.add_patch(p)
        ax.text(xy[0]+width+.5,xy[1],str(i),color=colors[int(c%len(colors))],fontsize=12)
        ax.add_artist(poly)
    plt.hold(False)
    plt.legend(loc=0)
    plt.title('Classifications')
    plt.show()

class TEM_sliding_collector(object):
    def __init__(self,patch_w=7,patch_h=5):
        self.img_vec = None
        self.sub_img_vec = None
        self.win_width = None
        self.win_height = None

        #classification related
        self.samples = None
        self.xys = None
        self.classifications = None
        self.centers = None
        self.classes = {"positive":1,"negative":0}

        #plotting related
        self.patch_w = patch_w #width of white patch for window enumeration
        self.patch_h = patch_h #height of white patch for window enumeration

    def save_sample_positions_and_class(self,path):
        """
        frame x y positive/negative
        """
        print("Writing samples for classifier training to disk at {}...".format(path))
        with open(path,'w') as f:
            f.write("frame x y positive/negative\n")
            for i in range(len(self.xys)):
                for c, pos in zip(self.classifications[i],self.xys[i]):
                    x = pos[1] + .5*self.win_width
                    y = pos[0] + .5*self.win_height 
                    f.write("{} {} {} {}\n".format(i,x,y,c))
                    
    def load_sample_positions_and_class(self,path):
        print("Loading samples for classifier training from disk at {}...".format(path))
        self.xys = []
        self.classifications = []
        self.samples = []
        assert (self.win_width != None and self.win_height != None), "Assertion failed - window width and heigh are not specified!"
        assert (self.sub_img_vec != None), "Assertion failed - no stack of images present for processing!"
        with open(path,'r') as f:
            tmp_xys = []
            tmp_cs = []
            tmp_samples = []
            lines = f.readlines()[1:]
            num_lines = len(lines)
            for ix, line in enumerate(lines):
                
                t, x, y, c = map(float,line.rstrip('\n').split())
                t, x, y = int(t), x, y
                ji = [int(y - .5*self.win_height), int(x - .5*self.win_width)]
                tmp_xys += [ji]
                tmp_cs += [c]
                tmp_samples += [np.array(self.sub_img_vec[t][ji[0]:ji[0]+self.win_height-1,ji[1]:ji[1]+self.win_width-1])]
                
                if not 't0' in locals():
                    t0 = int(t)    
                if t!=t0: #new frame
                    t0 = int(t)
                    self.xys += [copy.deepcopy(tmp_xys[:-1])]
                    self.classifications += [copy.deepcopy(tmp_cs[:-1])]
                    self.samples += [copy.deepcopy(tmp_samples[:-1])]
                    tmp_xys = [tmp_xys[-1]]
                    tmp_cs = [tmp_cs[-1]]
                    tmp_samples = [tmp_samples[-1]]
                elif ix == num_lines-1: #eof
                    self.xys += [copy.deepcopy(tmp_xys)]
                    self.classifications += [copy.deepcopy(tmp_cs)]
                    self.samples += [copy.deepcopy(tmp_samples)]
                     
    
    def set_image(self,img):
        self.img_vec = img

    def get_sub_image_vec(self,xlim=[0,100],ylim=[0,100],padding=5):
        self.sub_img_vec = []
        for index in xrange(len(self.img_vec)):
            sub_img = self.img_vec[index][ylim[0]:ylim[1],xlim[0]:xlim[1]]
            if type(padding)==int:
                self.sub_img_vec += [sub_img]

    def set_window_size(self,width=15,height=10):
        self.win_width = width
        self.win_height = height

    def show_window(self,index=0,sub=True,xy=[0,0]):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if sub==True:
            plt.imshow(self.sub_img_vec[index],origin='lower', interpolation='none', cmap=plt.cm.gray)
        else:
            plt.imshow(self.img,origin='lower', interpolation='none', cmap=plt.cm.gray)
        
        if ax is None:
            ax = plt.gca()


        width = self.win_width
        height = self.win_height

        poly = Rectangle(xy,width,height,alpha=1,facecolor='none',
                        edgecolor='r',lw=2)
        ax.add_artist(poly)
        plt.title("xy {}".format(xy))
        plt.show()

    def show_images(self):
        for i,img in enumerate(self.img_vec):
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)

            plt.imshow(self.img_vec[i],origin='lower', interpolation='none', cmap=plt.cm.gray)
            plt.title('frame #{}'.format(i))
            plt.show()

    def window(self,sub_img,xstep,ystep,i,j):
        xi, yj = i*xstep, j*ystep
        win = np.array(sub_img[yj:yj+self.win_height-1,xi:xi+self.win_width-1])
        return win,[xi,yj],[yj,xi]

    def sliding_window(self,ix,xstep=1,ystep=1,show=False):
        print '\nGenerating samples with the sliding window technique...'
        known_centers = copy.deepcopy(self.centers[ix])
        known_classifications = copy.deepcopy(self.classifications[ix])
        #select centers which are associated with positive classifications for kd tree creation
        positive_centers = np.array([v for i,v in enumerate(known_centers) if known_classifications[i]==self.classes["positive"]])
        print("positive_centers {}".format(positive_centers.shape))
        kdt = KDTree(positive_centers)
        r_cut = np.sqrt(self.win_width**2+self.win_height**2)
        
        sub_img = copy.deepcopy(self.img_vec[ix])
        x_wins = int((sub_img.shape[1]-self.win_width)/float(xstep))
        y_wins = int((sub_img.shape[0]-self.win_height)/float(ystep))
        
        img_count = 0
        samples = []
        xys = []
        corners = []
        centers = []
        
        for i in xrange(x_wins+1):
            for j in xrange(y_wins+1):
                win, xy, ji = self.window(sub_img,xstep,ystep,i,j)
                
                if win.shape[0]==self.win_height-1 and win.shape[1]==self.win_width-1: #sanity check of window dimensions
                    cen = ([int(ji[0]+.5*self.win_height)-1,int(ji[1]+.5*self.win_width)-1])
                    neighbors = kdt.query_ball_point(cen,r_cut)
                    
                    if len(neighbors)==0: #sliding window far enough away from any known positive windows
                        samples += [win]
                        xys += [ji]
                        centers += [cen]
                        
                    if show:

                        fig = plt.figure(figsize=(10,10))
                        img = self.img_vec[ix]
                        plt.hold(True)
                        plt.imshow(img,origin='lower', interpolation='none', cmap=plt.cm.gray)
                        plt.plot([xy[1]],[xy[0]],'ro',label='new negative')
                        plt.plot(positive_centers[:,1],positive_centers[:,0],'bd',label='known positive')
                        plt.hold(False)
                        plt.legend(loc=0)
                        plt.show()
                    img_count += 1
                    
                else:
                    print 'xi ',xy[0],' yj ',xy[1]
                    print 'sub_img ',sub_img.shape,sub_img[xy[0]+self.win_height-1,xy[1]+self.win_width-1]
                    print 'win ',win.shape,win
                    print ' xstep ',xstep,' ystep ',ystep
                    print 'i ',i,' j ',j
                    raise ValueError
                        
        return xys, samples, centers, [self.classes["negative"]]*len(xys)

    def _flatten_intensity_diff_stack(self,stack):
        """
        Holy cow this works! Aim is to reshape the stack of dim (num neighbor pixels used for intensity difference + 1, img.shape[0], img.shape[1])
        into dim (img.shape[0] * img.shape[1], num neighbor pixels used for intensity difference + 1)
        so the stack can be used as a feature stack.
        """
        new_stack = np.reshape(np.ravel(stack),(9,stack.shape[1]*stack.shape[2]))
        return new_stack.T

    def _flattened_stack_to_img(self,X,s):
        img = np.reshape(X,s)
        return img

    def _img_difference_features(self,ix,distance=1,periodic=False):
        """

        Input:
            distance - float (optional), defines the number of neighboring pixels
                taken into account for difference calculation
            periodic - bool (optional), if True the images are treated as if they
                had periodic boundary conditions maintaining the array shape. If False the
                frame of the image (as thick as 'distance') will be removed.
        """
        assert isinstance(distance,int), "Assertion failed - expected 'distance' to be of type int, got {} instead.".format(type(distance))
        img = self.img_vec[ix]
        idx = np.arange(-distance,distance+1,1)
        idx.astype('i4')
        rolling_info = [v for v in list(itertools.product(idx,idx)) if v!=(0,0)]
        rolled_imgs = [np.roll(img,v[0],axis=0) for v in rolling_info]
        rolled_imgs = [np.roll(vimg,v[1],axis=1) for v,vimg in zip(rolling_info,rolled_imgs)]
        intensity_diff_stack = np.array([img]+[np.add(img,-vimg) for vimg in rolled_imgs])

        if not periodic:
            intensity_diff_stack = intensity_diff_stack[:,distance:-distance,distance:-distance]

        flat_intensity_diff_stack = self._flatten_intensity_diff_stack(intensity_diff_stack)
        return flat_intensity_diff_stack

    def _decompose_img_pca_and_km(self,ix,distance,n_clusters,max_iter=300,tol=1e-4,show=False):
        pca = decomposition.PCA()
        X = self._img_difference_features(ix,distance=distance)
        pca.fit(X)
        X = pca.transform(X)

        km = sk_cluster.KMeans(n_clusters=n_clusters,n_init=10,max_iter=300,tol=1e-4)
        km.fit(X)
        X_km = km.predict(X)

        s = (self.img_vec[0].shape[0]-2*distance,self.img_vec[0].shape[1]-2*distance)
        img_km = self._flattened_stack_to_img(X_km,s)

        if show:
            fig = plt.figure(figsize=(10,10))
            plt.subplot(121)
            i=0
            plt.imshow(self.img_vec[i],origin='lower', interpolation='none', cmap=plt.cm.gray)
            plt.title('frame #{}'.format(i))
            plt.subplot(122)
            plt.imshow(img_km,origin='lower', interpolation='none', cmap=plt.cm.gray)
            plt.title('KM img')
            plt.show()

        #key = cluster id, values image indices
        cluster_idx = {k: np.where(img_km==k) for k in set(X_km)}

        #sorted clusters for ascending average intensity per pixel -> list of tuples (cluster id, image indices)
        cluster_sorted = sorted([k for k in cluster_idx.items()],key=lambda x: np.sum(self.img_vec[ix][x[1]])/len(x[1][0]))
        
        if show:
            for i2,(cks,vals) in enumerate(cluster_sorted):
                img_part = np.zeros(self.img_vec[i].shape)
                img_part[cluster_idx[cks]] = self.img_vec[i][cluster_idx[cks]]

                fig = plt.figure()
                plt.title("intensity {}".format(i2))
                plt.imshow(img_part,origin='lower', interpolation='none', cmap=plt.cm.gray)
                plt.show()
        return cluster_sorted

    def _agglomerate_clusters(self,ix,cluster_sorted,t=1,cluster_min_size=9,show=False):
        """
        To deal with input from self._decompose_img_pca_and_km.
        """
        
        #positive samples
        positive_samples = cluster_sorted[-1][1] #highest average pixel intensity cluster #img coords ij
        
        X = np.array(positive_samples).T
        fdata = cluster.hierarchy.fclusterdata(X,t,criterion='distance',metric='euclidean',method='single')
        centers = [[len(np.where(fdata==v)[0]),np.mean(X[np.where(fdata==v)],axis=0).astype('i4')] for v in list(set(fdata))]
        centers = [v[1] for v in centers if v[0]>=cluster_min_size]

        corners = [([int(vi-.5*self.win_height)+1,int(vj-.5*self.win_width)+1]) for (vi,vj) in centers] #corners in img coords ij
        new_corners = sorted(corners,key=lambda x: (x[0],x[1])) #sorts to create order along axis 0 and then axis 1
        idx = [corners.index(v) for v in new_corners]
        
        corners = [corners[v] for v in idx]
        centers = [centers[v] for v in idx]
        windows = [self.img_vec[ix][j:j+self.win_height-1, i:i+self.win_width-1] for j,i in corners]
        classifications = [self.classes["positive"]]*len(windows)
        
        if show:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            plt.hold(True)
            img_part = np.zeros(self.img_vec[ix].shape)
            img_part[positive_samples] = self.img_vec[ix][positive_samples]
            plt.imshow(img_part,origin='lower', interpolation='none', cmap=plt.cm.gray)
            for i,ji in enumerate(corners):
                if ax is None:
                    ax = plt.gca()
                xy = [ji[1],ji[0]]
                poly = Rectangle(xy,self.win_width,self.win_height,alpha=1,facecolor='none',
                                edgecolor='m',lw=2)
                ax.add_artist(poly)
                p = patches.Rectangle((xy[0]+self.win_width+.5,xy[1]), self.patch_w, self.patch_h,fill=True,color='w')
                ax.add_patch(p)
                ax.text(xy[0]+self.win_width+.5,xy[1],str(i),color='r',fontsize=12)

            plt.hold(False)
            plt.title("High intensity signals clustered frame {}".format(ix))
            plt.show()
        return corners, windows, centers, classifications

    def show_all_sliding_window_classifications(self,default_class=0,colors=['r','g','y','m'],frames=[]):

        if self.classifications == None:
            classifications = [[default_class]*len(v) for v in self.samples]
        else:
            classifications = self.classifications

        for i, xys in enumerate(self.xys):
            pic = self.img_vec[i]
            if frames != []:
                print 'image {} frame {}'.format(i,frames[i])
            else:
                print 'image {}'.format(i)
            Show_group_assignment_windows(pic,xys,classifications[i],
                                    self.win_width,self.win_height,colors=colors)

    def set_classifications(self,classifications):
        """
        This function swaps the current self.classifications values for the passesd "classifications"
        list.
        
        Input:
            - classifications - list of lists of int. each entry indicates a false current classification
                                as in self.classifications. Note that each frame of "self.classifications"
                                has to be represented in form of a list, if nothing is to change, then 
                                an empty list for this frame will suffice.
        """
        assert isinstance(classifications,list), "Assertion failed - expected 'classifications' to be of type list, got {} instead{}.".format(type(classifications))
        
        for t,idx in enumerate(classifications):
            for ix in idx:
                if self.classifications[t][ix] == self.classes["negative"]:
                    self.classifications[t][ix] = self.classes["positive"]
                elif self.classifications[t][ix] == self.classes["positive"]:
                    self.classifications[t][ix] = self.classes["negative"]
                else:
                    raise ValueError("Error - self.classifications[{}][{}] contains unexpected class {}. Expected one of these classes: {}".format(t,ix,self.classifications[t][ix],self.classes.values()))

    def create_samples(self,n_frames,n_clusters,distance,max_iter=300,tol=1e-4,t=1,cluster_min_size=9,
                      xstep=25,ystep=25):
        self.xys = [[]]*n_frames
        self.samples = [[]]*n_frames
        self.centers = [[]]*n_frames
        self.classifications = [[]]*n_frames

        for ix in range(n_frames):
            cluster_sorted = self._decompose_img_pca_and_km(ix,distance,n_clusters,max_iter=max_iter,tol=tol,show=False)
            xys, samples, centers, classifications = self._agglomerate_clusters(ix,cluster_sorted,t=t,cluster_min_size=cluster_min_size,show=True)
            print("\nagglomeration: xys {} centers {}, classifications {}".format(len(self.xys[ix]),len(self.centers[ix]),len(self.classifications[ix])))
            print("xys {} centers {} samples {} classifications {}".format(len(xys),len(centers),len(samples),len(classifications)))
            self.xys[ix] = xys
            self.centers[ix] = centers
            self.samples[ix] = samples
            self.classifications[ix] = classifications
            xys_s, samples_s, centers_s, classifications_s = self.sliding_window(ix,xstep=xstep,ystep=ystep,show=False)
            print("xys {} centers {} samples {} classifications {}".format(len(xys_s),len(centers_s),len(samples_s),len(classifications_s)))
            print("sliding: xys {} centers {}, classifications {}".format(len(self.xys[ix]),len(self.centers[ix]),len(self.classifications[ix])))
            
            #update\
            self.xys[ix] += xys_s
            self.centers[ix] += centers_s
            self.samples[ix] += samples_s
            self.classifications[ix] += classifications_s
            
    def multiply_all_samples(self,distance=1,target_class=1): #,deviation=1):
        """
        Input:
            distance - int (optional), maximum displacement of the window along a single dimension
            target_class - int (optional), defines the positive class and creates multiple samples for it. but 
                    also creates samples for negative samples which simply is assumed to be every other class.
        """
        d = np.arange(-distance,distance+1,1)
        deviations = [v for v in list(itertools.product(d,d)) if v!=(0,0)]
        
        m_xys = []
        m_samples = []
        m_classifications = []
        for i in xrange(len(self.xys)):
            img = self.img_vec[i]

            positive_xys = [self.xys[i][iv] for iv in xrange(len(self.xys[i])) if self.classifications[i][iv]==target_class]
            n_xys = [[np.add(dev,vxy).astype('i4') for dev in deviations] for vxy in positive_xys]
            n_xys = [x for y in n_xys for x in y]
            n_xys = [v for v in n_xys if img.shape[0]>v[0]+self.win_width and v[0]>=0 and img.shape[1]>v[1]+self.win_height and v[1]>=0]
            n_samples = [img[yj:yj+self.win_height-1,xi:xi+self.win_width-1] for yj,xi in n_xys]
            n_classifications = [1]*len(n_samples)

            negative_xys = [self.xys[i][iv] for iv in xrange(len(self.xys[i])) if self.classifications[i][iv]!=target_class]
            n_xys_neg = [[np.add(dev,vxy).astype('i4') for dev in deviations] for vxy in negative_xys]
            n_xys_neg = [x for y in n_xys_neg for x in y]
            n_xys_neg = [v for v in n_xys_neg if img.shape[0]>v[0]+self.win_width and v[0]>=0 and img.shape[1]>v[1]+self.win_height and v[1]>=0]
            n_samples_neg = [img[yj:yj+self.win_height-1,xi:xi+self.win_width-1] for yj,xi in n_xys_neg]
            n_classifications_neg = [0]*len(n_samples_neg)

            m_xys += [n_xys+n_xys_neg]
            m_samples += [n_samples+n_samples_neg]
            m_classifications += [n_classifications+n_classifications_neg]

        for i in xrange(len(m_samples)):
            self.samples[i] += m_samples[i]
            self.xys[i] += m_xys[i]
            self.classifications[i] += m_classifications[i]

        num_positive_samples = int(sum([sum(v) for v in self.classifications]))
        num_total_samples = int(sum([len(v) for v in self.classifications]))
        num_negative_samples = num_total_samples - num_positive_samples
        print("Created amount of samples:\n{} positive samples\n{} negative samples\n{} number of samples in total".format(num_positive_samples,num_negative_samples,num_total_samples))

    def get_all_samples_and_classifications(self):
        print("Getting samples...")
        all_samples = [x for y in copy.deepcopy(self.samples) for x in y]
        all_classifications = [x for y in copy.deepcopy(self.classifications) for x in y]
        
        idx = [iv for iv,sample in enumerate(all_samples) if sample.shape[0]*sample.shape[1]==(self.win_width-1)*(self.win_height-1)]
        all_samples = [all_samples[v] for v in idx]
        all_classifications = [all_classifications[v] for v in idx]
        return all_samples, all_classifications

if __name__=="__main__":
    path = './TEM_images/FEI_HAADF_Image_movie_282-CLEANED-subtracted.h5'#BackgroundSubtracted.h5'
    frames = range(10)
    xlim = [0,240]
    ylim = [0,280]
    width=14
    height=8
    xstep=25 #shift of pixels in x per iteration
    ystep=25 #shift of pixels in y per iteration
    padding=5
    distance = 1
    ix = 0
    n_clusters = 4
    n_frames_pca_km = 2

    img_vec = [load_TEM_image(path,h5=True,frame=v) for v in frames]
    Tsc = TEM_sliding_collector()
    Tsc.set_window_size(width=width,height=height)
    Tsc.set_image(img_vec)

    Tsc.create_samples(n_frames_pca_km,n_clusters,distance,xstep=xstep,ystep=ystep,max_iter=300,tol=1e-4,t=1,cluster_min_size=9)
    Tsc.show_all_sliding_window_classifications()
