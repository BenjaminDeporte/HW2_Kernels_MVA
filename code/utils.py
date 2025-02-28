import numpy as np

from sklearn import svm, linear_model, discriminant_analysis, metrics
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import seaborn as sns





def scatter_label_points(X,y, ax=None, title=''):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 7))
    colormap = np.array(['r', 'g', 'b'])
    ax.scatter(X[:,0], X[:,1], s=200, c=colormap[y],alpha=0.5)
    ax.set_title(title)
def plot_multiple_images(images, num_row=1, num_col=10):
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col,num_row))
    num = num_row*num_col
    for i in range(num):
        if num_row>1:
            ax = axes[i//num_col, i%num_col]
        else:
            ax = axes[i]
        ax.imshow(images[i], cmap='gray')
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()





def gaussian_mixture(N, mu=0.3, sigma=0.1):
    """ Mixture of two gaussians """
    X = np.random.normal(mu, sigma, (N, 2))
    u = np.random.uniform(0, 1, N) > 0.5
    Y = 2. * u - 1
    X *= Y[:, np.newaxis]
    X -= X.mean(axis=0)
    return X,Y
def generateXor(n, mu=0.5, sigma=0.5):
    """ Four gaussian clouds in a Xor fashion """
    X = np.random.normal(mu, sigma, (n, 2))
    yB0 = np.random.uniform(0, 1, n) > 0.5
    yB1 = np.random.uniform(0, 1, n) > 0.5
    # y is in {-1, 1}
    y0 = 2. * yB0 - 1
    y1 = 2. * yB1 - 1
    X[:,0] *= y0
    X[:,1] *= y1
    X -= X.mean(axis=0)
    return X, y0*y1


def generateMexicanHat(N, stochastic = False):
    xMin = -1
    xMax = 1.
    sigma = .2
    std= 0.1
    if stochastic:
        x = np.random.uniform(xMin, xMax, N)
    else:
        x = np.linspace(xMin, xMax, N)
    yClean = (1- x**2/sigma**2)*np.exp(-x**2/(2*sigma**2))
    y =  yClean + np.random.normal(0, std, N) 
    return x, y,yClean

def generateRings(N):

    N_rings = 3
    Idex = [0,int(N/3),int(2*N/3),N]
    std = 0.1
    
    Radius = np.array([1.,2.,3.])
    y = np.ones(N)
    
    X = np.random.normal(size=(N,2))
    X = np.einsum('ij,i->ij',X,1./np.sqrt(np.sum(X**2,axis=1)))
    for i in range(N_rings):
        X[Idex[i]:Idex[i+1],:] *= Radius[i]
        y[Idex[i]:Idex[i+1]]  = i
    y = y.astype(int)
    return X, y

def loadMNIST(path):
	import gzip
	a_file = gzip.open(path, "rb")
	N = 2000
	image_size = 28
	num_images = 2*N
	buf = a_file.read(image_size * image_size * num_images)
	data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
	data = data.reshape(num_images, image_size, image_size)
	data = data/255.

	clean_train, clean_test = data[:N], data[N:]
	train = clean_train + np.random.normal(loc=0.0, scale=0.5, size=clean_train.shape)
	test = clean_test + np.random.normal(loc=0.0, scale=0.5, size=clean_test.shape)

	data = {'cleanMNIST': {'train':clean_train , 'test':clean_test},
			'noisyMNIST': {'train':train , 'test':test},
			}
	return data













