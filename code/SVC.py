import numpy as np
import pickle as pkl
from scipy import optimize
from scipy.optimize import LinearConstraint
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from utils import plotClassification, plotRegression, plot_multiple_images, generateRings, scatter_label_points, loadMNIST
# %matplotlib inline    # notebook only

    
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import seaborn as sns

file = open('code/datasets/classification_datasets', 'rb')
datasets = pkl.load(file)
# file.close()
# fig, ax = plt.subplots(1,3, figsize=(20, 5))

# for i, (name, dataset) in enumerate(datasets.items()):    
#     plotClassification(dataset['train']['x'], dataset['train']['y'], ax=ax[i])
#     ax[i].set_title(name)
    
class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        squared_distances = np.array([ [ np.sum((X[i,:]-Y[j,:])**2) for j in range(Y.shape[0]) ] for i in range(X.shape[0])])
        return np.exp( -1/(2*self.sigma**2) * squared_distances )
    
class Linear:
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return  X @ Y.T
    
class KernelSVC:
    
    def __init__(self, C, kernel, type='non-linear', epsilon = 1e-3):
        self.type = type
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None # support vectors
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        self.X = X
        self.y = y
        # compute gram matrix, we might need it :-)
        self.gram = self.kernel(X,X)
        # vector of ones, size N
        self.ones = np.ones(N)
        # matrix NxN of y_i on diagonal
        self.Dy = np.diag(y)

        # Lagrange dual problem
        def loss(alpha):
            objective_function = 1/2 * alpha @ self.gram @ alpha - alpha @ self.y
            return  objective_function

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            gradient = self.gram @ alpha - self.y
            return gradient
        
        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        # equality constraint
        fun_eq = lambda alpha: alpha @ self.ones      
        jac_eq = lambda alpha: self.ones
        # inequality constraint
        inequality_constraint = LinearConstraint(self.Dy, np.zeros(N), self.C * self.ones)
        
        # fun_ineq_1 = lambda alpha: - self.Dy @ self.alpha
        # jac_ineq_1 = lambda alpha: - self.Dy  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        # fun_ineq_2 = lambda alpha: - self.Dy @ self.alpha + self.C * self.ones
        # jac_ineq_2 = lambda alpha: - self.Dy  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        # # fun_ineq = lambda alpha: np.concatenate((fun_ineq_1(alpha), fun_ineq_2(alpha)))
        
        # constraints = ( [{'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
        #                 {'type': 'ineq', 'fun': fun_ineq_1, 'jac': jac_ineq_1},
        #                 {'type': 'ineq', 'fun': fun_ineq_2, 'jac': jac_ineq_2}]
        #                 )
        constraints = ( [{'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
                        inequality_constraint]
                        )

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        # list of indices of support vectors in dataset, None if not a support vector
        self.indices_support = np.array([ i if (self.epsilon < self.alpha[i]*self.y[i]) and (self.alpha[i]*self.y[i] < (C - self.epsilon) ) else None for i in range(N) ])
        self.indices_support = self.indices_support[self.indices_support != None].astype(int)
        # support vectors (data points on margin)
        self.support = self.X[self.indices_support]
        # alphas on support vectors
        self.alpha_support = self.alpha[self.indices_support]
        # compute b by averaging over support vectors
        b = self.y - self.gram @ self.alpha
        b_sv = b[self.indices_support]
        self.b = np.mean(b_sv)
        # '''------------------------RKHS norm of the function f ------------------------------'''
        self.norm_f = 1/2 * self.alpha @ self.gram @ self.alpha
        
        return self


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support) @ self.alpha_support + self.b
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1

def plotHyperSurface(ax, xRange, model, intercept, label, color='grey', linestyle='-', alpha=1.):
    xx = np.linspace(-1, 1, 100)
    
    if model.type=='linear':
        xRange = np.array(xRange)
        yy = -(model.w[0] / model.w[1]) * xRange  - intercept/model.w[1]
        ax.plot(xRange, yy, color=color, label=label, linestyle=linestyle)
    else:
        xRange = np.linspace(xRange[0], xRange[1], 100)
        X0, X1 = np.meshgrid(xRange, xRange)
        xy = np.vstack([X0.ravel(), X1.ravel()]).T
        Y30 = model.separating_function(xy).reshape(X0.shape) + intercept
        Y30 = Y30.astype(float)
        ax.contour(X0, X1, Y30, colors=color, levels=[0.], alpha=alpha, linestyles=[linestyle], label=label)


def plotClassification(X, y, model=None, label='',  separatorLabel='Separator', 
            ax=None, bound=[[-1., 1.], [-1., 1.]]):
    """ Plot the SVM separation, and margin """
    colors = ['blue','red']
    labels = [1,-1]
    cmap = pltcolors.ListedColormap(colors)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 7))
    for k, label in enumerate(labels):
        im = ax.scatter(X[y==label,0], X[y==label,1],  alpha=0.5,label='class '+str(label))

    if model is not None:
        # Plot the separating function
        plotHyperSurface(ax, bound[0], model, model.b, label=separatorLabel)
        if model.support is not None:
            ax.scatter(model.support[:,0], model.support[:,1], label='Support', s=80, facecolors='none', edgecolors='r', color='r')
            print("Number of support vectors = %d" % (len(model.support)))
        
        # Plot the margins
        intercept_neg = -0.0 # -1.0 - model.b #- 1.0 ### compute the intercept for the negative margin
        intercept_pos = +0.0 # -1.0 + model.b # + 1.0 ### compute the intercept for the positive margin
        xx = np.array(bound[0])
        plotHyperSurface(ax, xx, model, intercept_neg , label='Margin -', linestyle='-.', alpha=0.8)
        plotHyperSurface(ax, xx, model, intercept_pos , label='Margin +', linestyle='--', alpha=0.8)
            
        # Plot points on the wrong side of the margin
        # wrong_side_points = # find wrong points
        # ax.scatter(wrong_side_points[:,0], wrong_side_points[:,1], label='Beyond the margin', s=80, facecolors='none', 
        #        edgecolors='grey', color='grey')  
        
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])
    
C=1.
kernel = RBF().kernel
model = KernelSVC(C=C, kernel=kernel, epsilon=1e-14)
train_dataset = datasets['dataset_1']['train']
model.fit(train_dataset['x'], train_dataset['y'])
plotClassification(train_dataset['x'], train_dataset['y'], model, label='Training')
plt.show()