import pickle as pkl
from scipy import optimize
from scipy.optimize import LinearConstraint
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from utils import plotClassification, plotRegression, plot_multiple_images, generateRings, scatter_label_points, loadMNIST
# %matplotlib inline    # notebook only

from ClassKernels import RBF, Linear, KernelSVC, plotClassification, plotHyperSurface
# pour comparaison
from sklearn.svm import SVC

file = open('code/datasets/classification_datasets', 'rb')
datasets = pkl.load(file)

#----------------------------- ALGO MAISON --------------------------------    
C=1.
kernel = Linear().kernel
model = KernelSVC(C=C, kernel=kernel, epsilon=1e-13)
train_dataset = datasets['dataset_1']['train']
model.fit(train_dataset['x'], train_dataset['y'])
print(f"Résultats algo maison :-)")
print(f"Number of support vectors = {len(model.support)}")
print(f"Support vectors = {model.support}")
print(f"Alphas = {model.alpha[model.indices_support]}")
print(f"b = {model.b}")

# ---------------------------- ALGO SCIKIT --------------------------------
# comparaison avec scikit
clf = SVC(C=C, kernel='linear')
clf.fit(train_dataset['x'], train_dataset['y'])
print(f"Résultats algo scikit :")
print(f"Number of support vectors = {len(clf.support_vectors_)}")
print(f"Support vectors = {clf.support_vectors_}")
print(f"Dual coefficients = {clf.dual_coef_}")
print(f"b = {clf.intercept_}")


#-------------- ANALYSE -----------------------------------------------------
print(f"\n")
print(f"Valeur des alpha_i * y_i du modèle maison (à comparer à C={C}), aux indices de vecteurs support indiqués par scikit :")
print(f"alpha_i * y_i = {model.alpha[clf.support_] * model.y[clf.support_]}")



# plotClassification(train_dataset['x'], train_dataset['y'], model, label='Training')
# plt.show()