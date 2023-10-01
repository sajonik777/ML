# ML

# Find patterns that help us predict 

# Indicate strong indicators

# Forest tree - it is based on the concept of ensemble learning, which is a process of combining multiple algorithms
# to solve a particular problem.

# Principal Component Analysis (PCA) - is aiming to convert a set of observations of possibly correlated variables 
# into a set of values of linearly uncorrelated variables called principal components.
# PCA is extracting important variables (in form of components) from a large set of variables 
# available in a data set.
# It extracts low dimensional set of features from a high dimensional data set with a motive to capture 
# as much information as possible. 

 a simple step-by-step explanation of how PCA works:

Standardization: The aim of this step is to standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis.

Covariance Matrix computation: The aim of this step is to understand how the variables of the input data set are varying from the mean with respect to each other, or in other words, to see if there is any relationship between them. Because sometimes, variables are highly correlated in such a way that they contain redundant information. So, in order to identify these correlations, we compute the covariance matrix.

Compute the Eigenvalues and Eigenvectors: Eigenvectors and eigenvalues are the linear algebra concepts that we need to compute from the covariance matrix in order to determine the principal components of the data. Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components. So, the idea is 10-dimensional data gives you 10 principal components, but PCA tries to put maximum possible information in the first component, then maximum remaining information in the second and so on, until having something like shown in the scree plot below.

Scree plot: The scree plot helps you to determine the optimal number of components. The eigenvalue of each component in the initial solution is plotted. Generally, you want to extract the components on the steep slope. The components on the shallow slope contribute little to the solution.

Compute the Principal Components: Once we have the eigenvalues and eigenvectors from the covariance matrix, we can form the principal components using this information.