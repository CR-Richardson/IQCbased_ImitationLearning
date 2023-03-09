# IQCbased_ImitationLearning
The original code from the inverted pendulum example was generalised to allow for experimentation with an arbitrary number of hidden layers with arbitrary width. To this end, the scripts `NN_policy.py` and `solve_sdp.m` were modified. Some additional plotting scripts were also uploaded for comparing the ROA when using various neural network controllers. 

### Original Authors:
* He Yin (he_yin at berkeley.edu)
* Peter Seiler (pseiler at umich.edu)
* Ming Jin (jinming at vt.edu)
* Murat Arcak (arcak at berkeley.edu)

### Author of modified code:
* Carl R Richardson (cr2g16 at soton.ac.uk)

## Getting Started
The code is written in Python3 and MATLAB.

### Prerequisites
There are several packages required:
* [MOSEK](https://www.mosek.com/): Commercial semidefinite programming solver
* [CVX](http://cvxr.com/cvx/): MATLAB Software for Convex Programming
* [Tensorflow](https://www.tensorflow.org/): Open source machine learning platform

To plot the computed ROA, two more packages are required:
* [SOSOPT](https://dept.aem.umn.edu/~AerospaceControl/): General SOS optimization utility
* [Multipoly](https://dept.aem.umn.edu/~AerospaceControl/): Package used to represent multivariate polynomials

### Way of Using the Code
* To start the safe imitation learing process run `NN_policy.py`. The computation results are stored in the folder **data**. 
* To visualize the results for the inverted pendulum example, run `result_analysis.m`.
* To compare the results using two different neural network controllers, run `result_analysis_ROA.m`.
