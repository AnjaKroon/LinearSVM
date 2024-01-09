# Inputs
linear_svm.mat <br/>
X_test            900x2             14400  double <br/>
X_train           100x2              1600  double <br/>
labels_test       900x1              7200  double <br/>
labels_train      100x1               800  double<br/>

# Outputs
See terminal output for full description of the various evaluation metrics.

# Description
Given two classes of data points on a cartesian plane, linear support vector machines (linear SVMs) attempt
to find the optimal hyperplane separating the two classes of data. Optimal is defined by the separating 
hyperplane that maximizes the distance between itself and the closest point in each class, respectively. 
This problem setting can be framed as a convex optimization problem.
If given data are linearly separable, the problem can be solved manually via the newton step with log barrier 
approach (NT Log Barrier) or with the CVX solver. However, real-world 
data are often not linearly separable. Thus, a relaxation of the problem setting to enable violations of the 
constraints is presented and solved via the gradient descent (GD) method. 
The performance of these three solvers on their solution to the convex optimization problem presented by linear 
SVMs is evaluated. Evaluation is based on CPU time, number of iterations, the objective 
function value, and classification accuracy on the test set.
