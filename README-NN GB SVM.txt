SVM
Execute: python HW2.py [classifier] [kernel] [degree] [C] [penalty]
example: python HW32.py -SVM poly 1 1 l2

classifier: string [-SVM | -MLPC | -GBC] (should be -SVM for SVM and the below hyper-parameters)
kernel: string [poly | rbf | sigmoid]
degree: integer degree of the polynomial kernel (for rbf and sigmoid degree should be 0)
C: float penalty parameter C of the error term
penalty: string [l1 | l2] regularization penalty (for polynomial kernel of degree 1)

output:
LinearSVC with l2 and C 1.0 :  8.18
Time taken: 32.566481828689575
----------------------------------------------------------------------------------------------
Neural Network
Execute: python HW2.py [classifier] [activation] [solver]
example: python HW32.py -MLPC identity adam

classifier: string [-SVM | -MLPC | -GBC] (should be -MLPC for Neural Network and the below hyper-parameters)
activation: string [identity | logistic | tanh | relu]
solver: string [sgd | lbfgs | adam]

output:
MLPC with activation identity and solver adam : 7.41
Time taken: 26.653262853622437
----------------------------------------------------------------------------------------------
Gradient Boosting Classifier
Execute: python HW2.py [classifier] [n_estimators] [max_features] [max_depth]
example: python HW32.py -GBC 10 2 64

classifier: string [-SVM | -MLPC | -GBC] (should be -GBC for Gradient Boosting Classifier and the below hyper-parameters)
n_estimators: integer number of the estimators of the classifier
max_features: integer maximum number of features to be considered at each node
max_depth: integer maximum depth of the estimator

output:
GBC with 10 estimators, 2 features, 64 max_depth : 8.02
Time taken: 160.77276229858398