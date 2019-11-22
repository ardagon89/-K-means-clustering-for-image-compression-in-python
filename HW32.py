#!/usr/bin/env python
# coding: utf-8

# In[1]:


def PrintCores():
        import psutil;                  
        print( "{0:17s}{1:} CPUs PHYSICAL".format(
              "psutil:",
               psutil.cpu_count( logical = False ) ) )
        pass;                           print( "{0:17s}{1:} CPUs LOGICAL".format(
              "psutil:",
               psutil.cpu_count( logical = True  ) ) )

def LoadDS():
    from sklearn.datasets import fetch_openml
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255
    return X, y
    
def Split(X):
    return X[:60000], X[60000:]

def SavePkl(file, name):
    import pickle
    f=open(name+'.pkl', 'wb')
    pickle.dump(file, f)
    f.close()
    
def LoadPkl():
    import pickle
    # open a file, where you stored the pickled data
    try:
        file = open('X_train.pkl', 'rb')
        # dump information to that file
        X_train = pickle.load(file)
        # close the file
        file.close()
        
        file = open('X_test.pkl', 'rb')
        X_test = pickle.load(file)
        file.close()
        
        file = open('y_train.pkl', 'rb')
        y_train = pickle.load(file)
        file.close()
        
        file = open('y_test.pkl', 'rb')
        y_test = pickle.load(file)
        file.close()
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError:
        #print('Pickle file not found! Loading entire data..')
        X, y = LoadDS()
        X_train, X_test = Split(X)
        y_train, y_test = Split(y)
        #SavePkl(X_train, 'X_train')
        #SavePkl(X_test, 'X_test')
        #SavePkl(y_train, 'y_train')
        #SavePkl(y_test, 'y_test')
        return X_train, X_test, y_train, y_test
    
    except Exception:
        print('Other Errors! Please check. Exiting..')
        return None, None, None, None
    
def FindError(y_test, y_pred):
    return sum(y_test!=y_pred)*100/y_test.size

def GBC(X_train, X_test, y_train, y_test, _n_estimators, _max_features, _max_depth):
    from sklearn.ensemble import GradientBoostingClassifier
    
    gbclassifier = GradientBoostingClassifier(n_estimators=_n_estimators, max_features=_max_features, max_depth=_max_depth)

    gbclassifier.fit(X_train, y_train)
    y_pred = gbclassifier.predict(X_test)
    
    print('GBC with', _n_estimators, 'estimators,', _max_features, 'features,', _max_depth, 'max_depth :', FindError(y_test,y_pred))
    
def MLPC(X_train, X_test, y_train, y_test, _activation, _solver):
    from sklearn.neural_network import MLPClassifier
    
    mlpclassifier = MLPClassifier(hidden_layer_sizes=(100,), activation=_activation, alpha=0.0001, solver=_solver, tol=0.001, verbose=0, max_iter=1000, early_stopping=True, momentum=0.9)

    mlpclassifier.fit(X_train, y_train)
    y_pred = mlpclassifier.predict(X_test)
    
    print('MLPC with activation', _activation, 'and solver', _solver, ':', FindError(y_test,y_pred))
    
def LinearSVC(X_train, X_test, y_train, y_test, _penalty, _C):
    from sklearn.svm import LinearSVC
    
    svclassifier = LinearSVC(penalty=_penalty, dual=False, tol=0.001, C=_C, verbose=0, max_iter=1000)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    
    print('LinearSVC with '+_penalty+' and C '+ str(_C) +' : ', FindError(y_test,y_pred))

def SVCPolyKernel(X_train, X_test, y_train, y_test, _degree):
    from sklearn.svm import SVC
    
    svclassifier = SVC(kernel='poly', degree=_degree, gamma='auto', cache_size=1000, shrinking=True, verbose=False, max_iter=-1, tol=0.001, C=1.0)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    
    print('SVC with polynomial kernel of degree', _degree, ':', FindError(y_test,y_pred))
    
def SVCGaussKernel(X_train, X_test, y_train, y_test, _degree):
    from sklearn.svm import SVC
    
    svclassifier = SVC(kernel='rbf', degree=_degree, gamma='auto', cache_size=1000, shrinking=True, verbose=False, max_iter=-1, tol=0.001, C=1.0)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    
    print('SVC with gaussian kernel of degree', _degree, ':', FindError(y_test,y_pred))
    
def SVCwithKernel(X_train, X_test, y_train, y_test, _kernel, _degree, _C):
    from sklearn.svm import SVC
    
    svclassifier = SVC(kernel=_kernel, degree=_degree, gamma='auto', cache_size=1000, shrinking=True, verbose=False, max_iter=-1, tol=0.001, C=_C)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    
    print('SVC with', _kernel, 'kernel of degree', _degree,'and C', _C, ':', FindError(y_test,y_pred))
    
if __name__ == '__main__':
    
    import sys
    import time

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        
    if len(sys.argv) < 4:
        print("Usage:python HW32.py <classifier> <param1> <param2> <param3>")
        try:
            X_train
        except NameError:
            #print("Variable not defined. Initializing..")
            X_train, X_test, y_train, y_test = LoadPkl()
        
        run_entire = False
        if run_entire:
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 2, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 2, 4)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 2, 8)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 2, 16)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 2, 32)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 2, 64)
            print("Time taken:", time.time()-t)  

            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 20, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 40, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 80, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 160, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10, 320, 2)
            print("Time taken:", time.time()-t)

            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 20, 2, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 40, 2, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 80, 2, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 160, 2, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 320, 2, 2)
            print("Time taken:", time.time()-t)  
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 640, 2, 2)
            print("Time taken:", time.time()-t) 
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 1200, 2, 2)
            print("Time taken:", time.time()-t)
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 3600, 2, 2)
            print("Time taken:", time.time()-t)  
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, 10000, 2, 2)
            print("Time taken:", time.time()-t)  

            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'identity', 'sgd')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'identity', 'lbfgs')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'identity', 'adam')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'logistic', 'sgd')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'logistic', 'lbfgs')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'logistic', 'adam')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'tanh', 'sgd')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'tanh', 'lbfgs')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'tanh', 'adam')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'relu', 'sgd')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'relu', 'lbfgs')
            print("Time taken:", time.time()-t)
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, 'relu', 'adam')
            print("Time taken:", time.time()-t)

            t=time.time()
            LinearSVC(X_train, X_test, y_train, y_test, 'l1', 1)
            print("Time taken:", time.time()-t)
            t=time.time()
            LinearSVC(X_train, X_test, y_train, y_test, 'l2', 1)
            print("Time taken:", time.time()-t)
            t=time.time()
            LinearSVC(X_train, X_test, y_train, y_test, 'l1', 0.1)
            print("Time taken:", time.time()-t)
            t=time.time()
            LinearSVC(X_train, X_test, y_train, y_test, 'l2', 0.1)
            print("Time taken:", time.time()-t)
            t=time.time()
            LinearSVC(X_train, X_test, y_train, y_test, 'l1', 10)
            print("Time taken:", time.time()-t)
            t=time.time()
            LinearSVC(X_train, X_test, y_train, y_test, 'l2', 10)
            print("Time taken:", time.time()-t)

            t=time.time()
            SVCwithKernel(X_train, X_test, y_train, y_test, 'poly', 2, 1)
            print("Time taken:", time.time()-t)
            t=time.time()
            SVCwithKernel(X_train, X_test, y_train, y_test, 'poly', 3, 1)
            print("Time taken:", time.time()-t)
            t=time.time()
            SVCwithKernel(X_train, X_test, y_train, y_test, 'poly', 4, 1)
            print("Time taken:", time.time()-t)

            t=time.time()
            SVCwithKernel(X_train, X_test, y_train, y_test, 'rbf', 0, 1)
            print("Time taken:", time.time()-t)

            t=time.time()
            SVCwithKernel(X_train, X_test, y_train, y_test, 'rbf', 0, 0.1)
            print("Time taken:", time.time()-t)

            t=time.time()
            SVCwithKernel(X_train, X_test, y_train, y_test, 'sigmoid', 0, 1)
            print("Time taken:", time.time()-t)

    else:
        
        try:
            X_train
        except NameError:
            #print("Variable not defined. Initializing..")
            X_train, X_test, y_train, y_test = LoadPkl()
        
        classifier = sys.argv[1]
        
        if classifier == "-GBC":
            n_estimators = int(sys.argv[2]) 
            max_features = int(sys.argv[3]) 
            max_depth = int(sys.argv[4])
            t=time.time()
            GBC(X_train, X_test, y_train, y_test, n_estimators, max_features, max_depth)
            print("Time taken:", time.time()-t)
        elif classifier == "-MLPC":
            activation = sys.argv[2]
            solver = sys.argv[3]
            t=time.time()
            MLPC(X_train, X_test, y_train, y_test, activation, solver)
            print("Time taken:", time.time()-t)
            
        elif classifier == "-SVM":
            kernel = sys.argv[2]
            degree = int(sys.argv[3]) 
            C = float(sys.argv[4]) 
            penalty = sys.argv[5]
            
            if kernel == 'poly' and degree == 1:
                t=time.time()
                LinearSVC(X_train, X_test, y_train, y_test, penalty, C)
                print("Time taken:", time.time()-t)
                
            else:
                t=time.time()
                SVCwithKernel(X_train, X_test, y_train, y_test, kernel, degree, C)
                print("Time taken:", time.time()-t)
