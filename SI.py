"""
Implementation of the paper 
B. K. Baghbaderani, A. Hasanebrahimi, A. Kalhor and R. Hosseini, 
"Adversarial Robustness Evaluation with Separation Index," 
2023 13th International Conference on Computer and Knowledge Engineering (ICCKE),

"""
import numpy as np
def mean(arr,y,X):
  indexes= (y==arr).flatten()
  return np.mean(X[indexes],axis=0)
def distance(arr,centers):
  shape_c=centers.shape

  return np.argmin(np.linalg.norm(centers-arr.reshape(1,arr.shape[0]),axis=1))
def CSI(X,y):
  shape_X=X.shape
  prod=np.prod(shape_X[1:])
  X=np.reshape(X,(shape_X[0],prod))
  y_prime=y.reshape(shape_X[0],1)
  label_uniques=np.sort(np.unique(y_prime))
  label_uniques=label_uniques.reshape(len(label_uniques),1)

  centers=np.apply_along_axis(mean,arr=label_uniques,y=y,X=X,axis=1)
  c_stars=np.apply_along_axis(distance,arr=X,centers=centers,axis=1)
  result = (y==c_stars.reshape(y.shape))
  return np.sum(result)/X.shape[0]
def kronecker_delta(arr,y,order,index,mode,y_test=None):
  if np.any(y_test):
    class_=arr[-1]
    result = (y[arr[:order]] == class_).all().astype(int)
  else :
    class_=y[arr[-1]]
    result = (y[arr[:order]] == class_).all().astype(int)
  if mode=="anti":
    result=1-result
    ordered=np.prod(result)
  if mode=="soft":
    ordered=np.sum(result)
  else:
    ordered=np.prod(result)

  return ordered
def calculate_norms(arr,X):
  return np.linalg.norm(X-arr.reshape(1,arr.shape[0]),axis=1)
def SI(X,y,order=1,mode=None,X_test=None,y_test=None):

  shape_X=X.shape
  prod=np.prod(shape_X[1:])
  X=np.reshape(X,(shape_X[0],prod))
  X_reshaped =X.reshape(X.shape[0], 1, X.shape[1])
  if np.any(X_test):
    shape_X_test=X_test.shape
    prod2=np.prod(shape_X_test[1:])
    X_test=np.reshape(X_test,(shape_X_test[0],prod2))
    X_test_reshaped =X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    norms=np.apply_along_axis(calculate_norms,axis=1,arr=X_test,X=X)#np.sqrt(np.einsum('ijk, ijk->ij', X-X_test_reshaped, X-X_test_reshaped))
    norms=np.argsort(norms, axis=1,kind='quick')
    norms=np.hstack((norms,y_test.reshape(shape_X_test[0],1)))
    seperation_indexes = np.apply_along_axis(kronecker_delta, axis=1, arr=norms, y=y, order=order, index=norms.shape[0],mode=mode,y_test=y_test)
    denominator = shape_X_test[0]
  else :
    norms=np.apply_along_axis(calculate_norms,axis=1,arr=X,X=X)#norms2=np.sqrt(np.einsum('ijk, ijk->ij', X-X_reshaped, X-X_reshaped))
    np.fill_diagonal(norms, np.inf)
    norms=np.argsort(norms, axis=1,kind='quick')
    seperation_indexes = np.apply_along_axis(kronecker_delta, axis=1, arr=norms, y=y, order=order, index=norms.shape[0],mode=mode)
    denominator = shape_X[0]
  seperation_indexes=np.sum(seperation_indexes)
  if mode=="soft":
    seperation_indexes=seperation_indexes/(denominator*order)
  else:
    seperation_indexes=seperation_indexes/denominator
  return seperation_indexes

