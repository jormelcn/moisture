import numpy as np

def exponentialMoisture(x, p):   
  return np.hstack((np.ones((x.shape[0],1), dtype=float), np.exp(p[0]*x)))
