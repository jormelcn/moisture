import numpy as np

def exponentialMoisture(moisture, p):
  return p[1]*np.exp(p[0]*moisture)
