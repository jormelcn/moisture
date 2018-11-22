
import numpy as np
import scipy.optimize as opt
from models import exponentialMoisture as model

bands = np.load('./data/bands.npy')
reflect = np.load('./data/reflect.npy')
moisture = np.load('./data/moisture.npy')

def objetive(p):
  dif = model(moisture, p) - reflect[:,objetiveBand]
  return sum(dif**2)

def expNeg(p):
  return -p[0]

def facPos(p):
  return p[1]

constraints=[
  {'type':'ineq', 'fun':expNeg},
  {'type':'ineq', 'fun':facPos},
]

results = np.zeros((len(bands), 2))
errors = np.zeros(len(bands))
success = np.zeros(len(bands), dtype=bool)
for i in range(len(bands)):
  objetiveBand = i
  result = opt.minimize(objetive, [-0.1,0.1], method='SLSQP', constraints=constraints)
  results[i,:] = result['x']
  success[i] = result['success']
  errors[i] = np.var((model(moisture, results[i,:]) - reflect[:,objetiveBand]))/np.var(reflect[:,objetiveBand])
  
np.save('./regression/exponential_moisture.npy', results)
np.save('./regression/exponential_moisture_success.npy', success)
np.save('./regression/exponential_moisture_errors.npy', errors)

print(np.sum(success), ' success regressions')