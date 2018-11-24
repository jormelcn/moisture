
import numpy as np
import scipy.optimize as opt
from models import exponentialMoisture as model
import generic_regresor as gr

bands = np.load('./data/bands.npy')
reflect = np.load('./data/reflect.npy')
moisture = np.load('./data/moisture.npy')

X = moisture
Y = reflect[:,0:1]
p = np.array([-1.3], dtype=float)
F = model(X, p)

def objetive(p):
   r, s, m = gr.ridge(X, Y, p, model)
   return - gr.R2(Y, gr.eval(X, p, model, r, s, m))

def expNeg(p):
   return -p[0]

constraints=[
  {'type':'ineq', 'fun':expNeg}
]

Rs = np.zeros((len(bands), F.shape[1], Y.shape[1]))
Ps = np.zeros((len(bands), p.shape[0]))
means = np.zeros((len(bands), X.shape[1]))
scales = np.zeros((len(bands), X.shape[1]))
R2s = np.zeros((len(bands), Y.shape[1]))
success = np.zeros(len(bands), dtype=bool)

for i in range(len(bands)):
  Y = reflect[:,i:i+1]
  result = opt.minimize(objetive, [-1], constraints=constraints)
  if result['success'] :
    success[i] = True
    Ps[i,:] = np.array(result['x'])
    Rs[i,:,:], scales[i,:], means[i,:] = gr.ridge(X, Y, Ps[i,:], model)
    R2s[i,:] = gr.R2(Y, gr.eval(X, Ps[i,:], model, Rs[i,:,:], scales[i,:], means[i,:]))

print('%d success Regressions' % np.sum(success))  

np.save('./regression/exponential_moisture_p.npy', Ps)
np.save('./regression/exponential_moisture_r.npy', Rs)
np.save('./regression/exponential_moisture_means.npy', means)
np.save('./regression/exponential_moisture_scales.npy', scales)
np.save('./regression/exponential_moisture_success.npy', success)
np.save('./regression/exponential_moisture_R2s.npy', R2s)

# print(np.sum(success), ' success regressions')