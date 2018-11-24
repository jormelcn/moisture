import numpy as np
import matplotlib.pyplot as plt
from models import exponentialMoisture as model
import generic_regresor as gr

bands = np.load('./data/bands.npy')
reflect = np.load('./data/reflect.npy')
moisture = np.load('./data/moisture.npy')

Rs = np.load('./regression/exponential_moisture_r.npy')
Ps = np.load('./regression/exponential_moisture_p.npy')
means = np.load('./regression/exponential_moisture_means.npy')
scales = np.load('./regression/exponential_moisture_scales.npy')
success = np.load('./regression/exponential_moisture_success.npy')
R2s = np.load('./regression/exponential_moisture_R2s.npy')

_R2s = np.mean(R2s, axis=1)
s = np.argsort(_R2s, axis=0)


def plotModel(i, title):
  plt.figure()
  plt.ylim([0, 0.4])
  plt.xlabel('Moisture (%)')
  plt.ylabel('Reflectance')
  plt.title(title)
  p = Ps[i,:]
  r = Rs[i,:,:]
  m = means[i,:]
  s = scales[i,:]
  _model = gr.eval(moisture, p, model, r, s, m)
  plt.plot(moisture , _model)
  plt.plot(moisture , reflect[:, i], 'k+')

plotModel(s[-1], 'Best Model %s nm R = %.3f' % (bands[s[-1]], R2s[s[-1]]))
plt.savefig('./graphics/exponential_moisture/best_model.png')
plotModel(s[0], 'Poor Model %s nm R = %.3f' % (bands[s[0]], R2s[s[0]]))
plt.savefig('./graphics/exponential_moisture/poor_model.png')

print('Best band: ',s[-1], ' ', bands[s[-1]])

plt.figure()
plt.xlabel('Longitud de Onda (nm)')
plt.ylabel('R2')
plt.title('Rendimiento')
plt.plot(bands, _R2s)
plt.savefig('./graphics/exponential_moisture/performance.png')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.set_title('Comportamiento') 
ax1.set_ylabel('Exponente')
ax1.plot(bands, np.abs(Ps[:,0]))

ax2.set_ylabel('Coeficiente')
ax2.plot(bands, Rs[:,1,0])

ax3.set_xlabel('Longitud de Onda (nm)')
ax3.set_ylabel('constante')
ax3.plot(bands, Rs[:,0,0])

fig.savefig('./graphics/exponential_moisture/behavior.png')

plt.show()
