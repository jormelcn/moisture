import numpy as np
import matplotlib.pyplot as plt
from models import exponentialMoisture as model


bands = np.load('./data/bands.npy')
reflect = np.load('./data/reflect.npy')
moisture = np.load('./data/moisture.npy')

regression = np.load('./regression/exponential_moisture.npy')
success = np.load('./regression/exponential_moisture_success.npy')
errors = np.load('./regression/exponential_moisture_errors.npy')

sort_err = np.argsort(errors)
sort_moist = np.argsort(moisture)

def plotModel(index, title):
  plt.figure()
  plt.ylim([0, 0.4])
  plt.xlabel('Moisture (%)')
  plt.ylabel('Reflectance')
  plt.title(title)
  _model = model(moisture[sort_moist], regression[index,:])
  plt.plot(moisture[sort_moist] , _model)
  plt.plot(moisture[sort_moist] , reflect[sort_moist , index], 'k+')

plotModel(sort_err[0], 'Best Model %s nm R = %.3f' % (bands[sort_err[0]], 1 - errors[sort_err[0]]))
plt.savefig('./graphics/exponential_moisture/best_model.png')
plotModel(sort_err[-1], 'Poor Model %s nm R = %.3f' % (bands[sort_err[-1]], 1 - errors[sort_err[-1]]))
plt.savefig('./graphics/exponential_moisture/poor_model.png')

plt.figure()
plt.xlabel('Longitud de Onda (nm)')
plt.ylabel('R2')
plt.title('Rendimiento')
plt.plot(bands, 1 - errors)
plt.savefig('./graphics/exponential_moisture/performance.png')

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title('Comportamiento Exponencial') 
ax1.set_ylabel('Exponente')
ax1.plot(bands, np.abs(regression[:,0]))

ax2.set_xlabel('Longitud de Onda (nm)')
ax2.set_ylabel('Coeficiente')
ax2.plot(bands, regression[:,1])
fig.savefig('./graphics/exponential_moisture/behavior.png')

plt.show()
