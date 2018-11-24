import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


def ridge(X, Y, p, model, alpha = 0):
  scaler = StandardScaler()
  scaler.fit(X)
  X_std = scaler.transform(X)
  Pred = model(X_std,p)
  regresor = linear_model.Ridge(alpha = alpha, fit_intercept=False)
  regresor.fit(Pred, Y)
  return (regresor.coef_.T, scaler.scale_, scaler.mean_)

def eval(X, p, model, R, scale, mean):
  X_std = (X - mean)/ scale
  return model(X_std, p).dot(R)

def R2(Y, Y_stm):
  return 1 - np.var(Y - Y_stm, axis=0)/np.var(Y, axis=0)


