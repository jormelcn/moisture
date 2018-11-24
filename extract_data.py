import pandas as pd
import numpy as np

# load dataframe
df = pd.read_csv("soilmoisture_dataset.csv", index_col=0)

# get hyperspectral bands:
bands = []
for column in df.columns:
  try:
    bands.append(int(column))
  except:
    pass

bands = np.array(bands)
moisture = df["soil_moisture"].values
reflect = df[[str(b) for b in  bands]].values
sort = np.argsort(moisture)

np.save('./data/bands.npy', bands)
np.save('./data/reflect.npy', reflect[sort,:])
np.save('./data/moisture.npy', moisture[sort].reshape((moisture.shape[0], 1)))