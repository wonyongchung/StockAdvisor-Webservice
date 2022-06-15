import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv(os.path.join(os.getcwd(), "webapp", "media", "377300", "377300.csv"), encoding='cp949')

ma30 = data['종가'].rolling(window=30).mean()
ma100 = data['종가'].rolling(window=100).mean()

data.insert(len(data.columns), "MA30", ma30)
data.insert(len(data.columns), "MA100", ma100)

plt.plot(data.index, data['종가'], label="종가")
plt.plot(data.index, data['MA30'], label="MA30")
plt.plot(data.index, data['MA100'], label="MA100")
plt.legend()
plt.show()
