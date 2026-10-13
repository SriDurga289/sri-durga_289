import pandas as pd
import numpy as np

data = pd.DataFrame({'A': [10, 20, 30, 40, 50], 'B': [5, 15, 25, 35, 45]})
log_data = np.log1p(data)
print(log_data)
