import numpy as np
import matplotlib.pyplot as plt
from fABBA import fABBA

ts = [np.sin(0.05*i) for i in range(1000)]  # original time series
fabba = fABBA(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)

string = fabba.fit_transform(ts)            # string representation of the time series
print(string)                               # prints aBbCbCbCbCbCbCbCA

inverse_ts = fabba.inverse_transform(string, ts[0]) # numerical time series reconstruction

plt.plot(ts, label='time series')
plt.plot(inverse_ts, label='reconstruction')
plt.legend()
plt.grid(True, axis='y')
plt.show()
plt.savefig("reconstruction.png")
