import numpy as np

values = [13576.4912109375, 13270.2783203125, 13382.40625]
mean = np.mean(values)
std = np.std(values, ddof=1)

print(mean, std)