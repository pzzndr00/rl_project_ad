import numpy
import random
import matplotlib.pyplot as plt
# print(list(range(5-1,-1, -1)))

array = numpy.array([0.1, 0.3, 0.4, 0.2])
recurrency = numpy.zeros_like(array)
print(random.choices(range(array.shape[0]), array))

for _ in range(100000):
    i = random.choices(range(array.shape[0]), array)
    recurrency[i] += 1


plt.stairs(recurrency)
plt.show()
