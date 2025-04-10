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




import torch

# Example tensor (2D)
tensor = torch.tensor([
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18]
])

# Index array: one column index for each row
indices = torch.tensor([0, 2, 1]) # Means: [row0, col0], [row1, col2], [row2, col1]

# Extract values
result = tensor[torch.arange(tensor.size(0)), indices]
print(result)  # tensor([10, 15, 17])

t1 = torch.tensor([1,2,3,4])
t2 = torch.tensor([1,2,3,4])

print(t1/t2)
