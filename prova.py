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



bool_tensor = torch.tensor([True, False, True])  # dtype=torch.bool

# Option 1: Cast to int32
int_tensor = bool_tensor.int()  # dtype=torch.int32

# Option 2: Cast to int64 (common default for indexing etc.)
long_tensor = bool_tensor.long()  # dtype=torch.int64

# Option 3: Generic casting
int_tensor_alt = bool_tensor.to(torch.int)


print(numpy.arange(5))