import torch

a = 4
b = 5
c = 3

# Random tensor of shape (a, b, c)
arr1 = torch.randn(a, b, c)

# Random integer indices tensor of shape (a, b) with values between 0 and c-1
arr2 = torch.randint(low=0, high=c, size=[a, b])

# Create a tensor of zeros to store the values
arr3 = torch.zeros([a, b])

# Use advanced indexing to extract values from arr1 based on arr2 indices
arr3 = arr1[torch.arange(a).unsqueeze(1), torch.arange(b), arr2]

print(arr3)

