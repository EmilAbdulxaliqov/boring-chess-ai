import numpy as np
import torch

# print(len(np.zeros((8, 8, 5))), np.zeros((8, 8, 5), dtype=np.float32))
# print(np.zeros((5,8,8), np.uint8))


print(torch.cuda.is_available())