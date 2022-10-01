import numpy as np
# import scipy.io as scio
from tqdm import tqdm

# small_dataset = np.zeros((100, 10001, 5))
# for i in tqdm(range(100)):
#     file_name = f'Data\\Data{i+1}.mat'
#     data = scio.loadmat(file_name)['Data']
#     small_dataset[i] = data
#
#
# np.savez_compressed('small_dataset.npz', small_dataset)
#
# print("end test")

small_dataset = np.load('small_dataset.npz')['arr_0']
first_data = small_dataset[0]
print(first_data)
print(first_data.shape)
print(small_dataset.shape)
print("ff")