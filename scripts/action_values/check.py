import numpy as np

# Your original list
original_list = np.array([[0., 1., 1., 0.16, 0.2857143, -0.42857143],
                          [0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0.]])

# Remove the 2nd column
result_list = np.delete(original_list, 1, axis=1)

print(result_list)
