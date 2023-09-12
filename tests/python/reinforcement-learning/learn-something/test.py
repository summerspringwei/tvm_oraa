import numpy as np

# a = np.array([[3, 4, 5],
#        [6, 7, 8],
#        [0, 1, 2],
#        [1, 1, 1],
#        [9, 9, 9],
#        [8, 8, 8]])
# # print(np.random.shuffle(a))
# # print(a[:, 0:-1])
# # print(a[:, -1])


# print(a[0::2])
# print(a[1::2])

# import torch

# x = torch.tensor(np.array([1.0, 2.0]), requires_grad=True)
# b = torch.exp(x)
# print(b.grad_fn)
# print(0.73*0.73)
# print(-0.53*0.37)

# # import pandas library
# import pandas as pd
 
# # creating the dataframe
# df = pd.DataFrame({'PassengerId': [892, 893,
#                                    894, 895,
#                                    896, 897,
#                                    898, 899],
#                    'PassengerClass': [1, 1, 2,
#                                       1, 3, 3,
#                                       2, 2],
#                    'PassengerName': ['John',
#                                      'Prity',
#                                      'Harry',
#                                      'Smith',
#                                      'James',
#                                      'Amora',
#                                      'Kiara', 'Joseph'],
#                    'Age': [32, 54, 71, 21,
#                            37, 9, 11, 54]})
 
# print("The DataFrame :")
# print(type(df.columns))
# print(df.columns)
# print(df.loc[2:, :])
 
# # multiple ways of getting column names as list
# print("\nThe column headers :")
# print("Column headers from list(df.columns.values):",
#       set(list(df.columns.values)))
# print("Column headers from list(df):", list(df))
# print("Column headers from list(df.columns):",
#       list(df.columns))

ids = np.array([0,0,0,1,1,1])
weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
print(np.bincount(ids, weight))