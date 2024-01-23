import pandas as pd


a = [-2, -1, 0]
df = pd.DataFrame(a)

def func(x):
   return x**2

# print(list(map(func, l)))
print('-----')
print(a)

print('-----')
print(df)

print('-----')
df_mapped = df.map(func)
print(df_mapped)
