import pandas as pd
data = pd.read_csv("iris.csv")
df = pd.DataFrame(data)

stats = df[['Age', 'Salary']].groupby('Salary').describe()

print(stats)