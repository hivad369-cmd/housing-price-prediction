from load_data import load_data

df = load_data()

print("=" * 50)
print("Shape")
print(df.shape)

print("\n" + "=" * 50)
print("Columns")
print(df.columns)

print("\n" + "=" * 50)
print("Information")
print(df.info())

print("\n" + "=" * 50)
print("Missing Values")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Statistics")
print(df.describe())