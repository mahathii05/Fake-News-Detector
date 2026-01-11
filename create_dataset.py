import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake['label'] = 0   # Fake
true['label'] = 1   # Real

fake = fake[['text', 'label']]
true = true[['text', 'label']]

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42)

data.to_csv("dataset.csv", index=False)

print("✅ Clean dataset.csv created")
print(data['label'].value_counts())
