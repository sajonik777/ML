from sklearn.preprocessing import StandardScaler
import numpy as np

ss = StandardScaler()

numeric_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Assuming that 'train_data' and 'test_data' are predefined
train_data[numeric_columns] = ss.fit_transform(train_data[numeric_columns])
test_data[numeric_columns] = ss.transform(test_data[numeric_columns])

train_data = np.array(train_data)

print("Success")