import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

# Check for NaN values and replace them with '0'
train_data.fillna(0, inplace=True)

# Exclude rows with 'Age' value less than '10'
train_data = train_data[train_data.Age >= 10]

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# Check for NaN values and replace them with '0'
test_data.fillna(0, inplace=True)

# Exclude rows with 'Age' value less than '10'
test_data = test_data[test_data.Age >= 10]

test_data.describe()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

print(round(model.score(X, y), 3))
#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
#output.to_csv('submission.csv', index=False)

#print("Your submission was successfully saved!")

# Extract feature importances
importances = model.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(8, 4))
plt.bar(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature importances from RandomForestClassifier')
plt.show()