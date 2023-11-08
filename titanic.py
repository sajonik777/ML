# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

# Feature Engineering
# Update the 'Embarked' column for passengers with ID "830" and "62"
train_data.loc[train_data['PassengerId'].isin([830, 62]), 'Embarked'] = 'S'

# Drop rows where 'Age' is NaN
train_data.dropna(subset=['Age'], inplace=True)

# Calculate the mean age for each combination of 'Sex' and 'Pclass'
mean_ages = train_data.groupby(['Sex', 'Pclass'])['Age'].mean()

# Replace the missing 'Age' values with the mean age of the corresponding 'Sex' and 'Pclass'
train_data.loc[train_data.Age.isnull(), 'Age'] = train_data[train_data.Age.isnull()].apply(lambda row: mean_ages[row['Sex'], row['Pclass']], axis=1)

# Add condition, if male traveled alone then survived=0
Alone = (train_data['SibSp'] + train_data['Parch'] == 0) & (train_data['Sex'] == 'male') & (train_data['Age'] > 17)
train_data.loc[Alone, 'Survived'] = 0
train_data.loc[Alone, 'Alone'] = train_data.loc[Alone, 'Survived']

# if female from 17 to 65 then survived=1
Women = (train_data['Sex'] == 'female') & ((train_data['Age'] >= 17) & (train_data['Age'] <= 65))
train_data.loc[Women, 'Survived'] = 1
train_data.loc[Women, 'Women'] = train_data.loc[Women, 'Survived']
train_data.loc[~Women, 'Women'] = 0

# if children (both male and female) from 5 to 17 then survived=1
Children = ((train_data['Age'] >= 3) & (train_data['Age'] <= 15))
train_data.loc[Children, 'Survived'] = 1
train_data['Children'] = Children.astype(int)

train_data.head()

train_data.shape

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# Feature Engineering
average_ages_test = test_data.groupby(['Sex', "Parch"])['Age'].mean()
test_data.loc[test_data.Age == 0, 'Age'] = test_data[test_data.Age == 0].apply(lambda row: average_ages_test[row['Sex'], row['Parch']], axis=1)

# if female from 17 to 65 then survived=1
Women = (test_data['Sex'] == 'female') & ((test_data['Age'] >= 17) & (test_data['Age'] <= 65))
test_data.loc[Women, 'Survived'] = 1
test_data.loc[Women, 'Women'] = test_data.loc[Women, 'Survived']
test_data.loc[~Women, 'Women'] = 0

# if children (both male and female) from 5 to 17 then survived=1
Children = ((test_data['Age'] >= 3) & (test_data['Age'] <= 15))
test_data.loc[Children, 'Survived'] = 1
test_data['Children'] = Children.astype(int)

# Drop rows where 'Age' is NaN
#test_data.dropna(subset=['Age'], inplace=True)

# Add condition, if male traveled alone then survived=0
Alone = (test_data['SibSp'] + test_data['Parch'] == 0) & (test_data['Sex'] == 'male')
test_data.loc[Alone, 'Alone'] = test_data.loc[Alone, 'Survived']

test_data.fillna(0, inplace=True)

test_data.describe()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", round((rate_women), 3))

# Calculate survival rate for children (age 1-17)
children = train_data.loc[(train_data.Age >= 1) & (train_data.Age <= 17)]["Survived"]
rate_children = sum(children)/len(children)

print("% of children who survived:", round((rate_children), 3))

# Summarize the survival rates
print("% of women and children who survived:", round((rate_women + rate_children) / 2, 3))

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

y = train_data["Survived"]

features = ["Women", "Children", "Sex"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Apply PCA
#pca = PCA(n_components=4)
#principalComponents = pca.fit_transform(X_train)
#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
model.fit(X_train, y)
predictions = model.predict(X_test)

print(round(model.score(X_train, y), 3))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we

train_X, val_X, train_y, val_y = train_test_split(X_train, y, random_state = 0)
# Define model
model_dtr = RandomForestClassifier()
# Fit model
model_dtr.fit(train_X, train_y)

# get predicted on validation data
val_predictions = model_dtr.predict(val_X)

print("MAE for out-of-sample data is:")

mae = mean_absolute_error(val_y, val_predictions)
print(round(mae, 4))


# define get_mae function that takes the maximum number of leaf nodes for dtr
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    
# define training and validation data for features and target    
    model_leaf = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    
# fit the model to the training data    
    model_leaf.fit(train_X, train_y)
    
# makes predictions on the validation data    
    preds_val = model_leaf.predict(val_X)
    
# calculates the mean absolute error    
    mae_leaf = mean_absolute_error(val_y, preds_val)
    
    return(mae_leaf)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 10, 50, 500]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %.5f" %(max_leaf_nodes, round(my_mae, 5)))

 # Define a parameter grid for the RandomForestClassifier
param_grid = {
     'n_estimators': [5, 10, 50, 100],
     'max_depth': [5, 10, 15, 20, None],
     'random_state': [0]
 }

# Initialize a GridSearchCV object with the RandomForestClassifier and the parameter grid
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y)

# Print out the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:", grid_search.best_params_)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Set seaborn style
sns.set(style='whitegrid')
 
# Plot the cumulative sum of the explained variance ratio
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
 
# Add a vertical dashed red line at x=10
plt.axvline(linewidth=2, color='r', linestyle = '--', x=5, ymin=0, ymax=1)
 
# Display the plot
plt.show()

# After training the model, we extract the feature importances
importances = model.feature_importances_

# We create a pandas DataFrame that maps these importance scores back to the feature names
feature_importances = pd.DataFrame({"feature": X_train.columns, "importance": importances})

# We sort this DataFrame by the importance scores in descending order
feature_importances = feature_importances.sort_values("importance", ascending=False)

# We create a bar plot using matplotlib, with the feature names on the y-axis and their importance scores on the x-axis
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feature_importances, palette="viridis")

# We set the title of the plot and display it
plt.title("Feature Importance")
plt.show()

# Define the features and target variable
features = ["Women", "Children", "Sex"]
X_train = pd.get_dummies(train_data[features])
y = train_data["Survived"]

# Use SelectKBest with f_classif as the scoring function to select the top K features
selector = SelectKBest(f_classif, k=5)

# Fit the SelectKBest object to the training data
selector.fit(X_train, y)

# Get the scores and p-values of the feature selection process
scores = selector.scores_
pvalues = selector.pvalues_

# Create a DataFrame with the scores and p-values
df_scores = pd.DataFrame({'Feature': X_train.columns, 'Score': scores, 'PValue': pvalues})

# Sort the DataFrame by Score in descending order
df_scores = df_scores.sort_values('Score', ascending=False)

# Display the top K features in a colorful plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=df_scores, x='Score', y='Feature', palette='viridis')
plt.title('Top 5 Features based on SelectKBest Scores')
plt.show()