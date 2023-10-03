import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
median_ages_train = train_data.groupby(['Sex', 'Pclass'])['Age'].median()
train_data.fillna(0, inplace=True)
train_data.loc[train_data.Age == 0, 'Age'] = train_data[train_data.Age == 0].apply(lambda row: median_ages_train[row['Sex'], row['Pclass']], axis=1)
train_data = train_data[train_data.Age >= 10]
train_data.describe()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
median_ages_test = test_data.groupby(['Sex', 'Pclass'])['Age'].median()
test_data.fillna(0, inplace=True)
test_data.loc[test_data.Age == 0, 'Age'] = test_data[test_data.Age == 0].apply(lambda row: median_ages_test[row['Sex'], row['Pclass']], axis=1)
test_data = test_data[test_data.Age >= 10]
test_data.describe()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

y = train_data["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=30)
principalComponents = pca.fit_transform(X_scaled)

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio: ", explained_variance_ratio)

# Create labels for PCA components
labels = ['PC' + str(x) for x in range(1, len(pca.components_) + 1)]

# Create a DataFrame from the PCA components
df_pca = pd.DataFrame(pca.components_, columns=labels, index=X.columns)

# Display the transpose of the first 10 rows of the DataFrame
print(df_pca.head(10).round(5).T)

# Set seaborn style
sns.set(style='whitegrid')

# Plot the cumulative sum of the explained variance ratio
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# Add a vertical dashed red line at x=10
plt.axvline(linewidth=4, color='r', linestyle='--', x=10, ymin=0, ymax=1)

# Display the plot
plt.show()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 15, num = 15)]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose = 1, n_jobs=-1, random_state=0)
rf_random.fit(X, y)
predictions = rf_random.predict(X_test)

print(round(rf_random.score(X, y), 2))

# Create a DataFrame from cv_results_
cv_results_df = pd.DataFrame(rf_random.cv_results_)

# Drop the specified columns
cv_results_df = cv_results_df.drop([
            'mean_fit_time', 
            'std_fit_time', 
            'mean_score_time',
            'std_score_time', 
            'params', 
            'split0_test_score', 
            'split1_test_score', 
            'split2_test_score', 
            'std_test_score'],
            axis=1)

# Sort the DataFrame by rank_test_score
cv_results_df = cv_results_df.sort_values(by='rank_test_score')

# Print the DataFrame
print(cv_results_df)

# Create four figures and bar plots
for column in ['param_n_estimators', 'param_max_depth', 'mean_test_score', 'rank_test_score']:
    fig = plt.figure()
    fig.set_size_inches(30, 25)
    sns.barplot(x=cv_results_df.index, y=column, data=cv_results_df)
    plt.title(f'Bar plot of {column}')
    plt.show()

# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")