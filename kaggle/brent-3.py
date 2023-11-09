# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
# We take the average of absolute errors. 
# This is our measure of model quality 
from sklearn.model_selection import GridSearchCV, KFold



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

brent_data = pd.read_csv("/kaggle/input/brent-4/brent_seasonality_comma-2.txt")

brent_data.head()

# Parse the 'date' column to datetime format
brent_data['date'] = pd.to_datetime(brent_data['date'], format='%d.%m.%Y')

# Calculate the difference between the 'price' for each day and the next day
brent_data['price_diff'] = brent_data['price'].diff()

# Remove the first row since it has no previous day to compare with
brent_data = brent_data.iloc[1:]

# Group the data by day of the year and calculate the average difference for each day
average_diff = brent_data.groupby(brent_data['date'].dt.dayofyear)['price_diff'].mean()

# Create a new DataFrame that maps each day of the year to a date in the current year
current_year = datetime.now().year
dates = pd.DataFrame({
    'day_of_year': range(1, 366),
    'date': pd.date_range(f'{current_year}-01-01', f'{current_year}-12-31')
})

# Format the date as '%d.%m.%Y'
dates['date'] = dates['date'].dt.strftime('%d.%m.%Y')

# Merge the two DataFrames on the day of the year
result = pd.merge(dates, average_diff, left_on='day_of_year', right_index=True)

# Round the 'price_diff' column to zero decimal places
result['price_diff'] = result['price_diff'].round(0)

# Convert the 'date' column back to datetime format
result['date'] = pd.to_datetime(result['date'], format='%d.%m.%Y')

# Filter the DataFrame to include only November and December
result = result[result['date'].dt.month.isin([12])]

# Create a vertical bar plot
plt.figure(figsize=(10, 6))
colors = cm.rainbow(np.linspace(0, 1, len(result)))
plt.bar(result['date'].dt.strftime('%d.%m.%Y'), result['price_diff'], color=colors)
plt.xticks(rotation=90)
plt.xlabel('Date')
plt.ylabel('Average Price Difference')
plt.title('Average Price Difference by Date')
plt.show()

# Exclude the years 2022, 2023, 2020, and 2014
brent_data = brent_data[~brent_data.index.year.isin([2022, 2023, 2020, 2014, 2013])]

# Group the data by year and 'day of the year', and calculate the average 'price' for each day
grouped = brent_data.groupby([brent_data.index.year, brent_data.index.dayofyear])['price'].mean()

# Create a new plot figure
fig, ax = plt.subplots(figsize=(12, 6))

# Loop through each year in the grouped data
for year in grouped.index.levels[0]:
    # For each year, plot a line on the plot with x-axis as the day of the year and y-axis as the 'price'
    ax.plot(grouped[year].index, grouped[year].values, label=str(year))

# Set the plot's title, x-label, and y-label
ax.set_title('Price vs Date')
ax.set_xlabel('Day of the year')
ax.set_ylabel('Price')

# Display the legend
ax.legend(loc='upper center', ncol=len(grouped.index.levels[0]))

# Display the plot
plt.show()