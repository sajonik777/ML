import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to plot data for a specific year
def plot_year_month(year, month):
    # Load the data
    brent_data = pd.read_csv("/kaggle/input/brent-4/brent_seasonality_comma-2.txt")

    # Parse the 'date' column to datetime format
    brent_data['date'] = pd.to_datetime(brent_data['date'], format='%d.%m.%Y')

    # Filter data for the specified year and month
    brent_data = brent_data[(brent_data['date'].dt.year == year) & (brent_data['date'].dt.month == month)]

    # Identify the increasing and decreasing sequences
    brent_data['increasing'] = True
    brent_data['decreasing'] = True

    # Calculate the sequence lengths
    brent_data['increasing_sequence'] = brent_data['increasing'].cumsum()
    brent_data['decreasing_sequence'] = brent_data['decreasing'].cumsum()

    # Reset the sequence lengths when the price becomes less than the price two days before (for increasing sequences)
    # or more than the price two days before (for decreasing sequences)
    brent_data.loc[brent_data['price'] < brent_data['price'].shift(2), 'increasing_sequence'] = 0
    brent_data.loc[brent_data['price'] > brent_data['price'].shift(2), 'decreasing_sequence'] = 0

    # Filter the increasing and decreasing sequences
    increasing_sequences = brent_data[(brent_data['increasing_sequence'] > 0) & (brent_data['increasing'] == True)]
    decreasing_sequences = brent_data[(brent_data['decreasing_sequence'] > 0) & (brent_data['decreasing'] == True)]

    # Create a subplot for each month
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the increasing and decreasing sequences on this subplot
    for _, sequence in increasing_sequences.groupby(brent_data['increasing_sequence']):
        ax.plot(sequence['date'], sequence['price'], color='red', linewidth=2, marker='o')
    for _, sequence in decreasing_sequences.groupby(brent_data['decreasing_sequence']):
        ax.plot(sequence['date'], sequence['price'], color='green', linewidth=2, marker='o')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Increasing and Decreasing Price Sequences in {year}-{month:02d}', fontsize=16)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# Call the function with the desired year and month
plot_year_month(2020, 1)