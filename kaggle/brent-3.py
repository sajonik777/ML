import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data
#brent_data = pd.read_csv("/kaggle/input/brent-4/brent_seasonality_comma-2.txt")
#brent_data.head()

# Function to plot data for a specific year
def plot_year(year):
    # Load the data
    brent_data = pd.read_csv("/kaggle/input/brent-4/brent_seasonality_comma-2.txt")

    # Parse the 'date' column to datetime format
    brent_data['date'] = pd.to_datetime(brent_data['date'], format='%d.%m.%Y')

    # Filter data for the specified year
    brent_data = brent_data[brent_data['date'].dt.year == year]

    # Identify the increasing and decreasing sequences
    brent_data['increasing'] = brent_data['price'] >= brent_data['price'].shift(1)
    brent_data['decreasing'] = brent_data['price'] <= brent_data['price'].shift(1)

    # Calculate the sequence lengths
    brent_data['increasing_sequence'] = brent_data['increasing'].cumsum()
    brent_data['decreasing_sequence'] = brent_data['decreasing'].cumsum()

    # Reset the sequence lengths when the price becomes less than the price two days before (for increasing sequences)
    # or more than the price two days before (for decreasing sequences)
    brent_data.loc[brent_data['price'] < brent_data['price'].shift(2), 'increasing_sequence'] = 0
    brent_data.loc[brent_data['price'] > brent_data['price'].shift(2), 'decreasing_sequence'] = 0

    # Group the data by month
    grouped = brent_data.groupby(brent_data['date'].dt.month)

    # Create a subplot for each month
    fig, axs = plt.subplots(len(grouped), 1, figsize=(10, 6 * len(grouped)))

    for (month, data), ax in zip(grouped, axs):
        # Extract the increasing and decreasing sequences for this month
        increasing_sequences = data[data['increasing_sequence'] > 0]
        decreasing_sequences = data[data['decreasing_sequence'] > 0]

        # Plot the increasing and decreasing sequences on this subplot
        for _, sequence in increasing_sequences.groupby(data['increasing_sequence']):
            ax.plot(sequence['date'], sequence['price'], color='red')
        for _, sequence in decreasing_sequences.groupby(data['decreasing_sequence']):
            ax.plot(sequence['date'], sequence['price'], color='green')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Increasing and Decreasing Price Sequences in {year}-{month:02d}')

    plt.tight_layout()
    plt.show()

# Call the function with the desired year
plot_year(2020)