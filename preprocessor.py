import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class preprocessor():

  """
  The preprocessor class is designed for preprocessing time series data, particularly for stress detection using physiological signals.
  It includes methods for scaling, splitting, and computing rolling averages of the input data. 
  The class takes the raw data as input and generates a time series dataset with a specified window size and stride.
  The scaler method scales the selected features using the StandardScaler, while the split method separates the dataset into training and testing subsets based on a given percentage of unique IDs. 
  Finally, the make_time_series method calculates the rolling average of the data within a specified window, generating a processed dataset ready for further analysis or modeling.
  """

  def __init__(self, data):
    self.data = data
    self.scaled_data = None
    self.timeserise_data = None

  def scaler(self, train_df, test_df):

    X_scaled = train_df.copy()
    X_test_scaled = test_df.copy()
    # Columns to be scaled
    columns_to_scale = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'HR', 'IBI_d', 'TEMP']

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Fit and transform the selected columns
    scaled_columns_train = scaler.fit_transform(X_scaled[columns_to_scale])
    scaled_columns_test = scaler.transform(X_test_scaled[columns_to_scale])
    # Replace the original columns with the scaled values
    X_scaled[columns_to_scale] = scaled_columns_train
    X_test_scaled[columns_to_scale] = scaled_columns_test
    return X_scaled, X_test_scaled

  def split(self, test_size= 0.2): # Create a condition to split the data based on a percentage of unique IDs
    condition = self.timeserise_data['ID'] > len(self.timeserise_data['ID'].unique()) - np.round(len(self.timeserise_data['ID'].unique()) * test_size)
    df_test = self.timeserise_data[condition] # Separate the test data based on the condition
    df_train = self.timeserise_data[~condition]
    return df_train, df_test
  
  def rolling_average(self, dataframe, window_size, stride):   # Calculate the rolling average for the given window size and select every 'stride' rows
    averaged_df = dataframe.rolling(window=window_size).mean().iloc[::stride]
    averaged_df['Stress'] = dataframe['Stress'].shift(-1).iloc[::stride]   # Shift the 'Stress' column by -1 and select every 'stride' rows
    return averaged_df

  def make_time_serises(self, window_size=64, stride=64): # Calculates the rolling average of the data within a specified window
    procced_datas = pd.DataFrame({})
    for id in self.data['ID'].unique():
      d = self.data.loc[self.data['ID'] == id].copy() # Select the data corresponding to the current ID
      d.drop(columns=['ID','Timestamp'], inplace=True)
      averaged_df = self.rolling_average(d, window_size, stride) # Calculate the rolling average for the current data
      averaged_df = averaged_df.dropna()
      averaged_df['ID'] = id
      averaged_df = averaged_df[['ID', 'ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'HR', 'IBI_d', 'TEMP', 'Stress']]
      procced_datas = pd.concat([procced_datas, averaged_df], ignore_index=True) # Concatenate the averaged_df with the main processed data
    self.timeserise_data = procced_datas