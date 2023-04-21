import os
import glob
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import dask.dataframe as dd
from functools import reduce
from dask.delayed import delayed





class dataloader():
  """
  this code process and load stress prediction data from multiple CSV files.
  The dataloader class handles the data extraction, preprocessing, and loading, 
  making it easy to use the resulting DataFrame for further analysis or modeling.
  """

  def __init__(self, path,save =True): 
    
    # Initialize the dataloader with the path to the data directory and a flag to save processed data
    self.save = save
    self.path = path

  
  def extractor(self, data_path): 
    
    # Extract CSV files from the data path and read them into a list of DataFrames
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    csv_files.sort()
    names_list = [['ACC_x', 'ACC_y', 'ACC_z'], ['BVP'], ['EDA'], ['HR'], ['IBI_t','IBI_d'], ['TEMP'], ['Tags']]
    return [pd.read_csv(f, header=None, names=name) for f, name in zip(csv_files, names_list)] 


  def labling(self, tags, time):

     # Assign stress labels (0 or 1) based on the tags and the timestamp
    if tags[0] >= time:
        return 0
    elif tags[0] < time and tags[1] >= time:
        return 1
    elif tags[1] < time and tags[2] >= time:
        return 0
    elif tags[2] < time and tags[3] >= time:
        return 1
    elif tags[3] < time and tags[4] >= time:
        return 0
    elif tags[4] < time and tags[5] >= time:
        return 1
    elif tags[5] < time:
        return 0
    else:
        print(f"Timestamp not covered: {time}")
        return None

  def making_timetable(self, participant):
    # Process the raw signals and create a timetable by assigning timestamps to each measurement
    def process_signal(signal, start_idx, freq_idx):
        start = int(signal.iloc[start_idx, 0])
        freq = int(signal.iloc[freq_idx, 0])
        signal.drop(labels=[start_idx, freq_idx], axis=0, inplace=True)
        end = start + (len(signal) / freq)
        signal = signal.assign(timestamp=np.arange(start, end, 1 / freq))
        return signal
    
    # Process the signals for each participant
    participant[0] = process_signal(participant[0], 0, 1)
    participant[1] = process_signal(participant[1], 0, 1)
    participant[2] = process_signal(participant[2], 0, 1)
    participant[3] = process_signal(participant[3], 0, 1)

    IBI_start = int(participant[4].iloc[0, 0])
    participant[4].drop(labels=[0], axis=0, inplace=True)
    participant[4]['timestamp'] = participant[4].apply(lambda x: x[0] + IBI_start, axis=1)
    
    participant[5] = process_signal(participant[5], 0, 1)
    
    return participant

  def preprocess(self, participant, f):

    # Preprocess the participant's data by merging and interpolating the signals, and adding stress labels

    # Determine start and end times of the measurements for each signal
    start_end_times = [(measurement.iloc[0, 0], measurement.iloc[0, 0] + (len(measurement)-2)/int(measurement.iloc[1, 0])) for measurement in participant[:-1]]
    earliest_start_time, latest_end_time = min([t[0] for t in start_end_times]), max([t[1] for t in start_end_times])
    
    # Calculate the total number of seconds, and create an array of evenly spaced timestamps
    num_seconds = int(latest_end_time - earliest_start_time + 1)
    seconds = np.arange(earliest_start_time, earliest_start_time + num_seconds, step=1)
    step = np.arange(0, 1, step=1/f)
    
    # Process the signals and create a timetable by assigning timestamps to each measurement
    participant = self.making_timetable(participant)

    # Create a DataFrame with timestamp values for each row
    timestamp = pd.DataFrame({"timestamp": np.array(list(itertools.chain(*[s + step for s in seconds])))})
    participant.insert(0, timestamp)
    
    # Sort the 'Tags' column and merge all DataFrames on the 'timestamp' column
    participant[-1]['Tags'] = participant[-1]['Tags'].sort_values(ascending=True)
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['timestamp'], how='left'), participant[:-1]).fillna(method="ffill").fillna(method="bfill") # merginng based on timestamp and then if there is any null value fill it with  ffill method
    df_merged['Stress'] = df_merged.apply(lambda x: self.labling(participant[-1]['Tags'], x['timestamp']), axis=1) # Add stress labels to the merged DataFrame
    df_merged.columns = ["Timestamp","ACC_x", "ACC_y", "ACC_z", "BVP", "EDA", "HR", "IBI_t", "IBI_d", "TEMP", "Stress"] # Set the column names for the merged DataFrame
    return df_merged


  def process_student_file(self, ID, path_file, fs):
    # Process a single student's file and return a DataFrame with the student's data and ID

    Student = self.preprocess(self.extractor(path_file), fs)   # Extract and preprocess the student's data
    Student['ID'] = ID + 1
    Student = Student[['ID','Timestamp', 'ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'HR', 'IBI_t','IBI_d', 'TEMP', 'Stress']]
    Student.drop(columns='IBI_t', inplace=True)  # Remove the 'IBI_t' column from the DataFrame
    return Student


  def load(self, fs=64):
    # Load the dataset, process it if needed, and return the final DataFrame. fs is for set the f for all the signals in proccessed datasett
    if self.save == False:
      data_dir = Path(self.path)
      files = sorted(data_dir.glob('*'))[2:] # Get a list of all files in the directory, sorted by name
      student_data = [delayed(self.process_student_file)(ID, path_file, fs) for ID, path_file in enumerate(files)] # Use Dask delayed to apply the process_student_file function in parallel
      final_df_dask = dd.from_delayed(student_data) # Combine the results into a Dask DataFrame
      final_df = final_df_dask.compute()  # Compute the final DataFrame
      self.save = True # Set the 'save' flag to 1 to indicate that the data has been processed
      final_df.to_csv('process_data.csv', index=False)
      return final_df
    else: # If the 'save' flag is 1, load the processed data from the CSV file
      return pd.read_csv('process_data.csv')
    
  def tags(self):
    # Extract tags from the dataset and return them in a list
    tags = [] #store all the tags for ploting
    for ID, path in enumerate(sorted(os.listdir(self.path))[2:]):
      path_file = self.path + path + '/'
      s = self.extractor(path_file)
      s = s[-1].values.reshape(1,-1)
      tags.append(s[0])
    return tags
