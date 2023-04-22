# Analysing Empatica E4 watch data

1. DataLoader.py
this code process and load stress prediction data from multiple CSV files.The dataloader class handles the data extraction, preprocessing, and loading, making it easy to use the resulting DataFrame for further analysis or modeling.

2. perproccess.py
The preprocessor class is designed for preprocessing time series data, particularly for stress detection using physiological signals. It includes methods for scaling, splitting, and computing rolling averages of the input data. The class takes the raw data as input and generates a time series dataset with a specified window size and stride.The scaler method scales the selected features using the StandardScaler, while the split method separates the dataset into training and testing subsets based on a given percentage of unique IDs. Finally, the make_time_series method calculates the rolling average of the data within a specified window, generating a processed dataset ready for further analysis or modeling.

3. Model.py
The Model class, which inherits from RandomForestClassifier, is designed to train and evaluate a stress detection model using physiological signals. The class initializes a RandomForestClassifier with the provided keyword arguments and contains methods for training, testing, and evaluating the model. The train method performs Leave-One-Subject-Out cross-validation, reporting performance metrics for each iteration. The test method evaluates the model on a separate test dataset, while the evaluation method computes accuracy and F1 score for the given predictions. Additionally, the important_features method displays the feature importances of the trained model, visualizing them as a bar plot to help users understand the significance of each feature in the model.

4.
 
