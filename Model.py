import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score



# Custom Model class inherits from RandomForestClassifier and DecisionTreeClassifier
class Model:
  """
  The Model class, which inherits from RandomForestClassifier, is designed to train and evaluate a stress detection model using physiological signals.
  The class initializes a RandomForestClassifier with the provided keyword arguments and contains methods for training, testing, and evaluating the model.
  The train method performs Leave-One-Subject-Out cross-validation, reporting performance metrics for each iteration.
  The test method evaluates the model on a separate test dataset, while the evaluation method computes accuracy and F1 score for the given predictions. 
  Additionally, the important_features method displays the feature importances of the trained model, visualizing them as a bar plot to help users understand the significance of each feature in the model.
  """

  def __init__(self, model_type="random_forest", **kwargs): # Initialize the model as a RandomForestClassifier of DecisionTreeClassifier with the given keyword arguments
      if model_type == "random_forest":
        self.model = RandomForestClassifier(**kwargs)
      elif model_type == "decision_tree":
        self.model = DecisionTreeClassifier(**kwargs)
      else:
        raise ValueError("Invalid model_type. Choose either 'random_forest' or 'decision_tree'.")


  def train(self, data): # Train the model on the given data using Leave-One-Subject-Out cross-validation

    scores = []
    for id in data['ID'].unique():

      # Split the data into training and validation sets
      val_df = data.loc[data['ID'] == id].copy()
      train_df = data.drop(val_df.index)

      y_val = val_df['Stress']
      X_val = val_df.drop(columns=['Stress', 'ID'])

      y_train = train_df['Stress']
      X_train = train_df.drop(columns=['Stress', 'ID'])

      # Train the model and make predictions
      self.model.fit(X_train, y_train)
      y_pred_train = self.model.predict(X_train)
      y_pred_val = self.model.predict(X_val)

      # Evaluate the model and store the scores
      train_score = self.evaluation(y_train, y_pred_train)
      val_score = self.evaluation(y_val, y_pred_val)
      scores.append(val_score)
      print(f'participant {id}/28    train_accuracy = {train_score[0]*100:.2f}    train_F1 = {train_score[1]*100:0.2f}   val_accuracy = {val_score[0]*100:0.2f}    val_F1 = {val_score[1]*100:.2f}')
      
    scores = np.array(scores)
    column_means = np.mean(scores, axis=0)
    column_stds = np.std(scores, axis=0)

    print('\n\n\n')
    print(f'average validation accuracy:{column_means[0]*100:.2f} +- {column_stds[0]*100:.2f}')
    print(f'average validation F1:{column_means[1]*100:.2f} +- {column_stds[1]*100:.2f}')
    print('\n\n\n')



  def test(self, train_data, test_data):     # Test the model on the given train and test datasets
    
    # Separate the features and target variable
    y_test = test_data['Stress']
    X_test = test_data.drop(columns=['Stress','ID'])

    y_train = train_data['Stress']
    X_train = train_data.drop(columns=['Stress', 'ID'])

    # Train the model on the training data and make predictions on the test data
    self.model.fit(X_train, y_train)
    y_pred = self.model.predict(X_test)

    # Evaluate the model and print the scores
    scores = self.evaluation(y_test, y_pred)
    print(f'Test Accuracy: {scores[0]*100:.2f}')
    print(f'Test F1: {scores[1]*100:.2f}')

  def evaluation(self,y_val, y_pred): # Evaluate the model's performance by calculating the accuracy, F1 score

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    return [accuracy, f1]
  


  def important_features(self, X_train):   # Display the importance of each feature in the trained model

    importances = self.model.feature_importances_ # get the importance of the features that I used for training
    feature_importances = pd.Series(importances, index=X_train.columns)
    sorted_feature_importances = feature_importances.sort_values(ascending=False)
    print('\n\n' + "Feature Importances:")
    print(sorted_feature_importances)
    print('\n\n')
  

    # Colors for the bars
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'brown', 'black', 'orange']

    plt.figure(figsize=(20, 3)) 
    bar_width = 1

    gap_width = 0.3
    bar_positions = [i + i * gap_width for i in range(len(sorted_feature_importances.index))]
    # Create a bar plot with different colors for each bar
    plt.bar(bar_positions, sorted_feature_importances.values, color=colors, width=bar_width, alpha=0.7)
  
    plt.xticks([pos + bar_width / 2 for pos in bar_positions], sorted_feature_importances.index)

    # Add labels and title
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Importance of Features')

    # Display the plot
    plt.show()