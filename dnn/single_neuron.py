import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

def load_database():
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    df = pd.read_csv("D:\\code\\py_1\\PythonApplication1\\auto-mpg.data", names=column_names, sep=r"\s+", na_values='?', comment='\t')

    print(df.size)
    print("============")

    # Drop rows with missing values
    df = df.dropna()

    df = df.drop('car_name', axis=1)

    return df

def data_preprocessing(df):
  # Data Preprocessing (1 Mark) -Train-test split (70-30/80–20/ 90–10), handle missing values, encode categorical variables, scale features, etc.

  # Features (x)
  x = df.drop('mpg', axis=1).to_numpy()
  print("Data Rows:", x.shape[0])
  print("Data Columns:", x.shape[1])

  # Target (y)
  y = df['mpg'].values

  # test_size: Determines the proportion of data for the test set (e.g., 0.2 for 20%). If not specified, it defaults to 0.25.
  # random_state: An integer seed (like 42 or 0) that ensures you get the exact same split every time you run the code, which is vital for reproducibility.
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
  print ("Train Set:", x_train.shape[0])
  print ("Test Set:", x_test.shape[0])

  # Standardise

  # Initialize the scaler
  scalar = StandardScaler()

  # 1. Fit and transform the training data
  # fit_transform: Calculates the mean and standard deviation of X_train and immediately applies the scaling.
  x_train_s = scalar.fit_transform(x_train)
  #y_train_y = scalar.fit_transform(y_train.reshape(-1, 1))

  # 2. Transform the test data (using the training mean/std)
  # transform: Uses those exact same calculations to scale the X_test. This ensures your model treats the test data exactly like it treated the training data.
  x_test_s = scalar.fit_transform(x_test)
  #y_test_y = scalar.fit_transform(y_test.reshape(-1, 1))

  print(x_train_s.mean(axis=0)) # Should be close to 0
  print(x_train_s.std(axis=0))  # Should be 1

  return x_train_s, x_test_s, y_train, y_test
  # z = train(x, y_or, b, w, n)
  # print (z)

  
class SingleNeuron(object):
    def __init__(self, learning_rate, iteration):
        print ("Initalise Seingle Neuron")
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weight = 1
        self.bias = 1

    def predict(self, x, weights):
        return np.dot(x, weights) + self.bias


    def train(self, x, y):
        print ("x_train_s: ", x)
        print ("y_train: ", y)

        N, n_features = x.shape
        weights = np.zeros(n_features)

        for i in range(self.iteration):
            # Forward Pass
            y_pred = self.predict(x, weights)
            print ("y_pred: ", y_pred)

            # Compute Loss
            loss = 1/(2*N) * np.sum((y_pred - y) *(y_pred - y))
            print ("Loss: ", loss)

            # Gradient
            errors = y_pred - y
            grad_w = (1/N) * np.dot(x.T, errors)
            grad_b = (1/N) * np.sum(errors)

            print("Errors: ", errors)
            print("grad_w", grad_w)
            print("grad_b", grad_b)

            # Update Weights
            weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b


def main():
  df = load_database()
  x_train_s, x_test_s, y_train, y_test = data_preprocessing(df)

  # Train Model
  model = SingleNeuron(learning_rate = 0.05, iteration=10)
  model.train(x_train_s, y_train)

if __name__ == '__main__':
    main()
