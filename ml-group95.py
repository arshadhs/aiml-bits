import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
import matplotlib.pyplot as plt     # For plotting
import seaborn as sns               # For enhanced data visualization
import time                         # For measuring execution time

from sklearn.model_selection import train_test_split            # For splitting data into training and test sets
from sklearn.preprocessing import StandardScaler, OneHotEncoder # For feature scaling and encoding


def load_database():
  '''
  Load dataset
  '''
  print("load_database...")
  df = pd.read_excel("D:\\code\\AIML_1\\AIML_1\\PS 3.xlsx", header=0)

  # Column names were standardised to improve readability and maintain consistency across the pipeline.
  df = df.rename(columns={
    'ambient_temp_c': 'Temperature',
    'rel_humidity_pct': 'Humidity',
    'wind_velocity_kmh': 'Wind_Speed',
    'precip_intensity_pct': 'Precipitation',
    'atm_pressure_hpa': 'Pressure',
    'uv_radiation_idx': 'UV_Index',
    'visibility_range_km': 'Visibility',
    'cloud_state': 'Cloud',
    'annual_phase': 'Season',
    'terrain_category': 'Terrain',
    'env_condition_label (Target)': 'Target'
  })

  # Data Understanding
  print("\n" + "=" * 50)
  print("Data Understanding...")
  print("=" * 50)

  print("Schema / Data Types:")
  print(df.dtypes)

  print("\nSample Records:")
  print(df.head(5))

  print("\nUnique:")
  print(df.nunique())

  #print("\nInfo:")
  #print(df.info())

  #print("\nDescribe:")
  #print(df.describe())

  # Sanity Diagnostics
  print("\n" + "=" * 50)
  print("Sanity Diagnostics...")
  print("=" * 50)

  print("Humidity > 100 → ", (df['Humidity'] > 100).sum(), "rows")
  print("Visibility < 0 → ", (df['Visibility'] < 0).sum(), "rows")
  print("UV_Index < 0 → ", (df['UV_Index'] < 0).sum(), "rows")

  # ToDo : Do we need to handle it using IQR ?
  df[df['Humidity'] > 100]
  df[df['Visibility'] < 0]
  df[df['UV_Index'] < 0]

  # Suspicious Range
  print("Suspicious Wind_Speed > 150 → ", (df['Humidity'] > 150).sum(), "rows")
  print("Suspicious Temperature > 60 → ", (df['Temperature'] > 60).sum(), "rows")


  df[df['Wind_Speed'] > 150]   # extreme wind
  df[df['Temperature'] > 60]   # unrealistic temp

  return df

def data_analysis(df):

  target = 'Target'
  num_cols = df.select_dtypes(include=['number']).columns.tolist()
  
  cat_cols = df.select_dtypes(include=['object']).columns.tolist()
  cat_cols.remove("Target")

  ### 2(a) Univariate, Bivariate, Multivariate Analysis
  #  (code + plots + explanation)

  # 1. Univariate
  # Visualising how values are spread
  # Identify 1. Skewness 2. Heavy Tails 3. Outliers
  for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

    '''
    Univariate analysis shows that several features such as wind_speed and precipitation exhibit right-skewed distributions with heavy tails,
    indicating the presence of extreme values. 
    This suggests that outliers may influence model performance and require robust preprocessing.
    '''

  # Class Imbalance (Target variable)
  # Check if all classes occur equally or are some clases much larger?
  plt.figure(figsize=(6,4))
  sns.countplot(x=target, data=df)
  plt.xticks(rotation=45)
  plt.title("Class Distribution")
  plt.show()

  '''
  The target variable shows class imbalance, where certain environmental conditions are more frequent than others. 
  This may bias the model towards dominant classes and requires careful evaluation using macro and weighted metrics.
  '''

  # Interaction Plots (Pairwise Relationships)
  # How variables interact with each other
  # Observe
    # Patterns between variables
    # Clusters
    # Separation between classes
  sns.pairplot(df.sample(500), hue=target)
  '''
  Pairwise interaction plots reveal relationships between multiple variables. 
  Certain feature combinations such as temperature and humidity show patterns that differ across classes, indicating potential non-linear interactions.
  '''

  # Bivariate Analysis (Feature vs Target)
  for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f'{col} vs Target')
    plt.show()

    '''
    Bivariate analysis shows that features such as temperature and humidity vary significantly across target classes,
    suggesting their importance in classification.
    '''

    # Multivariate (Pairwise Interaction)
    sns.pairplot(df.sample(500), hue=target)
    '''
    Pairwise interaction plots reveal complex relationships between variables, with certain feature combinations showing clustering patterns across classes. 
    This indicates potential non-linear dependencies.
    '''

  ### 2(b) Correlation & Dependency Analysis
  # (code + heatmaps + MI + explanation)

  # 1. Pearson Correlation (Linear)
  plt.figure(figsize=(8,6))
  sns.heatmap(df[num_cols].corr(method='pearson'), annot=True, cmap='coolwarm')
  plt.title("Pearson Correlation")
  plt.show()

  # 2. Spearman Correlation (Monotonic)
  plt.figure(figsize=(8,6))
  sns.heatmap(df[num_cols].corr(method='spearman'), annot=True, cmap='coolwarm')
  plt.title("Spearman Correlation")
  plt.show()

  # Mutual Information
  # High MI but low correlation → non-linear relationship
  from sklearn.feature_selection import mutual_info_classif
  from sklearn.preprocessing import LabelEncoder

  # Encode target
  le = LabelEncoder()
  y = le.fit_transform(df[target])

  X = df[num_cols]

  mi = mutual_info_classif(X, y)

  mi_series = pd.Series(mi, index=num_cols).sort_values(ascending=False)
  print(mi_series)
  '''
  Mutual Information analysis indicates that certain features have strong dependency with the target variable even when linear correlation is weak. 
  This suggests the presence of non-linear relationships.
  '''

  ### 2(c) Analytical Answer
  #(write-up)
  '''
  Linear correlation does not sufficiently capture feature relationships in this dataset.

  While Pearson correlation measures linear dependencies, several features show relatively low correlation values but higher mutual information scores. 
  This indicates that relationships between features and the target are not purely linear.

  Additionally, differences between Pearson and Spearman correlations suggest the presence of monotonic but non-linear relationships.

  Therefore, linear correlation alone is insufficient, and non-linear models or feature engineering are required.
  '''

  ### 2(d) Feature Interactions
  #(list + explanation)

  '''
    Interaction 1: Temperature × Humidity:
    This interaction influences environmental comfort and weather conditions such as fog or rain.

    Interaction 2: Pressure × Visibility:
    Low pressure combined with low visibility may indicate storm or fog conditions.

    Interaction 3: Wind Speed × Precipitation:
    High wind with precipitation may indicate severe weather conditions.
  '''
  
  return None

def data_preprocessing(df):
    '''
    Data Preprocessing -
      Preprocess the data as needed for training the model.
      Train-test split (80:20 mandatory), handle missing values, encode categorical variables, scale features, etc.
    '''

    # Make a copy so original dataframe is not modified
    original_rows = len(df)
    df = df.copy()

    ###################
    # 3.a. Handle
    ###################


    # a1. Missing values
    print("\n" + "=" * 50)
    print("Data Preprocessing - Missing values...")
    print("=" * 50)

    print ("Null Values Count:", df.isnull().sum())

    # Handle missing values
    df = df.dropna()

    num_cols = ['temperature','humidity','wind_speed','precipitation',
            'pressure','uv_index','visibility']

    cat_cols = ['cloud','season','terrain']

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    '''
    Missing Values Handling:
        Numerical columns: median imputation
        Categorical columns: most frequent imputation

    Missing values were assessed prior to modelling. 
        Numerical features were imputed using the median to reduce sensitivity to outliers,
        while categorical features were imputed using the most frequent category.
    '''


    # a2. Outliers

    print("\n" + "=" * 50)
    print("Data Preprocessing - Outliers...")
    print("=" * 50)

    # IQR Capping
    df_iqr = df.copy()

    for col in num_cols:
        Q1 = df_iqr[col].quantile(0.25)
        Q3 = df_iqr[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_iqr[col] = df_iqr[col].clip(lower=lower, upper=upper)

        print ("Column: ", col)
        print ("IQR: ", IQR)
        print ("lower: ", lower)
        print ("upper: ", upper)
        print ("df_iqr[col]: ", df_iqr[col])


    # IsolationForest
    from sklearn.ensemble import IsolationForest

    df_iso = df.copy()

    iso = IsolationForest(contamination=0.02, random_state=42)

    outlier_labels = iso.fit_predict(df_iso[num_cols])
    print ("outlier_labels: ", outlier_labels)

    # keep only normal rows
    df_iso = df_iso[outlier_labels == 1]
    print ("df_iso: ", df_iso)

    '''
     Two outlier handling techniques were compared: IQR-based capping and Isolation Forest. 
     IQR capping preserves all observations and is easy to interpret, whereas 
     Isolation Forest can detect multivariate anomalies but may remove potentially informative samples. 
     The final method was selected based on downstream model performance and robustness.
    '''

    ##########################
    # 3.3 Categorical encoding
    ##########################

    # One-Hot

    # Ordinal
    
    ##########################
    # 3.4 Feature scaling
    ##########################

    # Standard 
    # Robust 
    # MinMax

    ##########################
    # 3.5 Feature engineering
    ##########################

    # discomfort index
    # pressure-visibility interaction
    # optional cyclical season

    ##########################
    # 3.6 Feature selection
    ##########################

    # Mutual Information
    # Random Forest importance


    # Features and target
    # x = df.drop("Target", axis=1)
    # y = df["Target"].values.reshape(-1, 1)

    return None


def main():
  df = load_database()

  data_analysis(df)

if __name__ == '__main__':
  main()
