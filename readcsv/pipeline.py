import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

class Pipeline:
    def __init__(self):
        self.numerical_columns = ['ph', 'boron', 'fosfor']
        self.best_model_m = None
        self.minmax_scaler = None
        self.x_test = None #x_test = test_best_feat
        self.features_to_encode = ['Varietas', 'Warna', 'rasa', 'Musim', 'Grade_mutu', 'Penyakit', 'teknik']
        self.x = ['Varietas', 'Warna', 'rasa', 'Musim', 'Penyakit', 'teknik', 'ph', 'boron', 'fosfor']
        self.encoders = {feature: LabelEncoder() for feature in self.features_to_encode}

    def _load_model_files(self):
        self.best_model_m = joblib.load('data/best_model_m.joblib')
        self.minmax_scaler = joblib.load('data/x_scaled.joblib')
        self.x_test = joblib.load('data/x_test.joblib').tolist()

    def _label_encoder(self, df):
        for feature in self.features_to_encode:
            df[feature] = self.encoders[feature].fit_transform(df[feature])
        return df
    
    def _apply_smote(self, x, y):
        custom = {0: 900, 1: 1122, 2: 1270, 3: 1006}
        smote = SMOTE(sampling_strategy=custom, k_neighbors=3, random_state=100)
        X_resampled, y_resampled = smote.fit_resample(x, y)
        return X_resampled, y_resampled
    
    def normalization(self, x):
        x_scaled = self.minmax_scaler.fit_transform(x)
        return x_scaled
    
    def train_test_split(self, x, y, test_size=0.3, random_state=50):
        return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def preprocess(self, df):
    # Step 1: Label Encoding
        for feature in self.features_to_encode:
            df[feature] = self.encoders[feature].fit_transform(df[feature])
    
    # Step 2: Resampling using SMOTE
        X = df[self.x]
        y = df['Grade_mutu']
        X_resampled, y_resampled = self._apply_smote(X, y)
    
    # Step 3: Normalization
        X_normalized = self.normalization(X_resampled)
    
    # Step 4: Train-Test Split
        X_train, X_test, y_train, y_test = self.train_test_split(X_normalized, y_resampled)
    
        return X_train, X_test, y_train, y_test
    
    def predict(self, df):
        self._load_model_files()
        df = self._label_encoder(df)
        X_train, X_test, y_train, y_test = self.preprocess(df)
        return self.best_model_m.predict(X_test)
