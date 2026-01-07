import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split

# 1. Generating Clinical Research Data
np.random.seed(42)
data = {
    'fever': np.random.choice([0, 1], size=1000),
    'dry_cough': np.random.choice([0, 1], size=1000),
    'shortness_of_breath': np.random.choice([0, 1], size=1000),
    'sore_throat': np.random.choice([0, 1], size=1000),
    'age': np.random.randint(10, 85, size=1000),
    'label': np.random.choice([0, 1], size=1000)
}
df = pd.DataFrame(data)

# 2. Preparing the Model
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Training the XGBoost Classifier
# We use scale_pos_weight to improve detection of positive cases
model = XGBClassifier(scale_pos_weight=2)
model.fit(X_train, y_train)

# 4. Save the result
joblib.dump(model, 'models/covid_prediction_model.pkl')
print("Model training complete. File saved in /models.")