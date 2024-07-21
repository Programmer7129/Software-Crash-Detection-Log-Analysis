import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
file_path = 'Mac_2k.log_structured.csv'
data = pd.read_csv(file_path)

# Convert column names to lowercase
data.columns = data.columns.str.lower()

# Create a mapping from eventid to eventtemplate
eventid_to_eventtemplate = data.set_index('eventid')['eventtemplate'].to_dict()

# Encode categorical variables
label_encoders = {}
for column in ['month', 'user', 'component', 'content', 'eventtemplate']:
    le = LabelEncoder()
    le.fit(data[column].astype(str).tolist())  # Fit only on known classes
    data[column] = le.transform(data[column].astype(str))
    label_encoders[column] = le

# Drop irrelevant columns and keep relevant features
features = ['month', 'user', 'component', 'content', 'eventtemplate']
X = data[features]
y = data['eventid']

# Save feature names
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model, label encoders, feature names, and eventid to eventtemplate mapping
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)
with open('feature_names.pkl', 'wb') as fn_file:
    pickle.dump(feature_names, fn_file)
with open('eventid_to_eventtemplate.pkl', 'wb') as et_file:
    pickle.dump(eventid_to_eventtemplate, et_file)

# Debugging: Verify the eventid to eventtemplate mapping
print("Event ID to Event Template Mapping:")
print(eventid_to_eventtemplate)


