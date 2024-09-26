import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('traffic_data.csv')

# Categorize traffic flow based on Vehicle_Count
df['Traffic_Flow_Category'] = pd.cut(df['Vehicle_Count'], bins=[0, 10, 30, 100], labels=['Low', 'Medium', 'High'])

# Check for missing values or invalid categories
print(df['Traffic_Flow_Category'].isna().sum())  # Check for NaNs

# Apply label encoding to the target
label_encoder = LabelEncoder()
df['Traffic_Flow_Category'] = label_encoder.fit_transform(df['Traffic_Flow_Category'])

# Define features and target
X = df[['Light_Timing', 'Traffic_Density', 'Vehicle_Count']]  # Features
y = df['Traffic_Flow_Category']  # Encoded label

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
logistic_model = LogisticRegression()
knn_model = KNeighborsClassifier()
random_forest_model = RandomForestClassifier()

# Voting classifier to combine the models
voting_classifier = VotingClassifier(estimators=[
    ('lr', logistic_model),
    ('knn', knn_model),
    ('rf', random_forest_model)
], voting='hard')

# Train the voting classifier
voting_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = voting_classifier.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Best model accuracy: {accuracy:.2f}")



