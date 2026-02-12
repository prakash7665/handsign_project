import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load data
df = pd.read_csv('hand_sign_data.csv', header=None)
X = df.iloc[:, :-1]  # features
y = df.iloc[:, -1]   # labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Print accuracy
print(f"Test accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Save model
with open('hand_sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as hand_sign_model.pkl")
