import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix


df = pd.read_csv('airline_passenger_satisfaction.csv')
df.drop(columns=['id', 'Unnamed: 0'], inplace=True)
df.dropna(inplace=True)

label_encoder = LabelEncoder()
columns = ['Gender', 'Type of Travel', 'Customer Type', 'Class']
for col in columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop(columns='satisfaction')
y = df['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")


def evaluate_model(name, model, X_train, X_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    predictions = model.predict(X_test)
    predict_time = time.time() - start

    cm = confusion_matrix(y_test, predictions)

    print("")
    print("=" * 55)
    print(name)
    print("=" * 55)
    print(f"Accuracy:                {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision (satisfied):   {precision_score(y_test, predictions, pos_label='satisfied'):.4f}")
    print(f"Recall (satisfied):      {recall_score(y_test, predictions, pos_label='satisfied'):.4f}")
    print(f"F1-Score:                {f1_score(y_test, predictions, pos_label='satisfied'):.4f}")
    print(f"F-Beta 0.5 (precision):  {fbeta_score(y_test, predictions, beta=0.5, average='weighted'):.4f}")
    print(f"F-Beta 2.0 (recall):     {fbeta_score(y_test, predictions, beta=2.0, average='weighted'):.4f}")
    print(f"Training time:           {train_time:.1f} sec")
    print(f"Prediction time:         {predict_time:.1f} sec")
    print("")
    print(f"True Negative  (correctly 'dissatisfied'): {cm[0, 0]}")
    print(f"False Positive (wrongly 'satisfied'):      {cm[0, 1]}")
    print(f"False Negative (wrongly 'dissatisfied'):   {cm[1, 0]}")
    print(f"True Positive  (correctly 'satisfied'):    {cm[1, 1]}")

    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, pos_label='satisfied'),
        'Recall': recall_score(y_test, predictions, pos_label='satisfied'),
        'F1': f1_score(y_test, predictions, pos_label='satisfied'),
        'Train time': train_time,
        'Predict time': predict_time
    }


results = []
results.append(evaluate_model('Random Forest', RandomForestClassifier(random_state=42), X_train, X_test))
results.append(evaluate_model('K Neighbors', KNeighborsClassifier(n_neighbors=5), X_train, X_test))
results.append(evaluate_model('Logistic Regression', LogisticRegression(max_iter=10000), X_train, X_test))

print("")
print("=" * 55)
print("COMPARISON")
print("=" * 55)
print(pd.DataFrame(results).to_string(index=False, float_format='%.4f'))

print("")
print("=" * 55)
print("FEATURE SCALES")
print("=" * 55)
scales = X.describe().loc[['min', 'max']].T
scales['range'] = scales['max'] - scales['min']
print(scales.sort_values('range', ascending=False).head(5).to_string(float_format='%.1f'))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_results = []
scaled_results.append(evaluate_model('Random Forest (scaled)', RandomForestClassifier(random_state=42), X_train_scaled, X_test_scaled))
scaled_results.append(evaluate_model('K Neighbors (scaled)', KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled))
scaled_results.append(evaluate_model('Logistic Regression (scaled)', LogisticRegression(max_iter=10000), X_train_scaled, X_test_scaled))

print("")
print("=" * 55)
print("COMPARISON AFTER SCALING")
print("=" * 55)
print(pd.DataFrame(results + scaled_results).to_string(index=False, float_format='%.4f'))
