# Necessary imports
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load training and testing data files for Kannada MNIST dataset
train_data = pd.read_csv(
    "C:/Users/jonna/Downloads/p2_train.csv")  # Replace 'path_to_your_train_file.csv' with your training file path
test_data = pd.read_csv(
    "C:/Users/jonna/Downloads/p2_test.csv")  # Replace 'path_to_your_test_file.csv' with your testing file path

# Assuming the last column contains the labels and the rest are features
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values


# Function to perform PCA on data
def apply_pca(train_data, test_data, n_components):
    pca = PCA(n_components=n_components)
    train_data_pca = pca.fit_transform(train_data)
    test_data_pca = pca.transform(test_data)
    return train_data_pca, test_data_pca


# Function to train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    confusion_mat = confusion_matrix(y_test, predictions)

    return accuracy, precision, recall, f1, confusion_mat


# List of models to evaluate
models = {
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'K-NN': KNeighborsClassifier(),
    'SVM': SVC()  # SVM can also be used in this context
}

# Step 4: Experimentation with Different Component Sizes
component_sizes = [10, 15, 20, 25, 30]

for component_size in component_sizes:
    # Apply PCA
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, component_size)

    print(f"Component Size: {component_size}")
    for model_name, model in models.items():
        accuracy, precision, recall, f1, confusion_mat = train_and_evaluate_model(model, X_train_pca, y_train,
                                                                                  X_test_pca, y_test)

        # Print metrics for each model
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{confusion_mat}")
