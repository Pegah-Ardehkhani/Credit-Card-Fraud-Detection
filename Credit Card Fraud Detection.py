# **Import Libraries and Data**
"""

!pip install kaggle # Install the Kaggle API library, which allows interaction with datasets on Kaggle
!mkdir ~/.kaggle # Create a hidden directory named `.kaggle` in the home directory to store the Kaggle API key
!cp kaggle.json ~/.kaggle/ # Copy the `kaggle.json` file (which contains Kaggle API credentials) into the `.kaggle` directory
!chmod 600 ~/.kaggle/kaggle.json # Change the permission of the `kaggle.json` file to be readable and writable only by the owner
!kaggle datasets download mlg-ulb/creditcardfraud # Use the Kaggle API to download the credit card fraud detection dataset from the specified Kaggle dataset URL
!unzip /content/creditcardfraud.zip # Unzip the downloaded dataset file into the current working directory (usually `/content` in Google Colab)

import sklearn
print(sklearn.__version__)

!pip install scikit-learn==1.3.2

!pip install ydata-profiling  # Install the ydata-profiling library for automated EDA

from ydata_profiling import ProfileReport  # Import ProfileReport for automated EDA
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computations
import tensorflow as tf  # Deep learning framework
from tensorflow import keras  # High-level API for building models
from tensorflow.keras import layers  # Neural network layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.model_selection import train_test_split  # Split data into train/test sets
from sklearn.preprocessing import StandardScaler  # Normalize features
from sklearn.feature_selection import SelectKBest, f_classif # Import feature selection tools
from sklearn.svm import LinearSVC # Using LinearSVC for efficiency on large datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score  # Evaluation metrics
from imblearn.over_sampling import SMOTE  # Handle class imbalance with synthetic samples
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # Statistical data visualization
import missingno as msno
import warnings
warnings.filterwarnings('ignore') # ignore warnings
pd.set_option('display.max_columns', None) # to display the total number columns present in the dataset

df = pd.read_csv('/content/creditcard.csv')  # Load the credit card fraud dataset into a DataFrame

"""Credit card fraud poses a significant threat to financial institutions and consumers worldwide, leading to substantial financial losses and erosion of trust. The goal of this project is to develop robust machine learning models capable of accurately identifying fraudulent transactions within a highly imbalanced dataset. This Kaggle challenge dataset, characterized by anonymized principal components ('V' features) along with 'Time' and 'Amount' transaction details, presents a realistic scenario for building effective fraud detection systems.

Our Objectives:
Our primary objectives in this analysis are multifaceted, aiming to build a comprehensive understanding of the dataset and develop high-performing predictive models:

Feature Selection and Importance Analysis: We aim to identify the most influential features that contribute significantly to distinguishing between legitimate and fraudulent transactions. We will leverage statistical methods like ANOVA and model-based techniques such as Random Forest Feature Importance to rank features based on their predictive power. This step is crucial for understanding the underlying patterns of fraud and potentially reducing model complexity.

Model Performance Evaluation and Comparison: We will train and evaluate various classification models, including Random Forest, Decision Tree, Support Vector Machine (SVM), Artificial Neural Network (ANN), and Transformer models. Our evaluation will primarily rely on Receiver Operating Characteristic (ROC) curves and their corresponding Area Under the Curve (AUC) scores, which are particularly suitable for assessing performance on imbalanced datasets. This will allow us to compare the strengths and weaknesses of different algorithms in this context.

Building a Robust Fraud Detection System: Ultimately, our goal is to identify the most effective model and a subset of highly relevant features to construct a reliable and accurate credit card fraud detection system. This system should minimize false positives (legitimate transactions flagged as fraud) while maximizing true positives (actual fraudulent transactions correctly identified), thereby providing a valuable tool for preventing financial losses.

By systematically analyzing feature importance and rigorously comparing model performances, we aim to deliver a solution that not only achieves high accuracy but also provides actionable insights into the nature of credit card fraud.

# **Exploratory Data Analysis (EDA)**
"""

# Create a profile report for the dataset with a custom title and full-width styling
profile = ProfileReport(
    df,
    title="Credit Card Fraud Detection Dataset Profile",
    html={"style": {"full_width": True}},
    sort=None,  # Keep original column order
    progress_bar=True  # Display progress bar during report generation
)

# Save the generated profile report as an HTML file
output_file = "credit_card_fraud_profile_report.html"
profile.to_file(output_file)

df.head(5) # Display the first 10 rows of the dataset to preview the data

df.columns.to_list() # Display columns names

df.describe().transpose()  # Get summary statistics for each column and transpose for better readability

df.shape # Get the dimensions of the dataset (rows, columns)

num_cols = df.select_dtypes(include=['number']).shape[1] # Count numerical columns (int, float types)
cat_cols = df.select_dtypes(include=['object', 'category']).shape[1] # Count categorical columns (object, category types)

print(f"Numerical columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

"""**Numerical Features Only:**

- All features are numerical, making scaling straightforward (e.g., StandardScaler).

- No categorical variables to encode.
"""

df.duplicated().sum()

df.isna().sum()  # Count the number of missing (NaN) values in each column

msno.matrix(df);

"""**No Missing Data:**

The dataset contains no missing values, simplifying preprocessing.
"""

df['Class'].value_counts()  # Count the number of samples in each class (fraud vs. non-fraud)

# Calculate and print the percentage of non-fraudulent transactions
print('No Frauds', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% of the dataset')

# Calculate and print the percentage of fraudulent transactions
print('Frauds', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% of the dataset')

"""**Highly Imbalanced Dataset:**

The target variable Class is heavily skewed:

Non-fraudulent transactions (Class = 0) make up ~99.83%

Fraudulent transactions (Class = 1) make up only ~0.17%

This imbalance means models must be carefully trained to avoid bias towards the majority class.
"""

sns.countplot(x=df['Class'])  # Plot count of each class (fraud vs no fraud)
plt.title('Class Distributions \n (0: No Fraud, 1: Fraud)')  # Add title to the plot
plt.show();  # Display the plot

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=False)
plt.show();

# Plot distribution of the 'Amount' feature with KDE and red color
sns.displot(df['Amount'], color='r', kde=True, height=4, aspect=2)
plt.title('Distribution of Transaction Amount', fontsize=14)
plt.xlim([df['Amount'].min(), df['Amount'].max()])
plt.show()

# Plot distribution of the 'Time' feature with KDE and blue color
sns.displot(df['Time'], color='b', kde=True, height=4, aspect=2)
plt.title('Distribution of Transaction Time', fontsize=14)
plt.xlim([df['Time'].min(), df['Time'].max()])
plt.show()

"""**Feature Distributions:**

- Transaction Amount: Distribution is right-skewed — most transactions have low amounts, but a few are very large. This might help distinguish frauds, as frauds may cluster at different amount ranges.

- Transaction Time: Spans a wide range; analyzing frauds by time (e.g., odd hours) could provide useful signals.
"""

# Using the ANOVA statistical method as a feature selection technique

features = df.loc[:, :'Amount']  # Select all columns from start to 'Amount' as features
target = df.loc[:, 'Class']  # Select the target column

best_features = SelectKBest(score_func=f_classif, k='all')  # Initialize ANOVA F-test with all features
fit = best_features.fit(features, target)  # Fit the model

# Create a DataFrame of feature scores
featureScores = pd.DataFrame(data=fit.scores_, index=list(features.columns), columns=['ANOVA Score'])
featureScores = featureScores.sort_values(ascending=False, by='ANOVA Score')  # Sort by score descending

# Plot heatmap of ANOVA scores
plt.figure(figsize=(5, 10))
sns.heatmap(featureScores, annot=True, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
plt.title('ANOVA Score');  # Set title

"""This plot displays a heatmap titled "ANOVA Score," which represents the results of an ANOVA (Analysis of Variance) statistical test applied as a feature selection technique. The table-like structure shows various features and their corresponding ANOVA scores, sorted in descending order of importance.

ANOVA as a Feature Selection Method: The code explicitly states that ANOVA (specifically, f_classif for classification tasks) is used to calculate the variance between groups (e.g., fraudulent vs. non-fraudulent transactions) for each feature. A higher ANOVA score indicates that the feature is more effective at discriminating between the classes.

Feature Ranking:

Top Features: 'V17' has the highest ANOVA score of 33979.17, followed by 'V14' (28695.55), 'V12' (20749.82), 'V10' (14057.98), and 'V16' (11443.35). These features exhibit the largest differences in means across the target classes, suggesting they are highly relevant for predicting credit card fraud. The warmer colors (light orange to deep red) visually emphasize their high scores.

Mid-Range Features: Features like 'V3', 'V7', 'V11', 'V4', and 'V18' also show substantial ANOVA scores, indicating their importance.

Least Important Features: Towards the bottom, features such as 'V22' (0.18), 'V23' (2.05), 'V25' (3.12), 'V26' (5.08), 'V13' (5.95), and 'Amount' (9.03) have very low ANOVA scores. This implies that their means are not significantly different across the fraud and non-fraud classes, making them less impactful for distinguishing between them.

Comparison with Random Forest Importance (if available): It's interesting to note the different rankings compared to a Random Forest feature importance chart (if you have one). While some top features might overlap (e.g., V14, V17), the relative importance can differ as ANOVA is a univariate statistical test, while Random Forest considers interactions between features.

Insights from 'Time' and 'Amount': Similar to the Random Forest feature importance, the 'Time' and 'Amount' features have relatively low ANOVA scores (43.25 and 9.03, respectively). This reinforces the idea that, in this dataset, the anonymized 'V' features are far more discriminative for fraud detection.
"""

# Calculate the correlation matrix
corr_matrix = df.corr()

# Set the figure size
plt.figure(figsize=(15, 12))

# Create a heatmap of the correlation matrix
sns.heatmap(corr_matrix,
            annot=False,        # Set to True if you want correlation values shown
            cmap='coolwarm',    # Colormap for better visualization
            linewidths=0.5,
            linecolor='white')

# Set plot title
plt.title('Correlation Matrix of Credit Card Fraud Dataset')

# Show the plot
plt.show()

"""# **Preprocessing**

### **Dataset Splitting:**
"""

X = df.drop('Class', axis=1)  # Drop the target column 'Class' to create the feature matrix X
y = df['Class']  # Assign the target column 'Class' to the variable y (labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the shape (rows, columns) of the training and test feature sets
print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")
# Print the normalized distribution (as percentages) of the target classes in the training set
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
# Print the normalized distribution (as percentages) of the target classes in the test set
print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

"""### **Feature Scaling:**"""

scaler = StandardScaler()  # Initialize StandardScaler to standardize features by removing the mean and scaling to unit variance

# Fit the scaler on 'Time' and 'Amount' columns of the training set and transform them
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])

# Use the same scaler parameters to transform 'Time' and 'Amount' columns in the test set
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

"""### **Handle Class Imbalance:**"""

smote = SMOTE(random_state=42)  # Initialize SMOTE to handle class imbalance by generating synthetic minority samples

# Apply SMOTE to the training data to resample and balance the classes
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print the shape of the resampled training features
print(f"Resampled training set shape: {X_train_resampled.shape}")

# Print the class distribution after resampling (should be balanced)
print(f"Resampled training target distribution:\n{y_train_resampled.value_counts(normalize=True)}")

"""### **Prepare Data for Transformer Input:**"""

num_features = X_train_resampled.shape[1]  # Get the number of features after resampling
embedding_dim = 32  # Set embedding dimension for each feature in the Transformer model

# Reshape training data to fit Transformer input format: (samples, features, 1)
X_train_transformer = X_train_resampled.values.reshape(-1, num_features, 1)

# Reshape test data the same way
X_test_transformer = X_test.values.reshape(-1, num_features, 1)

# Print the new shapes of the data prepared for Transformer input
print(f"\nTransformer input shape (training): {X_train_transformer.shape}")
print(f"Transformer input shape (testing): {X_test_transformer.shape}")

"""# **Model**

### **1. Linear SVC:**
"""

# LinearSVC is suitable for large datasets and uses a linear kernel.
# 'dual=False' is preferred when n_samples > n_features.
# 'max_iter' might need adjustment for convergence.
svm_model = LinearSVC(random_state=42, dual=False, max_iter=10000) # Increased max_iter for convergence
svm_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the Model
y_pred = svm_model.predict(X_test)

# For ROC AUC, LinearSVC doesn't directly provide predict_proba.
# We can use decision_function and then scale it or simply use predict for classification metrics.
# If probability estimates are crucial, you might need to use CalibratedClassifierCV with LinearSVC
# or switch to SVC(probability=True) which is slower.
# For simplicity, we'll use decision_function for AUC if available, otherwise just classification report.
y_pred_proba = svm_model.decision_function(X_test)
# For binary classification, decision_function output can be directly used for AUC
# as it represents distance to hyperplane.
roc_auc_SVM = roc_auc_score(y_test, y_pred_proba)
accuracy_SVM = accuracy_score(y_test, y_pred)
print(f"ROC AUC Score (SVM) (using decision_function): {roc_auc_SVM:.4f}")
print(f"Accuracy (SVM): {accuracy_SVM:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.title('Confusion Matrix for Linear SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""### **2. Decision Tree:**"""

# Decision Tree with default settings (you can tune max_depth, min_samples_split, etc.)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_resampled, y_train_resampled)

# Predict labels and probabilities
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# ROC AUC Score using probabilities
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
accuracy_dt = accuracy_score(y_test, y_pred)
print(f"ROC AUC Score (DT): {roc_auc_dt:.4f}")
print(f"Accuracy (DT): {accuracy_dt:.4f}")

# Classification report
print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))

# Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Plotting Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""### **3. Random Forest:**"""

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=30,        # Number of trees
    random_state=42
    )
rf_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]  # Probability of the positive class (fraud)

roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
accuracy_rf = accuracy_score(y_test, y_pred)
print(f"ROC AUC Score (RF): {roc_auc_rf:.4f}")
print(f"Accuracy (RF): {accuracy_rf:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""### **3. ANN:**"""

# Define the ANN architecture
ann_model = Sequential([
    Dense(64, input_dim=X_train_resampled.shape[1], activation='relu'),
    Dropout(0.3),  # Dropout to prevent overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
ann_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        AUC(name='auc'),
        Precision(name='precision'),
        Recall(name='recall')
    ]
)

# Train the model
history = ann_model.fit(X_train_resampled, y_train_resampled,
                        epochs=20,
                        batch_size=64,
                        validation_data=(X_test, y_test),
                        verbose=1)

# Predict probabilities and class labels
y_pred_proba_ann = ann_model.predict(X_test).ravel()
y_pred_ann = (y_pred_proba_ann > 0.5).astype(int)

# ROC AUC Score
print(f"Accuracy (ANN): {ann_model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
roc_auc_ann = roc_auc_score(y_test, y_pred_proba_ann)
print(f"ROC AUC Score (ANN): {roc_auc_ann:.4f}")

# Classification report
print("\nClassification Report (ANN):")
print(classification_report(y_test, y_pred_ann))

# Confusion matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)

# Plotting Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.title('Confusion Matrix for ANN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""### **4. Transformer:**"""

y_train_resampled.astype("float32")
y_test = y_test.astype("float32")

# Build the Transformer Model

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Multi-Head Self-Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x) # Self-attention
    x = layers.Dropout(dropout)(x)
    res = x + inputs # Residual connection

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x) # Project back to original dim
    return x + res # Another residual connection

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)

    # Initial projection for each feature to embedding_dim
    x = layers.Dense(embedding_dim, activation='relu')(inputs) # Shape (batch, num_features, embedding_dim)

    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    # Global average pooling to flatten the sequence of feature embeddings
    x = layers.GlobalAveragePooling1D()(x)

    # MLP head for classification
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x) # Binary classification

    return keras.Model(inputs, outputs)

# Model parameters
input_shape = (num_features, 1) # Each feature is a 'token'
head_size = 256
num_heads = 4
ff_dim = 4
num_transformer_blocks = 2
mlp_units = [128]
dropout = 0.1
mlp_dropout = 0.1

model = build_transformer_model(
    input_shape,
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=mlp_units,
    dropout=dropout,
    mlp_dropout=mlp_dropout,
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
)

model.summary()

# Train the Model
print("\nTraining the Transformer model...")
history = model.fit(
    X_train_transformer,
    y_train_resampled.astype("float32"),
    validation_data=(X_test_transformer, y_test),
    epochs=20,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ],
    verbose=1
)

# Evaluate the Model
print("\nEvaluating the model on the test set...")
y_pred_proba_tf = model.predict(X_test_transformer).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Normal', 'Predicted Fraud'],
            yticklabels=['Actual Normal', 'Actual Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

roc_auc_tf = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score (Transformer): {roc_auc_tf:.4f}")
# Compute Accuracy (optional, if not using model.evaluate)
accuracy_tf = accuracy_score(y_test, y_pred)
print(f"Accuracy (Transformer): {accuracy_tf:.4f}")

# Plotting training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()

"""# **Compare Models**"""

# Create the DataFrame for feature importances
RF_I = rf_model  # just matching your naming
d = {
    'Features': X_train.columns,
    'Feature Importance': RF_I.feature_importances_
}
df = pd.DataFrame(d)
df_sorted = df.sort_values(by='Feature Importance', ascending=True).reset_index(drop=True)

# Normalize feature importance for color mapping
norm = plt.Normalize(df_sorted['Feature Importance'].min(), df_sorted['Feature Importance'].max())
colors = plt.cm.Blues(norm(df_sorted['Feature Importance'].values))

# Plotting
plt.figure(figsize=(10, 8))
bars = plt.barh(df_sorted['Features'], df_sorted['Feature Importance'], color=colors, edgecolor='black')

# Add feature importance values to the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
             f'{width:.4f}', va='center', fontsize=9)

plt.title('Feature Importance Based on Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

"""- Most Important Features: 'V14' stands out as the most important feature with an importance score of 0.2759, followed by 'V17' (0.1109), 'V3' (0.0999), and 'V4' (0.0887). These top features appear to be significantly more influential in the model's predictions compared to the others, as indicated by their longer bars and higher values. The use of darker shades of blue for these top features also highlights their prominence.

- Decreasing Importance: As we move down the list, the importance of the features gradually decreases. Features like 'V10', 'V12', and 'V16' still show notable importance.

- Least Important Features: Towards the bottom of the chart, features such as 'V24', 'V23', 'V25', and 'V22' have very low importance scores, suggesting they contribute minimally to the model's predictive power in this specific context.

- 'Amount' and 'Time' Features: Interestingly, the 'Amount' and 'Time' features, which are often fundamental in fraud detection datasets, are ranked quite low (0.0082 and 0.0069 respectively). This might imply that the 'V' features (which are typically PCA-transformed features in fraud datasets to protect user privacy) have captured most of the variance related to fraud, or that 'Amount' and 'Time' are less directly indicative of fraud after these transformations.
"""

# Compute ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_proba)
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_pred_proba_ann)
fpr_tf, tpr_tf, _ = roc_curve(y_test, y_pred_proba_tf)

# Plotting all ROC curves
plt.figure(figsize=(10, 7))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.4f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_SVM:.4f})')
plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC = {roc_auc_ann:.4f})')
plt.plot(fpr_tf, tpr_tf, label=f'Transformer (AUC = {roc_auc_tf:.4f})')

# Diagonal line for random chance
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

# Labels and formatting
plt.title('ROC Curve Comparison for All Models', fontsize=14)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

"""This plot displays a "ROC Curve Comparison for All Models," which is a standard way to evaluate the performance of binary classification models, especially in imbalanced datasets like those found in credit card fraud detection.

ROC Curve (Receiver Operating Characteristic Curve): Each colored line represents the ROC curve for a different classification model. The curve plots the True Positive Rate (Sensitivity) on the y-axis against the False Positive Rate (1 - Specificity) on the x-axis at various threshold settings.

True Positive Rate (TPR): The proportion of actual positive cases (e.g., fraudulent transactions) that were correctly identified.

False Positive Rate (FPR): The proportion of actual negative cases (e.g., legitimate transactions) that were incorrectly identified as positive.

Area Under the Curve (AUC): The AUC value for each model is provided in the legend. The AUC quantifies the overall performance of the classifier.

An AUC of 1.0 indicates a perfect classifier.

An AUC of 0.5 (represented by the "Random Guess" dashed line) indicates a classifier that performs no better than random chance.

Higher AUC values generally indicate better model performance.

- Model Performance Comparison:

SVM (Support Vector Machine) (AUC = 0.9739): This model has the highest AUC among all models, suggesting it is the best performer in distinguishing between fraudulent and legitimate transactions. Its curve is closest to the top-left corner, indicating a high true positive rate with a low false positive rate.

Random Forest (AUC = 0.9614): This model also performs exceptionally well, very close to SVM. Its curve is nearly indistinguishable from SVM for much of the plot.

ANN (Artificial Neural Network) (AUC = 0.9540): ANN shows strong performance, slightly below Random Forest and SVM.

Transformer (AUC = 0.9460): The Transformer model, while still performing well above random guess, is slightly behind the top three.

Decision Tree (AUC = 0.8865): This model has the lowest AUC among the evaluated classification models, indicating it is the least effective in this comparison. Its curve is further away from the top-left corner compared to the other models.

"Random Guess" Line (Dashed Black Line): This diagonal line represents a classifier that performs no better than random chance. Any model performing above this line is better than random, and the further away from this line (towards the top-left corner), the better the model.
"""

# Create your model metrics (replace with your actual metrics if you have cross-validation means)
accuracy_ann = ann_model.evaluate(X_test, y_test, verbose=0)[1]
model_names = ['Random Forest', 'Decision Tree', 'SVM', 'ANN', 'Transformer']
accuracy_means = [accuracy_rf, accuracy_dt, accuracy_SVM, accuracy_ann, accuracy_tf]
auc_means = [roc_auc_rf, roc_auc_dt, roc_auc_SVM, roc_auc_ann, roc_auc_tf]

# Build the DataFrame
compare = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy_means,
    'AUC Score': auc_means,
})

# Style the table with background gradient
compare.style.background_gradient(cmap='YlGn')

"""This table presents a comparative evaluation of five machine learning models — Random Forest, Decision Tree, SVM, ANN, and a Transformer-based model — on a binary classification task (fraud detection). The models are assessed using two key metrics: Accuracy and AUC Score (Area Under the ROC Curve), which together provide insight into overall performance and class discrimination capability.

Random Forest achieved the highest accuracy (99.95%), indicating it correctly classified almost all samples. It also delivered a strong AUC of 0.96, showing high capability in distinguishing between classes.

SVM performed exceptionally well in terms of AUC (0.97) — the highest among all models — suggesting it is particularly good at ranking positive and negative samples, despite a slightly lower accuracy.

ANN (Artificial Neural Network) closely followed Random Forest, with accuracy near 99.9% and an AUC of 0.95, balancing both precision and generalization well.

Transformer, while slightly behind in accuracy (97.3%), still showed strong AUC performance (0.94), indicating good potential especially with further tuning or more data.

Decision Tree, as expected from a simpler model, had the lowest AUC (0.88) and accuracy (97.7%), suggesting it may overfit or underperform in complex feature interactions.

Random Forest and SVM emerge as the most robust models in this setup, with excellent accuracy and discriminative power. The ANN and Transformer models also perform competitively, and may benefit further from architectural tuning. Decision Tree, though interpretable, may not be optimal for this task without enhancements like pruning or ensembling.
"""