import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Importing basic scikit-learn tools
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Please wait, loading the dataset from sklearn...")

# -----------------------------------------------------------------
# PART 1: DATA LOADING & PREPROCESSING 
# -----------------------------------------------------------------
# Selecting 4 easy categories for the project
my_categories = ['sci.space', 'rec.sport.hockey', 'talk.politics.mideast', 'comp.graphics']

# Loading train and test data
raw_train_data = fetch_20newsgroups(subset='train', categories=my_categories, remove=('headers', 'footers', 'quotes'))
raw_test_data = fetch_20newsgroups(subset='test', categories=my_categories, remove=('headers', 'footers', 'quotes'))

# Extracting the text text and labels
X_train_text = raw_train_data.data
y_train_labels = raw_train_data.target

X_test_text = raw_test_data.data
y_test_labels = raw_test_data.target

print("Data loaded successfully!")
print("Now converting text to numbers using TF-IDF...")

# Basic vectorizer setup with standard english stop words
my_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Transforming text into numerical arrays
X_train_vectors = my_vectorizer.fit_transform(X_train_text)
X_test_vectors = my_vectorizer.transform(X_test_text)

print("Text preprocessing completed.")

# -----------------------------------------------------------------
# PART 2: BUILDING AND TRAINING THE MODELS 
# -----------------------------------------------------------------
print("Starting model training now...")

# Model 1: Naive Bayes
naive_bayes_obj = MultinomialNB()
naive_bayes_obj.fit(X_train_vectors, y_train_labels)
print("Naive Bayes model trained.")

# Model 2: Logistic Regression
logistic_reg_obj = LogisticRegression(max_iter=500)
logistic_reg_obj.fit(X_train_vectors, y_train_labels)
print("Logistic Regression model trained.")

# -----------------------------------------------------------------
# PART 3: TESTING AND EVALUATION 
# -----------------------------------------------------------------
print("\n--- SHOWING FINAL RESULTS ---")

# 1. Testing Naive Bayes
nb_predictions = naive_bayes_obj.predict(X_test_vectors)
nb_acc = accuracy_score(y_test_labels, nb_predictions)
print("Naive Bayes Accuracy is: " + str(round(nb_acc * 100, 2)) + "%")

# 2. Testing Logistic Regression
lr_predictions = logistic_reg_obj.predict(X_test_vectors)
lr_acc = accuracy_score(y_test_labels, lr_predictions)
print("Logistic Regression Accuracy is: " + str(round(lr_acc * 100, 2)) + "%")

print("\nPrinting full classification report for Naive Bayes:")
print(classification_report(y_test_labels, nb_predictions, target_names=raw_train_data.target_names))

# Creating and saving the confusion matrix graph
print("Generating Confusion Matrix graph...")
matrix_data = confusion_matrix(y_test_labels, nb_predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=raw_train_data.target_names, 
            yticklabels=raw_train_data.target_names)
plt.title('Our Project Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')

# FIX: Automatically adjust padding so labels do not get cut off
plt.tight_layout()

plt.savefig('my_project_chart.png')
print("Graph saved in your folder as 'my_project_chart.png'!")

# -----------------------------------------------------------------
# TEST DEMO FUNCTION FOR USER INPUT
# -----------------------------------------------------------------
print("\n--- RUNNING THE LIVE DEMO TEST ---")

def check_my_sentence(user_text):
    # Process the single text input
    transformed_input = my_vectorizer.transform([user_text])
    # Predict using the trained Naive Bayes model
    result_number = naive_bayes_obj.predict(transformed_input)
    # Match number back to name
    category_name = raw_train_data.target_names[result_number[0]]
    return category_name

# Testing with an example line
sample_sentence = "The astronauts are working inside the space station."
predicted_output = check_my_sentence(sample_sentence)

print("Input Text written: " + sample_sentence)
print("System Output Prediction: " + predicted_output)
