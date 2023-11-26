import nltk
import random
from nltk.corpus import names
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
nltk.download('names')

# Function to extract features from a name
def gender_features(word):
    return {'last_letter': word[-1]}

# Create labeled dataset using NLTK names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])

# Shuffle the dataset
random.shuffle(labeled_names)

# Extract features and labels
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy:.2%}")

# Print the classification report
predicted_labels = [classifier.classify(features) for features, label in test_set]
true_labels = [label for features, label in test_set]
print("\nClassification Report:\n", classification_report(true_labels, predicted_labels))

# Test the model
names_to_test = ["John", "Jane", "Emma", "David", "Diana", "Michael", "Michelle"]
for name in names_to_test:
    gender = classifier.classify(gender_features(name))
    print(f"{name}: {gender}")

# Custom text input for bias detection
def detect_gender_bias(text):
    gender = classifier.classify(gender_features(text))
    return gender

# Example of using the gender bias detection function
text_to_analyze = "The nurse brought the medication to the patient."
detected_gender = detect_gender_bias(text_to_analyze)
print(f"Detected gender bias: {detected_gender}")
