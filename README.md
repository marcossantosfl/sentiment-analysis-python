# Feedback Classifier for Moodle Grades

This script leverages the power of machine learning to analyze feedback from Moodle grading data, classify them as positive or negative, and store the results. It employs a neural network, trained using TensorFlow and Keras.

## 🛠️ Functionality

1. **Data Loading and Cleaning:** The script initiates by loading grading data from a specified JSON file. This file must contain course details, including grading items and corresponding feedback texts. Feedback texts are then cleaned to remove line breaks and extra white spaces.

2. **Data Preparation:** The cleaned feedback texts undergo tokenization and transformation into sequences. They are then padded to maintain a uniform length across all sequences.

3. **Model Definition and Training:** A simple neural network model is formulated and trained on the tokenized feedback texts. The labeling of feedback as positive or negative is done based on predefined lists of positive and negative words.

4. **Model Saving:** The script saves the trained model, the tokenizer, and the maximum sequence length for future usage.

5. **Feedback Stats Updating:** The script updates and stores a file named `feedback_stats.json` which keeps a record of the number of positive and negative feedbacks for each course.

6. **Data Saving:** Finally, the script saves a new JSON file which contains the positive and negative feedbacks for each course as classified by the trained model.

## 🚀 Usage

To run the script, provide the name of the JSON file containing the grading data as a command-line argument:

```sh
python feedback_classifier.py your_file.json
