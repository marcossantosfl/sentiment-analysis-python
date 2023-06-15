Feedback Classifier for Moodle Grades
This script is designed to analyze feedback from Moodle grading data, classify them as positive or negative, and store the results. This classifier employs a neural network trained using TensorFlow and Keras.

Functionality
This script performs the following tasks:

It loads and reads grading data from a given JSON file. The file should contain course details and grading items, including feedback texts.
The script cleans up feedback text by removing line breaks and extra white spaces.
The cleaned feedback texts are then tokenized and transformed into sequences, and padded to ensure they all have the same length.
A simple neural network model is created and trained using the tokenized feedback texts. The labels are determined based on predefined lists of positive and negative words.
The script saves the trained model, the tokenizer, and the maximum length of the sequences for future use.
The script also updates and saves a file named feedback_stats.json which keeps track of the number of positive and negative feedbacks for each course.
Lastly, the script saves a new JSON file containing the positive and negative feedbacks for each course based on the trained model's classification.
Usage
To run the script, provide the name of the JSON file containing the grading data as a command-line argument, like so:

sh
Copy code
python feedback_classifier.py your_file.json
Error Handling
The script includes error handling for common situations such as missing command-line arguments and other unexpected errors.

Requirements
Python
TensorFlow
Keras
Numpy
Ensure that these libraries are installed before running the script.

Note
The JSON file with grading data is expected to follow a certain structure. Each course's data should be an object containing course details and an array of grading items. Each grading item should be an object with an item name, percentage, and feedback.
Positive and negative words used for training the classifier are currently hardcoded into the script. You may need to adjust these based on your specific use case.
The neural network model used is a simple one and might not produce highly accurate results for complex or ambiguous feedback texts. You might want to adjust the model architecture based on your requirements and the nature of the feedback texts you are working with.
