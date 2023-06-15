import json
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_feedback_stats():
    full_path_feedback_stats = os.path.join("data", "user_data", "feedback_stats.json")
    if os.path.exists(full_path_feedback_stats):
        with open(full_path_feedback_stats, "r") as f:
            feedback_stats = json.load(f)
    else:
        feedback_stats = {}
    return feedback_stats

def get_feedback_counts(course_id, feedback_stats):
    positive_count = feedback_stats.get(course_id, {}).get('positive', 0)
    negative_count = feedback_stats.get(course_id, {}).get('negative', 0)
    return positive_count, negative_count

def train_and_save_feedback(filename):
    full_path = os.path.join("", "data", "user_data", filename)
    with open(full_path, 'r') as f:
        data = json.load(f)
    feedbacks = []
    for course in data:
        for item in course["items"]:
            if item["feedback"].strip():
                feedbacks.append(item["feedback"])

    # Clean the feedbacks
    cleaned_feedbacks = []
    for feedback in feedbacks:
        # Remove line breaks and extra white spaces
        cleaned_feedback = feedback.replace('\r', ' ').replace('\n', ' ').strip()
        # Replace multiple white spaces with one space
        cleaned_feedback = " ".join(cleaned_feedback.split())
        cleaned_feedbacks.append(cleaned_feedback)

    feedbacks = cleaned_feedbacks

    # Create the full path for the output file
    full_path_cleaned = os.path.join("data", "user_data", "feedback_" + filename[:-5] + ".json")

    # Save the extracted words to the output file
    with open(full_path_cleaned, "w") as f:
       json.dump(feedbacks, f)


    if len(feedbacks) > 1:
        # Check if any feedback contains positive or negative words
        positive_words = ['Excellent work', 'Great', 'Solid', 'Excellent Work', 'Overall good', 'very good', 'overall good', 'Great', 'Excellent work', 'Overall good', 'Very good', 'very good', 'excellent work', 'excellent work two', 'great two', 'Very good', 'excellent work', 'Overall good', 'overall good THREE', 'Great', 'Excellent work', 'Great']
        negative_words = ['not provided', 'Not enough', 'NOT provided', 'NOT provided', 'some issues', 'short entries']

        # Tokenize the feedback texts
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(feedbacks)

        # Create sequences of the feedback texts
        sequences = tokenizer.texts_to_sequences(feedbacks)

        # Pad the sequences so that they have the same length
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

        # Define the neural network model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(10000, 16, input_length=max_length),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras
            .layers.Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Train the model
        labels = []
        positive_feedbacks = 0
        negative_feedbacks = 0
        feedback_stats = load_feedback_stats()

        for feedback in feedbacks:
            feedback_lower = feedback.lower()
            if any(word in feedback_lower for word in positive_words) and not any(word in feedback_lower for word in negative_words):
                labels.append(1)
                positive_feedbacks += 1
            elif any(word in feedback_lower for word in negative_words) and not any(word in feedback_lower for word in positive_words):
                labels.append(0)
                negative_feedbacks += 1
            else:
                labels.append(-1)

        # Filter out feedbacks with undefined labels (-1)
        filtered_feedbacks, labels = zip(*[(feedback, label) for feedback, label in zip(feedbacks, labels) if label != -1])
        feedbacks = list(filtered_feedbacks)
        labels = list(labels)

        # Update the padded_sequences to match the filtered feedbacks
        sequences = tokenizer.texts_to_sequences(feedbacks)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")
       
        full_path_length = os.path.join("data", "user_data", "max_length.txt")
        with open(full_path_length, "w") as f:
            f.write(str(max_length))

        model.fit(padded_sequences, np.array(labels), epochs=10)

        root, ext = os.path.splitext(filename)
        trained_filename = f"{root}_trained{ext}"
        os.rename(os.path.join("data", "user_data", filename), os.path.join("data", "user_data", trained_filename))
 
        tokenizer_json = tokenizer.to_json()
        model_json = model.to_json()
        full_path_tokenizer = os.path.join("data", "user_data", "tokenizer.json")
        with open(full_path_tokenizer, "w") as f:
            json.dump(tokenizer_json, f)
        full_path_feedback_classifier = os.path.join("data", "user_data", "feedback_classifier.json")
        with open(full_path_feedback_classifier, "w") as f:
            json.dump(model_json, f)

        # Update feedback_stats
        for course in data:
            course_id = course["courseid"]
            positive_count, negative_count = get_feedback_counts(course_id, feedback_stats)

            # Calculate the number of positive and negative feedbacks for the current course
            positive_feedbacks_course = sum([1 for idx, (feedback, label) in enumerate(zip(feedbacks, labels)) if label == 1 and idx in [i for i, item in enumerate(course["items"]) if item["feedback"].strip()]])
            negative_feedbacks_course = sum([1 for idx, (feedback, label) in enumerate(zip(feedbacks, labels)) if label == 0 and idx in [i for i, item in enumerate(course["items"]) if item["feedback"].strip()]])

            feedback_stats[course_id] = {
                'positive': positive_count + positive_feedbacks_course,
                'negative': negative_count + negative_feedbacks_course
            }


        # Save updated feedback_stats to file
        full_path_feedback_stats = os.path.join("data", "user_data", "feedback_stats.json")
        with open(full_path_feedback_stats, "w") as f:
            json.dump(feedback_stats, f)

        root, ext = os.path.splitext(filename)
        student_number = root[:7] # assuming the student number is the first 7 digits of the filename
        output_filename = f"feedback_{student_number}_AI.json"
        full_path_output = os.path.join("data", "user_data", output_filename)

        output_data = []

        for course in data:
            course_id = course["courseid"]
            positive_feedbacks_list = []
            negative_feedbacks_list = []

            for idx, (feedback, label) in enumerate(zip(feedbacks, labels)):
                if label == 1 and idx in [i for i, item in enumerate(course["items"]) if item["feedback"].strip()]:
                    positive_feedbacks_list.append(feedback)
                elif label == 0 and idx in [i for i, item in enumerate(course["items"]) if item["feedback"].strip()]:
                    negative_feedbacks_list.append(feedback)

            output_data.append({
                "courseid": course_id,
                "positive": positive_feedbacks_list,
                "negative": negative_feedbacks_list
            })


        # Save the extracted words to the output file
        with open(full_path_output, "w") as f:
          json.dump(output_data, f)
    

        # Check if the files have been saved successfully
        if all(map(lambda file: os.path.exists(file), [full_path_tokenizer, full_path_feedback_classifier, os.path.join("data", "user_data", trained_filename), full_path_feedback_stats])):
            print("The trained model, tokenizer, and updated feedback stats have been saved successfully.")
        else:
            print("Failed to save the trained model, tokenizer, and updated feedback stats.")

if __name__ == '__main__':
    try:
        # Get the username and password from the command-line arguments
        filename = sys.argv[1]


        train_and_save_feedback(filename)
    except IndexError:
        print("Please provide a valid data argument.")
    except Exception as e:
        print("An error occurred:" + str(e))
