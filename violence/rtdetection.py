import os
import shutil
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow 
import keras
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
 
from keras.layers import *
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# === EXTRACT FRAMES ===

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16

DATASET_DIR = "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset"

CLASSES_LIST = ["Non Violence", "Violence"]

def frames_extraction(video_path):
    
    frames_list = []

    # read the video file
    video_reader = cv2.VideoCapture(video_path)

    # fetch the total number of frames in the video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # calculate interval after which frames will be added to the list
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # iterate through video frames
    for frame_counter in range(SEQUENCE_LENGTH):

        # set current frame position of the video
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # read frame from the video
        success, frame = video_reader.read()

        if not success:
            break
        
        # resize frame to fixed width and height
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # normalize resized frame
        normalized_frame = resized_frame / 255

        # append normalized frame into the frames list
        frames_list.append(normalized_frame)
    
    video_reader.release()

    return frames_list

# === CREATE DATA ===

def create_dataset():

    features = []
    labels = []
    video_fpath = []

    # iterate through all classes
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f"extracting data of class: {class_name}")

        # list of video files in the specific class name directory
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

        # iterate through all files in the files directory
        for file_name in files_list:

            # get complete video path
            video_path = os.path.join(DATASET_DIR, class_name, file_name)

            # extract frames of video file
            frames = frames_extraction(video_fpath)

            # ignore videos whose frame length < sequence length
            if len(frames) == SEQUENCE_LENGTH:

                # append data to respective lists
                features.append(frames)
                labels.append(class_index)
                video_fpath.append(video_path)
    
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, video_fpath

# dataset gets created here
features, labels, video_fpath = create_dataset()

# save the extracted data
np.save("features.npy", features)
np.save("labels.npy", labels)
np.save("video_fpath.npy", video_fpath)

features, labels, video_fpath = np.load("features.npy"), np.load("labels.npy"), np.load("video_fpath.npy")

# === ENCODING AND SPLITTING ===

# convert labels into one hot encoded vectors
one_hot_encoded_labels = to_categorical(labels)

# split train/test 90/10
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size=0.1, shuffle=True, random_state=42)

print(features_train.shape, labels_train.shape)
print(features_test.shape, labels_test.shape)

# === IMPORT MOBILENET AND FINE-TUNE ===

mobilenet = MobileNetV2(include_top=False, weights="imagenet")

# finetune to make last 40 layers trainable
mobilenet.trainable = True

for layer in mobilenet.layers[:-40]:
    layer.trainable = False

# === BUILD THE MODEL ===
def create_model():
    
    model = Sequential()

    # specify input to match features shape
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    # pass mobilenet into the time distributed layer to handle the sequence
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))

    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards=True)

    model.add(Bidirectional(lstm_fw, backward_layer = lstm_bw))

    model.add(Dropout(0.25))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    model.summary()

    return model


# constructing the model
mobile_lstm = create_model()

# plot structure of constructed LRCN model
plot_model(mobile_lstm, to_file='mobile_lstm.png', show_shapes=True, show_layer_names=True)

# === SPECIFY CALLBACKS AND FIT ===

# create early stopping callback to monitor accuracy
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# reduce the learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=5e-5, verbose=1)

# compile model
mobile_lstm.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

# fit model
mobile_lstm_history = mobile_lstm.fit(x=features_train, y=labels_train, epochs=50, batch_size=8, shuffle=True,
                                      validation_split=0.2, callbacks=[early_stopping_callback, reduce_lr])


# === MODEL EVALUATION ===
def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # get epochs count
    epochs = range(len(metric_value_1))

    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'orange', label=metric_name_2)

    plt.title(str(plot_name))

    plt.legend()

plot_metric(mobile_lstm_history, 'loss', 'val_loss', 'Total Loss vs Validation Loss')

plot_metric(mobile_lstm_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Validation Accuracy')

# === PREDICT TEST SET ===

labels_predict = mobile_lstm.predict(features_test)

# decoding data to use in metrics
labels_predict = np.argmax(labels_predict, axis=1)
labels_test_normal = np.argmax(labels_test, axis=1)

print(f"Shapes: {labels_test_normal.shape}, {labels_predict.shape}")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_predict, labels_test_normal)
print(f"Accuracy: {accuracy}")

ax = plt.subplot()
cm = confusion_matrix(labels_test_normal, labels_predict)
sns.heatmap(cm, annot=True, fmt='g', ax=ax)

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['True', 'False'])
ax.yaxis.set_ticklabels(['Non Violence', 'Violence'])

report = classification_report(labels_test_normal, labels_predict)
print('Classification Report:\n', report)

# === FRAME-BY-FRAME PREDICTION ===

def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):

    # read video file
    video_reader = cv2.VideoCapture(video_file_path)

    # width and height of video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # store output video in the disk
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter,
                                   cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                   video_reader.get(cv2.CAP_PROP_FPS),
                                   (original_video_width, original_video_height))
    
    # declare queue to store video frames
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # store predicted class in the video
    predicted_class_name = ''

    # iterate until video is accessed successfully
    while video_reader.isOpened():
        ok, frame = video_reader.read()

        if not ok:
            break

        # resize frame to fixed dimensions
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # normalize resized image
        normalized_frame = resized_frame / 255

        # append preprocessed frame into frames list
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            pred_labels_probs = mobile_lstm.predict(np.expand_dims(frames_queue, axis=0))[0]

            # index of class with highest prob
            predicted_label = np.argmax(pred_labels_probs)

            # class name using retrieved index
            predicted_class_name = CLASSES_LIST[predicted_label]

            # write predicted class name on top of frame
            if predicted_class_name == 'Violence':
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
            else:
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)
            
            # write frame onto disk
            video_writer.write(frame)

        video_reader.release()
        video_writer.release()

plt.style.use("default")

# To show Random Frames from the saved output predicted video (output predicted video doesn't show on the notebook but can be downloaded)
def show_pred_frames(pred_video_path): 

    plt.figure(figsize=(20,15))

    video_reader = cv2.VideoCapture(pred_video_path)

    # Get the number of frames in the video.
    frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get Random Frames from the video then Sort it
    random_range = sorted(random.sample(range (SEQUENCE_LENGTH , frames_count ), 12))
        
    for counter, random_index in enumerate(random_range, 1):
        
        plt.subplot(5, 4, counter)

        # Set the current frame position of the video.  
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, random_index)
          
        ok, frame = video_reader.read() 

        if not ok:
          break 

        frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        plt.imshow(frame);ax.figure.set_size_inches(20,20);plt.tight_layout()
                            
    video_reader.release()

# Construct the output video path.
test_videos_directory = 'test_videos'
os.makedirs(test_videos_directory, exist_ok = True)
 
output_video_file_path = f'{test_videos_directory}/Output-Test-Video.mp4'

# Specifying video to be predicted
input_video_file_path = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/Violence/V_378.mp4"

# Perform Prediction on the Test Video.
predict_frames(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)

# Show random frames from the output video
show_pred_frames(output_video_file_path)

# Specifying video to be predicted
input_video_file_path = "../input/real-life-violence-situations-dataset/Real Life Violence Dataset/NonViolence/NV_1.mp4"

# Perform Prediction on the Test Video.
predict_frames(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)

# Show random frames from the output video
show_pred_frames(output_video_file_path)


# === PREDICTION FOR VIDEO ===

def predict_video(video_file_path, SEQUENCE_LENGTH):
 
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Store the predicted class in the video.
    predicted_class_name = ''
 
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
 
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        success, frame = video_reader.read() 
 
        if not success:
            break
 
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
 
    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = mobile_lstm.predict(np.expand_dims(frames_list, axis = 0))[0]
 
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
 
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted class along with the prediction confidence.
    print(f'Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    video_reader.release()
