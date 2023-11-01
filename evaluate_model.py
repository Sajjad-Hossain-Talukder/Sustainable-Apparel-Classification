import pandas as pd
import numpy as np
import sys
import os
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Reading Dataset in CSV format from Test Dataset Path 

if len(sys.argv) != 2:
    print("Usage: python evaluate_model.py /path/to/dataset_folder")
    sys.exit(1)

dataset_folder_path = sys.argv[1]

print(dataset_folder_path)

if not os.path.exists(dataset_folder_path):
    print(f"Error: The specified folder '{dataset_folder_path}' does not exist.")
    sys.exit(1)

files = os.listdir(dataset_folder_path)


if not files:
    print(f"No files found in the folder '{dataset_folder_path}'.")
    sys.exit(1)


csv_file = os.path.join(dataset_folder_path, files[0])


try:
    test_df = pd.read_csv(csv_file)
    print("\nTest Dataset loaded successfully.")
except Exception as e:
    print(f"Error reading the CSV file: {str(e)}")
    sys.exit(1)




# Feature Processing 

test_data = []
r,_ = test_df.shape

for row_index in range (0,r):
    image_data = test_df.iloc[row_index][1:].values
    image_data = image_data.reshape(28, 28)
    label = test_df.iloc[row_index]['label']
    test_data.append([image_data,label])
    

features = []
classes = [] 
for ft , lv in test_data : 
    features.append(ft)
    classes.append((int)(lv))
    
x_test = np.array(features)
y_test = np.array(classes)

x_test = x_test/255




# Loading Model 
model_file_path = 'trained_model.h5'  
loaded_model = keras.models.load_model(model_file_path)


print('Trained Model loaded successfully. Evaluating test data.........\n')


# Making prediction using test data

y_pred = loaded_model.predict(x_test)
y_pred= [np.argmax(element) for element in y_pred]
y_pred = np.array(y_pred)


# Generating Reports 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
confusion_df = pd.DataFrame(confusion, columns=class_names, index=class_names)
classification_rep = classification_report(y_test, y_pred,target_names=class_names)


#saveing the evaluation results to a fil

with open('output.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}\n')
    file.write('\n\nConfusion Matrix:\n')
    file.write(str(confusion_df) + '\n')
    file.write('\n\nClassification Report:\n')
    file.write(classification_rep)
    
 
print('Successfully evaluation results saved in output.txt\n')
sys.exit(0)