# Student ID: s343124
# Name: Muhammed Emin Oral
# Assignment #13: Protein Function Annotation

#Step 1: Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

# STEP 2: OBTAINING AND PREPROCESING THE RAW DATA

def process_raw_data():
    """
    My function to locate the raw PDB file, clean it up, and save a perfect 
    2-column CSV file (sequence, function) for the machine learning part.
    """
    print(">> Step 2: Processing RAW Data...")
    
    # I am finding the folder where this script is running
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. LOCATE RAW FILE
    # This is the messy file I downloaded from PDB
    raw_filename = 'rcsb_pdb_custom_report_20260110210009.csv'
    raw_path = os.path.join(script_dir, raw_filename)
    
    # This is the clean file I want to create
    clean_filename = 'protein_data.csv'
    clean_path = os.path.join(script_dir, clean_filename)

    # If I have the raw file, I will clean it from scratch
    if os.path.exists(raw_path):
        print(f"   I found the raw file: {raw_filename}")
        df = pd.read_csv(raw_path)
        
        # 2. FIX COLUMN NAMES
        # The PDB file has weird names like 'Polymer EntityData', so I'm renaming them
        # to 'sequence' and 'function' to make it easier to understand.
        print("   I am fixing the column names...")
        if 'Polymer EntityData' in df.columns:
            df = df.rename(columns={
                'Polymer EntityData': 'sequence',
                'StructureData': 'function'
            })
        
        # Sometimes there are repeated headers in the rows, I am deleting them.
        df = df[df['sequence'] != 'Sequence']
        
        # 3. EXTRACTING THE KNOWN FUNCTIONAL CLASSES
        print("  I am filtering the protein names to find the 4 classes...")
        
        # The titles are very long strings. I wrote this function to check 
        # if they contain the keywords I need.
        def clean_function(text):
            text = str(text).lower()
            if 'hemoglobin' in text: return 'Hemoglobin'
            elif 'myoglobin' in text: return 'Myoglobin'
            elif 'phyco' in text: return 'Phycobiliprotein'
            elif 'kinase' in text: return 'Kinase'
            else: return None # If it's none of these, I don't need it.

        # I apply my cleaning function here
        df['function_clean'] = df['function'].apply(clean_function)
        
        # I drop the rows that didn't match any of my 4 classes
        df = df.dropna(subset=['function_clean'])
        
        # 4. FINAL CLEANUP
        # I am keeping ONLY the 'sequence' and 'function' columns.
        # I don't need the 'Identifier' or other garbage columns for the model.
        df_final = df[['sequence', 'function_clean']].copy()
        df_final = df_final.rename(columns={'function_clean': 'function'})
        
        # I save this clean version so I can show that I processed the data.
        df_final.to_csv(clean_path, index=False)
        print(f"   Cleanup done. I saved the clean 2-column data to '{clean_filename}'.")
        print(f"   Total valid proteins: {len(df_final)}")
        
        return df_final

    # If the raw file is gone but I already have the clean file, I just load it.
    elif os.path.exists(clean_path):
        print("   Raw file not found, but I found the clean data. Loading it...")
        df = pd.read_csv(clean_path)
        # Just checking if column names are correct
        if 'function' not in df.columns and 'function_clean' in df.columns:
             df = df.rename(columns={'function_clean': 'function'})
        return df
    else:
        raise FileNotFoundError("Error: I couldn't find the input files!")

# Running my processing pipeline
df = process_raw_data()

#
# ENCODING (LABEL ENCODING)
print("\n>> Encoding Data (Label Encoding)...")

# The assignment asked for Label Encoding, so I am converting letters to numbers.

# Converting sequences to numbers
le_seq = LabelEncoder()
df['sequence_encoded'] = le_seq.fit_transform(df['sequence'])

# Converting function names (Hemoglobin etc.) to numbers (0, 1, 2, 3)
le_func = LabelEncoder()
df['target'] = le_func.fit_transform(df['function'])

# Checking what classes I have
classes = le_func.classes_
print(" My classes are:", classes)


# STEP 3: SPLIT DATA & TRAIN MODEL
# 
print("\n>> Step 3: Splitting Data and Training Model...")

X = df[['sequence_encoded']]
y = df['target']

# I split the data: 70% for training the model, 30% for testing it.
# I set random_state=42 so the results are the same every time I run it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# I chose Random Forest because it's good for classification.
# using 100 trees seems to give good results.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("   Model trained (Random Forest Classifier).")

# Now I test the model on data it has never seen before
y_pred = model.predict(X_test)

# STEP 4: EVALUATION
print("\n>> Step 4: Evaluation Metrics")

# Calculating the scores.
# I used 'weighted' average because my classes are unbalanced.
# (Hemoglobin has many samples, Kinase has very few).
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"   Accuracy : {accuracy:.3f}")
print(f"   Precision: {precision:.3f}")
print(f"   Recall   : {recall:.3f}")

# STEP 5: Visualization
print("\n>> Step 5: Visualizing Confusion Matrices...")

# I decide to creating TWO plots. 
# One for raw numbers (to show the dataset size)
# One for percentages (to show the actual success rate)

# FIGURE 1: RAW COUNTS :
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm, interpolation='nearest', cmap='Reds')
plt.title('Confusion Matrix (Raw Counts)')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Putting the numbers inside the boxes
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('confusion_matrix_raw.png')
print("   I saved the raw count plot as 'confusion_matrix_raw.png'")
plt.show()
plt.close()

# FIGURE 2: NORMALIZED 
plt.figure(figsize=(8, 6))

# I am normalizing the matrix (dividing by row sum).
# This is very important to see the Kinase performance clearly, 
# because Kinase has very few samples.
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

plt.imshow(cm_norm, interpolation='nearest', cmap='Reds')
plt.title('Normalized Confusion Matrix (Proportions)')
plt.colorbar()

plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Putting the percentages inside the boxes (like 0.98)
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, format(cm_norm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm_norm[i, j] > 0.5 else "black")

plt.tight_layout()
plt.savefig('confusion_matrix_norm.png')
print("   I saved the normalized plot as 'confusion_matrix_norm.png'")
plt.show()
plt.close()
