import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import keras
import streamlit as st


# Load the CSV file
file_path = '/home/amtics/Documents/dl/data/data.csv'
data = pd.read_csv(file_path)

# Extract sequences and labels
sequences = data['seq'].astype(str).tolist()
labels = data['label'].astype(int).values

# Tokenize the sequences
tokenizer = Tokenizer(char_level=True)  # Tokenize at the character level
tokenizer.fit_on_texts(sequences)

# Convert sequences to numeric data
sequences_numeric = tokenizer.texts_to_sequences(sequences)

# Pad the sequences to ensure uniform length
max_sequence_length = max(len(seq) for seq in sequences_numeric)
sequences_padded = pad_sequences(sequences_numeric, maxlen=max_sequence_length, padding='post')

# Convert labels to a numpy array
labels = np.array(labels)

# Display the shape of the padded sequences and labels
print(sequences_padded.shape, labels.shape)


# Define the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_sequence_length),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#model.fit(sequences_padded, labels, epochs=50, batch_size=32, validation_split=0.2)
#model.save("/home/amtics/Documents/dl/model.keras")
# Example new sequences
#new_sequences = "ADNPYKCNECDKVFSNSSNLVQHQRIHTGEKP"


new_sequences = st.text_input("Enter your protein sequence")
#print(type(new_sequences))
#btn_status = st.button("Predict", type="primary")
if st.button("Predit"):
	# Tokenize the new sequences
	new_sequences_numeric = tokenizer.texts_to_sequences([new_sequences])
	# Pad the new sequences to match the input length of the model
	new_sequences_padded = pad_sequences(new_sequences_numeric, maxlen=max_sequence_length, padding='post')
	model = keras.models.load_model('/home/amtics/Documents/dl/model.keras')
	# Make predictions with the model
	predictions = model.predict(new_sequences_padded)
	predicted_labels = 2
	# Convert predictions to binary labels (if using a binary classifier)
	predicted_labels = (predictions > 0.5).astype(int)

	# Print the predicted labels
	#print(predicted_labels)

	if predicted_labels == 0:
		st.title("_Your sequence is_ :blue[Not Z Protein]")
	elif predicted_labels == 1:
		st.title("_Your sequence is_ :blue[Z Protein]")
	else:
		st.write(":red[Somethingwent wrong. Please try again.]")

