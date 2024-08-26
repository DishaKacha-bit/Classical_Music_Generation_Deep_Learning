#%%
import numpy as np
import pretty_midi
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, LSTM, Dense, Softmax, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.config.experimental.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("Using CPU")

def map_duration_to_class(target_duration, duration_types):
    """Map target duration to the closest class value from an array of duration types."""
    duration_types = np.array(duration_types)  # Convert list to NumPy array
    closest_duration = min(duration_types, key=lambda x: abs(x - target_duration))
    class_index = np.where(duration_types <= closest_duration)[0][-1]  # Find the nearest value less than or equal to closest_duration
    return class_index

def create_one_hot(target_duration, duration_types):
    """Create a one-hot array based on the mapped class value."""
    class_index = map_duration_to_class(target_duration, duration_types)
    one_hot = np.zeros(len(duration_types))
    one_hot[class_index] = 1
    return one_hot


duration_types=[0.1*i for i in range(0,50)]
num_dur_classes=len(duration_types)
def load_midi_details(directory):
    """Load all MIDI files in the directory and extract detailed note sequences."""
    all_sequences = []
    for filename in os.listdir(directory):
        if filename.endswith('.mid'):
            path = os.path.join(directory, filename)
            midi_data = pretty_midi.PrettyMIDI(path)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    all_sequences.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end,
                        'duration': note.end-note.start,
                    })
    return all_sequences

def create_input_target_sequences(sequence, seq_length):
    """Create input and target sequences from detailed note sequences."""
    input_sequences = []
    target_sequences = []
    input_sequences_duration = []
    target_sequences_duration = []

    for i in range(len(sequence) - seq_length):
        # input_seq = [[event['pitch'], event['velocity'], event['start'], event['end']] for event in sequence[i:i+seq_length]]
        # target_seq = [[event['pitch'], event['velocity'], event['start'], event['end']] for event in sequence[i+1:i+1+seq_length]]
        input_seq = [[event['pitch']] for event in sequence[i:i + seq_length]]
        target_pitch = sequence[i + seq_length]['pitch']
        target_seq = to_categorical(target_pitch, num_classes=vocab_size)
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

        #Updating the duration

        input_seq = [[event['duration']] for event in sequence[i:i + seq_length]]
        target_duration = sequence[i + seq_length]['duration']
        # print("target_duration",target_duration)
        target_duration = min(target_duration, 5)
        # print("target_duration", target_duration)


        target_duration=round(target_duration, 1)
        # print(target_duration)
        # print("target_duration", target_duration)

        # target_duration=round(min(target_duration,5),1)
        # target_duration = to_categorical(target_duration, num_classes=num_dur_classes)
        print(target_duration)

        one_hot_array = create_one_hot(target_duration, duration_types)
        print(one_hot_array)
        input_sequences_duration.append(input_seq)
        target_sequences_duration.append(one_hot_array)

    print(len(input_sequences))

    return np.array(input_sequences), np.array(target_sequences),np.array(input_sequences_duration), np.array(target_sequences_duration)

#%%

def build_model(seq_length, vocab_size, embedding_dim):
    inputs = Input(shape=(seq_length,))

    # Embedding layer: Converts input sequence of token indices to sequences of vectors
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length)(inputs)

    x = LSTM(units=256, return_sequences=True)(x)  # return_sequences=True since next layer is also RNN
    x = LSTM(units=256)(x)  # Last LSTM layer does not return sequences

    # Output layer: Linear layer (Dense) with 'vocab_size' units to predict the next pitch
    outputs = Dense(vocab_size, activation='softmax')(x)  # Using softmax for the output distribution(temperature implements a weighted sampling from this)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Building model for duration. Same structure as previous

def build_model_duration(seq_length,vocab_size, embedding_dim):
    inputs = Input(shape=(seq_length,))

    # Embedding layer: Converts input sequence of token indices to sequences of vectors
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length)(inputs)

    x = LSTM(units=256, return_sequences=True)(x)  
    x = LSTM(units=256)(x)  # Last LSTM layer does not return sequences

    # Output layer: Linear layer (Dense) with 'vocab_size' units to predict the next pitch
    outputs = Dense(vocab_size, activation='softmax')(x)  # Using softmax for output distribution

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Directory containing MIDI files
directory = 'Data Sources/bach/'

# Load and preprocess data

seq_length = 30  # Length of the input sequences
vocab_size = 128  # Number of unique pitches (for MIDI, typically 128)
embedding_dim = 100 # Defining the dimension for the embedding layer
sequences = load_midi_details(directory)
input_sequences, target_sequences,input_sequences_duration,target_sequences_duration = create_input_target_sequences(sequences, seq_length)


print("Input sequences shape:", input_sequences.shape)
print("Target sequences shape:", target_sequences.shape)

print("Input sequences shape Duration:", input_sequences_duration.shape)
print("Target sequences shape Duration:", target_sequences_duration.shape)


#%%

model_duration=build_model_duration(seq_length,num_dur_classes,embedding_dim)
model_duration.summary()
model_duration.fit(input_sequences_duration, target_sequences_duration, epochs=50, batch_size=32)


# Recreate and compile the model
model = build_model(seq_length, vocab_size, embedding_dim)
model.summary()
model.fit(input_sequences, target_sequences, epochs=50, batch_size=32)



#%%
def generate_music(model, seed_sequence, length=10, steps_per_second=5, temperature=1):
    """Generate music from a seed sequence aiming for a total duration using a temperature parameter."""
    generated_sequence = np.copy(seed_sequence)  # Copy to avoid modifying the original seed
    total_steps = length * steps_per_second  # Total steps needed for desired duration

    for _ in range(total_steps):
        # Predict the next step using the last 'seq_length' elements in the generated_sequence
        prediction = model.predict(np.expand_dims(generated_sequence[-seq_length:], axis=0))[0]

        # Apply temperature to the prediction probabilities and normalize
        prediction = np.log(prediction + 1e-8) / temperature  # Smoothing and apply temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)

        # Sample an index from the probability array
        predicted_pitch = np.random.choice(len(prediction), p=prediction)

        # Append the predicted pitch to the generated sequence
        generated_sequence = np.vstack([generated_sequence, predicted_pitch])

    print("Generated sequence with variability:", generated_sequence[-30:])
    return generated_sequence



#%%
def generate_durations(model, seed_sequence, length=10, steps_per_second=5,temperature=2):
    """Generate music from a seed sequence aiming for a total duration."""
    # Ensure the seed sequence is of the correct length

    generated_sequence = np.copy(seed_sequence)  # Copy to avoid modifying the original seed
    total_steps = length * steps_per_second  # Total steps needed for desired duration
    # print(f"Total steps to generate: {total_steps}")

    for _ in range(total_steps):
        # Predict the next step using the last 'seq_length' elements in the generated_sequence
        prediction = model.predict(np.expand_dims(generated_sequence[-seq_length:], axis=0))[0]
        prediction = np.log(prediction + 1e-8) / temperature  # Smoothing and apply temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        # print(prediction)
        predicted_duration = np.random.choice(len(prediction), p=prediction)

        predicted_duration = duration_types[predicted_duration]
        generated_sequence = np.vstack([generated_sequence, predicted_duration])
        # print('Updated generated sequence:', generated_sequence)
    print("Duration Gen", generated_sequence)
    return generated_sequence




#%%
def generated_to_midi(generated_music, generated_durations, tempo=120):
    """Convert generated sequence to MIDI file, ensuring all notes are audible."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Calculate the duration per step based on tempo and number of notes
    ticks_per_beat = pm.resolution
    seconds_per_beat = 60.0 / tempo
    ticks_per_second = ticks_per_beat / seconds_per_beat

    # Initialize the current time to track the start time of each note
    current_time = 0
    for pitch, duration in zip(generated_music, generated_durations):
        pitch = int(np.clip(pitch, 21, 108))  # Scale and clip pitch values to MIDI range
        velocity = 100  # Fixed velocity for all notes


        # Convert duration to ticks
        print(duration)
        duration_ticks = duration

        # Set start and end time for each note
        start = current_time
        print("start",start)
        end = start + duration_ticks[0]
        print("end",end)
        # Create a MIDI note with the determined pitch, velocity, start, and end times
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)
        print(note)
        # Update the current time to the end of this note for the next note
        current_time = end

    # Add the instrument to the PrettyMIDI object
    pm.instruments.append(instrument)

    return pm

#%%
seed=0
seed_pitches = input_sequences[seed]
# Reshape to match expected model input shape (n, 1)
seed_sequence = (seed_pitches).reshape(-1, 1)  # Reshape to (n, 1)

# Call the generate_music function with the reshaped and normalized seed sequence
generated_music = generate_music(model, seed_sequence, length=40, steps_per_second=2)   # Generate enough steps


seed_durations = input_sequences_duration[seed]
seed_sequence = (seed_durations).reshape(-1, 1)  # Reshape to (n, 1)

generated_music_duration=generate_durations(model_duration, seed_sequence, length=40, steps_per_second=2,temperature=0.5)   # Generate enough steps


#%%
generated_music_midi = generated_to_midi(generated_music,generated_music_duration)
generated_music_midi.write('durations_1.mid')






