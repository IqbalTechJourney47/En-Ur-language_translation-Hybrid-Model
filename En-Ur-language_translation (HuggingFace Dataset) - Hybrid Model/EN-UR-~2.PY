
!pip install datasets transformers[sentencepiece] sacrebleu -q

import os
import sys
import transformers
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Input, Concatenate

model_checkpoint = 'Helsinki-NLP/opus-mt-en-ur'

raw_datasets = load_dataset('HaiderSultanArc/MT-Urdu-English')

raw_datasets['train'][0]

from datasets import load_dataset

# Load the dataset
raw_datasets = load_dataset('HaiderSultanArc/MT-Urdu-English')

# Take a small subset for testing/debugging
small_en_ur_train_dataset = raw_datasets['train'].shuffle(seed=42).select([i for i in range(500)])  # Adjust the number as needed
small_en_ur_test_dataset = raw_datasets['test'].shuffle(seed=42).select([i for i in range(300)])  # Adjust the number as needed

# Display the small datasets
small_en_ur_dataset = {
    "train": small_en_ur_train_dataset,
    "test": small_en_ur_test_dataset,
}

small_en_ur_dataset

small_en_ur_dataset['train'][0]

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 64
max_target_length = 64

source_lang = "en"
target_lang = "ur"

# Define your preprocess_function as before
def preprocess_function(examples):
    inputs = examples[source_lang]
    targets = examples[target_lang]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    # Add labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

preprocess_function(small_en_ur_dataset["train"][:2])

# Apply preprocess_function to train and test datasets
tokenized_train_datasets = small_en_ur_dataset["train"].map(preprocess_function, batched=True)

tokenized_test_datasets = small_en_ur_dataset["test"].map(preprocess_function, batched=True)

# Apply preprocess function to train and test datasets
tokenized_train_datasets = raw_datasets['train'].map(preprocess_function, batched=True)

tokenized_test_datasets = raw_datasets['test'].map(preprocess_function, batched=True)

from tensorflow.keras.layers import GlobalMaxPooling1D, RepeatVector

# Define CNN and RNN components
embedding_dim = 128

#cnn_input = Input(shape=(max_input_length, embedding_dim))
# cnn_input = Input(shape=(max_input_length, embedding_dim), name="input_9")

cnn_input = tf.keras.layers.Input(shape=(max_input_length, 128), name="cnn_input")
cnn_output = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_input)
cnn_output_pooled = GlobalMaxPooling1D()(cnn_output)
cnn_output_repeated = RepeatVector(max_input_length)(cnn_output_pooled)

rnn_input = Input(shape=(max_input_length, embedding_dim))
rnn_output = LSTM(units=64, return_sequences=True)(rnn_input)

# Concatenate or combine CNN and RNN outputs
combined_output = Concatenate()([cnn_output_repeated, rnn_output])

# Target_text column in dataset
target_texts = small_en_ur_train_dataset["ur"]  # Replace with the actual column name

final_model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Use the tokenizer to convert each target text to input IDs for the decoder
decoder_input_ids_list = [tokenizer(text, return_tensors="tf", max_length=max_target_length, truncation=True)["input_ids"] for text in target_texts]

# Find the maximum length among the decoded input IDs
max_decoder_length = max(len(ids[0]) for ids in decoder_input_ids_list)

# Get the padding token ID from the tokenizer
padding_token_id = tokenizer.pad_token_id

# Ensure all sequences have the same length by truncating or padding
decoder_input_ids_padded = tf.keras.preprocessing.sequence.pad_sequences([ids[0] for ids in decoder_input_ids_list], maxlen=max_decoder_length, padding='post', value=padding_token_id)

# Print the lengths before padding
print("Lengths before padding:", [len(ids[0]) for ids in decoder_input_ids_list])

# Print the lengths after ensuring consistent length
print("Lengths after ensuring consistent length:", [len(ids) for ids in decoder_input_ids_padded])

# Convert the padded input IDs to a TensorFlow tensor
decoder_input_ids = tf.constant(decoder_input_ids_padded)

# Use the tokenizer to convert each target text to input IDs for the decoder
decoder_input_ids_list = [tokenizer(text, return_tensors="tf", max_length=max_target_length, truncation=True)["input_ids"] for text in target_texts]

# Find the maximum length among the decoded input IDs
max_decoder_length = max(len(ids) for ids in decoder_input_ids_list)

# Get the padding token ID from the tokenizer
padding_token_id = tokenizer.pad_token_id

# Ensure all sequences have the same length by truncating or padding
decoder_input_ids_padded = tf.keras.preprocessing.sequence.pad_sequences([ids[0] for ids in decoder_input_ids_list], maxlen=max_decoder_length, padding='post', value=padding_token_id)

# Print the lengths before padding
print("Lengths before padding:", [len(ids) for ids in decoder_input_ids_list])

# Print the lengths after ensuring consistent length
print("Lengths after ensuring consistent length:", [len(ids) for ids in decoder_input_ids_padded])

# Convert the padded input IDs to a TensorFlow tensor
decoder_input_ids = tf.constant(decoder_input_ids_padded)

# Check the tokenizer's padding value
padding_token_id = tokenizer.pad_token_id
print("Tokenizer Padding Value:", padding_token_id)

# Adjust the input shape to match the model's expectation
# final_model_input = Input(shape=(max_input_length,))

# final_model_input = Input(shape=(max_input_length,), name="input_9")
# final_model_input = Input(shape=(max_input_length,), name="input_12")

final_model_input = tf.keras.layers.Input(shape=(max_input_length,), name="final_model_input")

from tensorflow.keras.layers import TimeDistributed, Flatten, Input, Dense
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from tensorflow.keras.layers import Embedding

# Get the predicted output
# final_model_output = final_model(final_model_input, decoder_input_ids=decoder_input_ids)

# Get model outputs
final_model_output = final_model(final_model_input, decoder_input_ids=tokenized_train_datasets['labels'])

# Access the logits tensor
logits = final_model_output.logits

# Print the shape of the logits before flattening
print("Shape before flattening:", logits.shape)

# Apply TimeDistributed and Flatten
flattened_output = TimeDistributed(Flatten())(logits)

# Get the size of the Urdu vocabulary using the tokenizer
target_vocab_size = len(tokenizer.get_vocab())

# Flatten the logits
flattened_output = Flatten()(flattened_output)

# Define the final dense layer
#output_layer = Dense(target_vocab_size, activation='softmax')(flattened_output)

# Use Embedding layer for vocabulary projection
embedding_dim = 128  # Choose an appropriate embedding dimension
# embedding_layer = Embedding(input_dim=target_vocab_size, output_dim=embedding_dim)(flattened_output)
# embedding_layer = Embedding(input_dim=target_vocab_size, output_dim=embedding_dim, name="output_layer")(flattened_output)
embedding_layer = tf.keras.layers.Embedding(input_dim=target_vocab_size, output_dim=embedding_dim, name="output_layer")(flattened_output)

# Compile the model and perform training
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_data, epochs=num_epochs, validation_data=val_data)

# Update the model to include CNN and RNN components

# model = tf.keras.Model(inputs=[cnn_input, final_model_input], outputs={'embedding_4': embedding_layer}, name="en_text_to_urdu_model")

# Define the model
model = tf.keras.Model(inputs=[cnn_input, final_model_input], outputs={'embedding': embedding_layer}, name="en_text_to_urdu_model")

# Check the keys of tokenized_train_datasets
print("Keys of tokenized_train_datasets:", tokenized_train_datasets.column_names)

# Check the keys of tokenized_test_datasets
print("Keys of tokenized_test_datasets:", tokenized_test_datasets.column_names)

padding_values = 0

# Find the maximum lengths
max_cnn_input_length = max(len(x['en']) for x in tokenized_train_datasets)
max_final_model_input_length = max(len(x['ur']) for x in tokenized_train_datasets)
max_decoder_input_ids_length = max(len(x['input_ids']) for x in tokenized_train_datasets)
max_labels_length = max(len(x['labels']) for x in tokenized_train_datasets)

print("Max CNN Input Length:", max_cnn_input_length)
print("Max Final Model Input Length:", max_final_model_input_length)
print("Max Decoder Input IDs Length:", max_decoder_input_ids_length)
print("Max Labels Length:", max_labels_length)

# Get the tokenizer's padding value
padding_token_id = tokenizer.pad_token_id

# Set appropriate padding values
padding_values = {
    'cnn_input': max_cnn_input_length,
    'final_model_input': max_final_model_input_length,
    'decoder_input_ids': padding_token_id,
    'labels': padding_token_id
}

# Define batch size
batch_size = 8

# Update the model to include CNN and RNN components
# model = tf.keras.Model(inputs=[cnn_input, final_model_input], outputs={'embedding_4': embedding_layer}, name="en_text_to_urdu_model")
model = tf.keras.Model(inputs=[cnn_input, final_model_input], outputs={'embedding': embedding_layer}, name="en_text_to_urdu_model")

# Rename the keys in the data dictionary to match the expected input names
train_batched_dataset = tf.data.Dataset.from_generator(
    lambda: ({
        'cnn_input': x['en'],
        'final_model_input': x['ur'],
        'decoder_input_ids': x['input_ids'],
        'labels': x['labels']
    } for x in tokenized_train_datasets),
    output_signature={
        'cnn_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'final_model_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'decoder_input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
).padded_batch(batch_size=16, padding_values=padding_values)

# Rename the keys in the data dictionary for the validation dataset as well
val_batched_dataset = tf.data.Dataset.from_generator(
    lambda: ({
        'cnn_input': x['en'],
        'final_model_input': x['ur'],
        'decoder_input_ids': x['input_ids'],
        'labels': x['labels']
    } for x in tokenized_test_datasets),
    output_signature={
        'cnn_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'final_model_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'decoder_input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
).padded_batch(batch_size=16, padding_values=padding_values)

# Check the keys of tokenized_train_datasets
print("Keys of tokenized_train_datasets:", tokenized_train_datasets.column_names)

# Check the keys of tokenized_test_datasets
print("Keys of tokenized_test_datasets:", tokenized_test_datasets.column_names)

Keys of tokenized_train_datasets: ['en', 'ur', 'input_ids', 'attention_mask', 'labels']
Keys of tokenized_test_datasets: ['en', 'ur', 'input_ids', 'attention_mask', 'labels']

# Rename the keys in the data dictionary to match the expected input names
train_batched_dataset = tf.data.Dataset.from_generator(
    lambda: ({
        'cnn_input': x['en'],
        'final_model_input': x['ur'],
        'decoder_input_ids': x['input_ids'],
        'labels': x['labels']
    } for x in tokenized_train_datasets),
    output_signature={
        'cnn_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'final_model_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'decoder_input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
).padded_batch(batch_size=16, padding_values=padding_values)

# Rename the keys in the data dictionary for the validation dataset as well
val_batched_dataset = tf.data.Dataset.from_generator(
    lambda: ({
        'cnn_input': x['en'],
        'final_model_input': x['ur'],
        'decoder_input_ids': x['input_ids'],
        'labels': x['labels']
    } for x in tokenized_test_datasets),
    output_signature={
        'cnn_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'final_model_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'decoder_input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
).padded_batch(batch_size=16, padding_values=padding_values)

# Check the keys of tokenized_train_datasets
print("Keys of tokenized_train_datasets:", train_batched_dataset.element_spec)

# Check the keys of tokenized_test_datasets
print("Keys of tokenized_test_datasets:", val_batched_dataset.element_spec)

Keys of tokenized_train_datasets: {'cnn_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'final_model_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'decoder_input_ids': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'labels': TensorSpec(shape=(None, None), dtype=tf.int32, name=None)}
Keys of tokenized_test_datasets: {'cnn_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'final_model_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'decoder_input_ids': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'labels': TensorSpec(shape=(None, None), dtype=tf.int32, name=None)}

num_epochs = 1

# Print model summary
model.summary()

Model: "en_text_to_urdu_model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_12 (InputLayer)       [(None, 128)]                0         []                            
                                                                                                  
 tf_marian_mt_model_5 (TFMa  TFSeq2SeqLMOutput(loss=Non   7648160   ['input_12[0][0]']            
 rianMTModel)                e, logits=(500, 1, 62025),   9                                       
                              past_key_values=(((500, 8                                           
                             , 1, 64),                                                            
                              (500, 8, 1, 64),                                                    
                              (500, 8, None, 64),                                                 
                              (500, 8, None, 64)),                                                
                              ((500, 8, 1, 64),                                                   
                              (500, 8, 1, 64),                                                    
                              (500, 8, None, 64),                                                 
                              (500, 8, None, 64)),                                                
                              ((500, 8, 1, 64),                                                   
                              (500, 8, 1, 64),                                                    
                              (500, 8, None, 64),                                                 
                              (500, 8, None, 64)),                                                
                              ((500, 8, 1, 64),                                                   
                              (500, 8, 1, 64),                                                    
                              (500, 8, None, 64),                                                 
                              (500, 8, None, 64)),                                                
                              ((500, 8, 1, 64),                                                   
                              (500, 8, 1, 64),                                                    
                              (500, 8, None, 64),                                                 
                              (500, 8, None, 64)),                                                
                              ((500, 8, 1, 64),                                                   
                              (500, 8, 1, 64),                                                    
                              (500, 8, None, 64),                                                 
                              (500, 8, None, 64))),                                               
                              decoder_hidden_states=Non                                           
                             e, decoder_attentions=None                                           
                             , cross_attentions=None, e                                           
                             ncoder_last_hidden_state=(                                           
                             None, 128, 512),                                                     
                              encoder_hidden_states=Non                                           
                             e, encoder_attentions=None                                           
                             )                                                                    
                                                                                                  
 time_distributed_4 (TimeDi  (500, 1, 62025)              0         ['tf_marian_mt_model_5[0][1]']
 stributed)                                                                                       
                                                                                                  
 flatten_9 (Flatten)         (500, 62025)                 0         ['time_distributed_4[0][0]']  
                                                                                                  
 input_9 (InputLayer)        [(None, 128, 128)]           0         []                            
                                                                                                  
 output_layer (Embedding)    (500, 62025, 128)            7939200   ['flatten_9[0][0]']           
                                                                                                  
==================================================================================================
Total params: 84420809 (322.04 MB)
Trainable params: 84358784 (321.80 MB)
Non-trainable params: 62025 (242.29 KB)

# Update the model to include CNN and RNN components
# model = tf.keras.Model(inputs=[cnn_input, final_model_input], outputs={'embedding_4': embedding_layer}, name="en_text_to_urdu_model")

model = tf.keras.Model(inputs=[cnn_input, final_model_input], outputs={'embedding': embedding_layer}, name="en_text_to_urdu_model")

from transformers import MarianConfig

# Replace 'your_marian_model_name_or_path' with the actual name or path of the Marian model
model_checkpoint = 'Helsinki-NLP/opus-mt-en-ur'

# Load the model configuration
config = MarianConfig.from_pretrained(model_checkpoint)

# Print the entire configuration
print(config)

# Access encoder-related parameters
encoder_layers = config.encoder_layers
encoder_ffn_dim = config.encoder_ffn_dim
encoder_attention_heads = config.encoder_attention_heads

# Print the encoder-related parameters
print("Encoder Layers:", encoder_layers)
print("Encoder FFN Dimension:", encoder_ffn_dim)
print("Encoder Attention Heads:", encoder_attention_heads)

MarianConfig {
  "_name_or_path": "/tmp/Helsinki-NLP/opus-mt-en-ur",
  "activation_dropout": 0.0,
  "activation_function": "swish",
  "add_bias_logits": false,
  "add_final_layer_norm": false,
  "architectures": [
    "MarianMTModel"
  ],
  "attention_dropout": 0.0,
  "bad_words_ids": [
    [
      62024
    ]
  ],
  "bos_token_id": 0,
  "classif_dropout": 0.0,
  "classifier_dropout": 0.0,
  "d_model": 512,
  "decoder_attention_heads": 8,
  "decoder_ffn_dim": 2048,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 6,
  "decoder_start_token_id": 62024,
  "decoder_vocab_size": 62025,
  "dropout": 0.1,
  "encoder_attention_heads": 8,
  "encoder_ffn_dim": 2048,
  "encoder_layerdrop": 0.0,
  "encoder_layers": 6,
  "eos_token_id": 0,
  "extra_pos_embeddings": 62025,
  "forced_eos_token_id": 0,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "init_std": 0.02,
  "is_encoder_decoder": true,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "max_length": 512,
  "max_position_embeddings": 512,
  "model_type": "marian",
  "normalize_before": false,
  "normalize_embedding": false,
  "num_beams": 4,
  "num_hidden_layers": 6,
  "pad_token_id": 62024,
  "scale_embedding": true,
  "share_encoder_decoder_embeddings": true,
  "static_position_embeddings": true,
  "transformers_version": "4.36.0.dev0",
  "use_cache": true,
  "vocab_size": 62025
}

Encoder Layers: 6
Encoder FFN Dimension: 2048
Encoder Attention Heads: 8

# Inspect the first few rows of the training dataset
train_df = tokenized_train_datasets.to_pandas()
print(train_df.head())

en  \
0  And why are the kneeling on your right and on ...   
1                       She said: "She frightens me.   
2              We will seek your guidance and input.   
3  Can you see the real motive of God the Father ...   
4            "Oh, you're writing historical novels!"   

                                                  ur  \
0  ’’اور آپ ﷺ نے اپنی بائیں ہتھیلی کو بائیں ران ا...   
1             کہا اس نے مجھے بس بد دعا سے خوف آتا ہے   
2  ہم تجھ سے ہدایت، تقوی، عفت اور بے نیازی طلب کر...   
3  ان مثالوں سے آپ خوب سمجھ سکتے ہیں کہ خدا کی اس...   
4                             ” آپ داستان لکھتے ہیں!   

                                           input_ids  \
0  [57, 1082, 56, 3, 19626, 95, 76, 418, 7, 95, 7...   
1   [1598, 108, 52, 82, 24319, 10376, 66, 179, 5, 0]   
2         [63, 42, 766, 76, 596, 7, 26, 16350, 5, 0]   
3  [2112, 17, 307, 3, 2915, 13784, 10, 71, 3, 219...   
4  [82, 9467, 2, 17, 128, 1707, 5395, 13666, 3046...   

                                  attention_mask  \
0     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   
1                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   
2                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   
3  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   
4           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   

                                              labels  
0  [19977, 346, 88, 20669, 3292, 202, 3573, 752, ...  
1  [139, 20, 18, 257, 771, 2426, 1129, 11, 710, 1...  
2  [37, 1229, 11, 330, 23, 2757, 23, 73, 12725, 4...  
3  [24, 636, 64, 11, 88, 423, 847, 310, 21, 19, 7...  
4                    [84, 88, 9397, 6066, 21, 67, 0]  

# Rename the keys in the data dictionary to match the expected input names
train_batched_dataset = tf.data.Dataset.from_generator(
    lambda: ({
        'cnn_input': x['en'],
        'final_model_input': x['ur'],
        'decoder_input_ids': x['input_ids'],
        'labels': x['labels']
    } for x in tokenized_train_datasets),
    output_signature={
        'cnn_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'final_model_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'decoder_input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
).padded_batch(batch_size=16, padding_values=padding_values)

# Rename the keys in the data dictionary for the validation dataset as well
val_batched_dataset = tf.data.Dataset.from_generator(
    lambda: ({
        'cnn_input': x['en'],
        'final_model_input': x['ur'],
        'decoder_input_ids': x['input_ids'],
        'labels': x['labels']
    } for x in tokenized_test_datasets),
    output_signature={
        'cnn_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'final_model_input': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'decoder_input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
).padded_batch(batch_size=16, padding_values=padding_values)

# Check the keys of tokenized_train_datasets
print("Keys of tokenized_train_datasets:", train_batched_dataset.element_spec)

# Check the keys of tokenized_test_datasets
print("Keys of tokenized_test_datasets:", val_batched_dataset.element_spec)

Keys of tokenized_train_datasets: {'cnn_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'final_model_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'decoder_input_ids': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'labels': TensorSpec(shape=(None, None), dtype=tf.int32, name=None)}
Keys of tokenized_test_datasets: {'cnn_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'final_model_input': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'decoder_input_ids': TensorSpec(shape=(None, None), dtype=tf.int32, name=None), 'labels': TensorSpec(shape=(None, None), dtype=tf.int32, name=None)}

# Print the column names of the train dataset
print("Column names of train dataset:", tokenized_train_datasets.column_names)

# Print the column names of the test dataset
print("Column names of test dataset:", tokenized_test_datasets.column_names)

Column names of train dataset: ['en', 'ur', 'input_ids', 'attention_mask', 'labels']
Column names of test dataset: ['en', 'ur', 'input_ids', 'attention_mask', 'labels']

# Check the lengths of sequences in train_df
train_sequence_lengths = train_df.apply(lambda x: len(x['input_ids']), axis=1)
print("Train Sequence Lengths:", train_sequence_lengths.unique())

# Assuming val_df is your validation DataFrame
# Check the lengths of sequences in val_df
val_sequence_lengths = tokenized_test_datasets.to_pandas().apply(lambda x: len(x['input_ids']), axis=1)
print("Validation Sequence Lengths:", val_sequence_lengths.unique())

Train Sequence Lengths: [ 14  10  15  12   8  11  23  13  28  32   6   9  33  31  36  57  41  20
  25  16  29  21   4  22  43  34  54  27  17  38   3  40  19  80  42  44
  48  53   5  18 104   7  66  26  30  24  51  45  77  47  62  37  35  92
  60  39  50  71  55  59]
Validation Sequence Lengths: [13 30 49 14 10  8 12  9 18 11 72 15 47 16  4 20 28  5 44  6 23 17  7 32
 21 26 19 29 40 42 35 27 37 24 25 34 22 31 92 48 33 38]



--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------










