import tensorflow as tf

# Path to the TensorBoard event file
event_file_path = 'runs\cartrack\events.out.tfevents.1732220671.RasmusOthar.47552.0'

# Use TFRecordDataset to parse the file
dataset = tf.data.TFRecordDataset(event_file_path)

# Iterate through the dataset
for raw_record in dataset:
    event = tf.compat.v1.Event.FromString(raw_record.numpy())
    for value in event.summary.value:
        print("Tag:", value.tag)  # Log the tag
        print("Raw Tensor:", value.tensor)  # Log raw tensor data

