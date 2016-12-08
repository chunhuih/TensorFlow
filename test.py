import tensorflow as tf

filename_queue = tf.train.string_input_producer(["alldata.scv"])

reader = tf.TextLineReader()

key, value = reader.read(filename_queue)

record_defaults = [[1], [1], [1], [1], [1]]

col1, col2, col3, col4, col5 = 