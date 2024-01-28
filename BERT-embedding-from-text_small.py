import tensorflow as tf
import collections
import json
import os
import pandas as pd
import csv
from transformers import DistilBertTokenizer
import sys

args = sys.argv

# myDir = "/mnt/c/Users/xxxxx/Downloads/"
myDir = args[1]
myParquet = myDir + "/amazon_reviews_2015.snappy.parquet"

max_seq_length = 64

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

REVIEW_BODY_COLUMN = "review_body"
REVIEW_ID_COLUMN = "review_id"

LABEL_COLUMN = "star_rating"
LABEL_VALUES = [1, 2, 3, 4, 5]

label_map = {}
for (i, label) in enumerate(LABEL_VALUES):
    label_map[label] = i


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, review_id, date, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.review_id = review_id
        self.date = date
        self.label = label

class Input(object):
    def __init__(self, text, review_id, date, label=None):
        self.text = text
        self.review_id = review_id
        self.date = date
        self.label = label

def convert_input(the_input, max_seq_length):
    tokens = tokenizer.tokenize(the_input.text)
    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')

    encode_plus_tokens = tokenizer.encode_plus(
        the_input.text,
        pad_to_max_length=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True
    )
                                                    
    input_ids = encode_plus_tokens["input_ids"]

    input_mask = encode_plus_tokens["attention_mask"]

    segment_ids = [0] * max_seq_length

    label_id = label_map[the_input.label]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        review_id=the_input.review_id,
        date=the_input.date,
        label=the_input.label,
    )

    return features

def transform_inputs_to_tfrecord(inputs, output_file, max_seq_length):
    records = []
    tf_record_writer = tf.io.TFRecordWriter(output_file)

    for (input_idx, the_input) in enumerate(inputs):
        if input_idx % 10000 == 0:
            print("Writing input {} of {}\n".format(input_idx, len(inputs)))

        features = convert_input(the_input, max_seq_length)

        all_features = collections.OrderedDict()

        all_features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_ids))
        all_features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_mask))
        all_features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.segment_ids))
        all_features["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features.label_id]))

        tf_record = tf.train.Example(features=tf.train.Features(feature=all_features))
        tf_record_writer.write(tf_record.SerializeToString())

        records.append(
            {
                "input_ids": features.input_ids,
                "input_mask": features.input_mask,
                "segment_ids": features.segment_ids,
                "label_id": features.label_id,
                "review_id": the_input.review_id,
                "date": the_input.date,
                "label": features.label,
            }
        )

    tf_record_writer.close()

    return records

from datetime import datetime
from time import strftime

timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
print(timestamp)

df = pd.read_parquet(myDir + "/amazon_reviews_2015_small.snappy.parquet",columns=["star_rating","review_id","review_body"])

for i in range(1,6):
   print("star_rating: ", i)
   print(df[df['star_rating'] == i].count())
   print('-------------------------')

for i in range(1,6):
   new_df = 'df' + str(i)
   globals()[new_df] = df[df['star_rating'] == i].sample(n=20000)

df_temp = pd.concat([df1,df2,df3,df4,df5])
print(df_temp.count())

df_shaffle = df_temp.sample(100000)
df_head = df_shaffle.head(80000)
df_tail = df_shaffle.tail(20000)

df_head.to_parquet(myDir + "/amazon_reviews_2015_small_head.snappy.parquet")
df_tail.to_parquet(myDir + "/amazon_reviews_2015_small_tail.snappy.parquet")

train_df = df_head.sample(n=80000)
validation_df = df_tail.sample(n=20000)

train_inputs = train_df.apply(
    lambda x: Input(label=x[LABEL_COLUMN], text=x[REVIEW_BODY_COLUMN], review_id=x[REVIEW_ID_COLUMN], date=timestamp),
    axis=1,
)

validation_inputs = validation_df.apply(
    lambda x: Input(label=x[LABEL_COLUMN], text=x[REVIEW_BODY_COLUMN], review_id=x[REVIEW_ID_COLUMN], date=timestamp),
    axis=1,
)

train_output_file = myDir + "/train_data_small.tfrecord"
validation_output_file = myDir + "/validation_data_small.tfrecord"

train_records = transform_inputs_to_tfrecord(train_inputs, train_output_file, max_seq_length)
validation_records = transform_inputs_to_tfrecord(validation_inputs, validation_output_file, max_seq_length)

