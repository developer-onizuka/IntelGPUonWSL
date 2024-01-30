import tensorflow as tf
import sys
import os 
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertConfig
from tensorflow.keras.utils import plot_model

args = sys.argv

if len(args)<2:
    args.append("/mnt")
    args.append("")

if len(args)<3:
    args.append("")

myDir = args[1]
precision = args[2]

##### using AMP BF16 precision for predict #####
if precision == "bf16":
    import intel_extension_for_tensorflow as itex
    print("intel_extension_for_tensorflow {}".format(itex.__version__))

    auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
    auto_mixed_precision_options.data_type = itex.BFLOAT16 

    graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options)
    graph_options.auto_mixed_precision = itex.ON

    config = itex.ConfigProto(graph_options=graph_options)
    itex.set_config(config)
    device = '/XPU:0'

elif precision == "xpu":
    device = '/XPU:0'

else:
    device = '/CPU:0'
################################################

    
model = tf.keras.models.load_model(myDir + '/tensorflow_small')

# Check its architecture
model.summary()

config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
)

transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

max_seq_length = 64

CLASSES = [1, 2, 3, 4, 5]

config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(CLASSES),
    id2label={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    label2id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
)
print(config)

import pandas as pd
import numpy as np

from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def predict(text):
    with tf.device(device):
        encode_plus_tokens = tokenizer.encode_plus(
                             text,
                             pad_to_max_length=True,
                             max_length=max_seq_length,
                             truncation=True,
                             return_tensors='tf')

        input_ids = encode_plus_tokens['input_ids']
        input_mask = encode_plus_tokens['attention_mask']

        outputs = model.predict(x=(input_ids,input_mask))

        prediction = [{"label":config.id2label[item.argmax()], \
                       "socre":item.max().item()} for item in outputs]

        return prediction[0]

#df = pd.read_parquet(myDir + "/amazon_reviews_2015_small_tail.snappy.parquet",columns=["star_rating","review_id","review_body"])
test_df = pd.read_parquet(myDir + "/amazon_reviews_2015_small_sameSample.snappy.parquet",columns=["star_rating","review_id","review_body"])


numOfSample=200
#for i in range(1,6):
#    new_df = 'df' + str(i)
#    globals()[new_df] = df[df['star_rating'] == i].sample(n=numOfSample)
#
#test_df = pd.concat([df1,df2,df3,df4,df5])

test_df.count()

#test_df.to_parquet(myDir + "/amazon_reviews_2015_small_sameSample.snappy.parquet")

y_test = test_df['review_body'].map(predict)
y_true = test_df['star_rating']

# print(type(y_true))
y_true2 = [x for x in y_true.values]

# print(type(y_test))
y_test2 = [x.get('label') for x in y_test.values]

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#print(CLASSES)
cm = confusion_matrix(y_true2, y_test2, labels=CLASSES)
cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
print(cm_df)

arrayX = []
for array in cm:
    temp = []
    # print(array)
    # sumOfValue = array.sum()
    for i in range(len(array)):
        # print(array[i])
        # temp.append(array[i]/sumOfValue)
        temp.append(array[i]/numOfSample)
    arrayX.append(temp)

# print(arrayX)
arrayX_df = pd.DataFrame(arrayX, index=CLASSES, columns=CLASSES)
print(arrayX_df)
