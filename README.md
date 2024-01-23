# IntelGPUonWSL
Run machine learning (ML) training on their existing DirectX 12-enabled hardware by using the DirectML Plugin for TensorFlow 2.

# 1. Install Intel iGPU driver
Intel HD Graphics 6xx or newer, and [28.20.100.8322 driver or newer](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html)

# 2. Install libraries
# 2-1. Install TensorFlow and tensorflow-directml-plugin
```
 pip install tensorflow-cpu==2.15.0
 pip install tensorflow-directml-plugin
```
# 2-2. Check if Intel GPU works well
You can see the number of GPUs including iGPU.
```
$ python3
>>> import tensorflow as tf
>>> print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

# 3. Fine Tuning with Intel GPU
![FineTuning_with_IntelGPU.png](https://github.com/developer-onizuka/IntelGPUonWSL/blob/main/FineTuning_With_IntelGPU.png)

# 3-1. Important parameters
Disable all of CUDA Devices in my system which has the Nvidia Quadro P1000.
```
os.environ['CUDA_VISIBLE_DEVICE'] = '-1'
```

Intel GPU is defined as GPU#1 in my system. You can do model fitting with tf.device method.
```
with tf.device('/device:GPU:1'):
    history = model.fit(
        train_dataset,
        shuffle=True,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
```

# 3-2. Install WSL on Windows 10
```
$ wsl --install -d Ubuntu
```

# 3-3. Install some modules
```
$ apt install pip -y
$ pip install pyspark
$ pip install fastparquet
$ pip install transformers
$ pip install ipywidgets widgetsnbextension pandas-profiling
$ pip install matplotlib==3.7.3
```

# 3-4. Download and transform the amazon review data
```
$ cd /mnt/c/Users/<yourAccount>/Downloads
$ wget https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/amazon_reviews_2015.snappy.parquet
$ python3 amazon_reviews_parquet_small.py /mnt/c/Users/<yourAccount>/Downloads
```

# 3-5. Create DataSet for MachineLearning
```
$ python3 BERT-embedding-from-text_small.py /mnt/c/Users/<yourAccount>/Downloads
```

# 3-6. Fine Tuning
```
$ python3 Fine-Tuning_small.py /mnt/c/Users/<yourAccount>/Downloads
```
