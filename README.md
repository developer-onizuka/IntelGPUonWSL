# IntelGPUonWSL

# 1. Install Intel iGPU driver
Intel HD Graphics 6xx or newer, and 28.20.100.8322 driver or newer

# 2. 
```
 conda create --name tfdml_plugin python=3.10
 conda activate tfdml_plugin
 pip install tensorflow-cpu==2.15.0
 pip install tensorflow-directml-plugin
```

# 3. 
```
 > python
Python 3.10.13 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:24:38) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.

>>> import tensorflow as tf
>>> print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
Num GPUs Available:  2

```
