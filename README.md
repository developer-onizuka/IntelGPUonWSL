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

>>> tf.debugging.set_log_device_placement(True)

>>> # Create some tensors
>>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

2024-01-21 21:16:09.979205: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-01-21 21:16:09.979797: I tensorflow/c/logging.cc:34] DirectML: creating device on adapter 0 (NVIDIA Quadro P400)
2024-01-21 21:16:10.119047: I tensorflow/c/logging.cc:34] Successfully opened dynamic library Kernel32.dll
2024-01-21 21:16:10.120176: I tensorflow/c/logging.cc:34] DirectML: creating device on adapter 1 (Intel(R) HD Graphics 530)
2024-01-21 21:16:10.160002: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-01-21 21:16:10.160108: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 1, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-01-21 21:16:10.161189: W tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc:28] Overriding allow_growth setting because force_memory_growth was requested by the device.
2024-01-21 21:16:10.161565: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 16181 MB memory) -> physical PluggableDevice (device: 0, name: DML, pci bus id: <undefined>)
2024-01-21 21:16:10.166058: W tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc:28] Overriding allow_growth setting because force_memory_growth was requested by the device.
2024-01-21 21:16:10.166165: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 14598 MB memory) -> physical PluggableDevice (device: 1, name: DML, pci bus id: <undefined>)
input: (_Arg): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.179741: I tensorflow/core/common_runtime/placer.cc:114] input: (_Arg): /job:localhost/replica:0/task:0/device:GPU:0
_EagerConst: (_EagerConst): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.183307: I tensorflow/core/common_runtime/placer.cc:114] _EagerConst: (_EagerConst): /job:localhost/replica:0/task:0/device:GPU:0
output_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.184209: I tensorflow/core/common_runtime/placer.cc:114] output_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.190165: I tensorflow/core/common_runtime/eager/execute.cc:1419] Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
>>> b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
2024-01-21 21:16:10.191498: I tensorflow/core/common_runtime/eager/execute.cc:1419] Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
>>> c = tf.matmul(a, b)
a: (_Arg): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.194636: I tensorflow/core/common_runtime/placer.cc:114] a: (_Arg): /job:localhost/replica:0/task:0/device:GPU:0
b: (_Arg): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.197805: I tensorflow/core/common_runtime/placer.cc:114] b: (_Arg): /job:localhost/replica:0/task:0/device:GPU:0
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.198661: I tensorflow/core/common_runtime/placer.cc:114] MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
product_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.199512: I tensorflow/core/common_runtime/placer.cc:114] product_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:0
2024-01-21 21:16:10.200890: I tensorflow/core/common_runtime/eager/execute.cc:1419] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0

>>> print(c)

tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```

# 4. tf.tensor with Intel GPU - HD Graphics 530
```
> python.exe .\test.py
2024-01-21 21:29:49.919093: I tensorflow/c/logging.cc:34] Successfully opened dynamic library C:\Users\Hisayuki\miniconda3\envs\tfdml_plugin\lib\site-packages\tensorflow-plugins/directml/directml.d6f03b303ac3c4f2eeb8ca631688c9757b361310.dll
2024-01-21 21:29:49.920097: I tensorflow/c/logging.cc:34] Successfully opened dynamic library dxgi.dll
2024-01-21 21:29:49.925996: I tensorflow/c/logging.cc:34] Successfully opened dynamic library d3d12.dll
2024-01-21 21:29:50.786148: I tensorflow/c/logging.cc:34] DirectML device enumeration: found 2 compatible adapters.
Num GPUs Available:  2
2024-01-21 21:29:51.239531: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-01-21 21:29:51.240799: I tensorflow/c/logging.cc:34] DirectML: creating device on adapter 0 (NVIDIA Quadro P400)
2024-01-21 21:29:51.364459: I tensorflow/c/logging.cc:34] Successfully opened dynamic library Kernel32.dll
2024-01-21 21:29:51.365622: I tensorflow/c/logging.cc:34] DirectML: creating device on adapter 1 (Intel(R) HD Graphics 530)
2024-01-21 21:29:51.404513: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-01-21 21:29:51.404627: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 1, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-01-21 21:29:51.405637: W tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc:28] Overriding allow_growth setting because force_memory_growth was requested by the device.
2024-01-21 21:29:51.411715: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 16181 MB memory) -> physical PluggableDevice (device: 0, name: DML, pci bus id: <undefined>)
2024-01-21 21:29:51.412871: W tensorflow/core/common_runtime/pluggable_device/pluggable_device_bfc_allocator.cc:28] Overriding allow_growth setting because force_memory_growth was requested by the device.
2024-01-21 21:29:51.412960: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 14598 MB memory) -> physical PluggableDevice (device: 1, name: DML, pci bus id: <undefined>)
input: (_Arg): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.426317: I tensorflow/core/common_runtime/placer.cc:114] input: (_Arg): /job:localhost/replica:0/task:0/device:GPU:1
_EagerConst: (_EagerConst): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.427208: I tensorflow/core/common_runtime/placer.cc:114] _EagerConst: (_EagerConst): /job:localhost/replica:0/task:0/device:GPU:1
output_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.428233: I tensorflow/core/common_runtime/placer.cc:114] output_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.436242: I tensorflow/core/common_runtime/eager/execute.cc:1419] Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.436697: I tensorflow/core/common_runtime/eager/execute.cc:1419] Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:1
a: (_Arg): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.437684: I tensorflow/core/common_runtime/placer.cc:114] a: (_Arg): /job:localhost/replica:0/task:0/device:GPU:1
b: (_Arg): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.438568: I tensorflow/core/common_runtime/placer.cc:114] b: (_Arg): /job:localhost/replica:0/task:0/device:GPU:1
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.439290: I tensorflow/core/common_runtime/placer.cc:114] MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:1
product_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.440142: I tensorflow/core/common_runtime/placer.cc:114] product_RetVal: (_Retval): /job:localhost/replica:0/task:0/device:GPU:1
2024-01-21 21:29:51.441435: I tensorflow/core/common_runtime/eager/execute.cc:1419] Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:1
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```
