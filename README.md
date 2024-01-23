# IntelGPUonWSL
Run machine learning (ML) training on their existing DirectX 12-enabled hardware by using the DirectML Plugin for TensorFlow 2.

# 1. Install Intel iGPU driver
Intel HD Graphics 6xx or newer, and [28.20.100.8322 driver or newer](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html)

# 2. Install TensorFlow and tensorflow-directml-plugin
```
 pip install tensorflow-cpu==2.15.0
 pip install tensorflow-directml-plugin
```

# 3. Important parameters
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
