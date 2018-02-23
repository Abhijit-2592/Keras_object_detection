# Keras_object_detection

A repository containing object detection architecture for Keras.

As of now contains only Faster RCNN architecture implementation:

[Faster_RCNN_API](https://github.com/Abhijit-2592/Keras_object_detection/tree/master/Faster_RCNN_API): Faster RCNN implementation in keras. The work is majorly adopted from [yhenon](https://github.com/yhenon/keras-frcnn). I have added new features like validation, four step alternating training,tensorboard support, easy implementation of other feature extractor architecture etc.

[resource_utilization.py](./resource_utilization.py): Handy script to monitor utilization of GPU and CPU while training. Works only for **NVIDIA GPU**. Run this script from the terminal only
Usage: `python resource_utilization -i 1`.This runs the script indefinitely and refreshes it every 1 second
