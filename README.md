# Estimating Meter Readings From An Image

This implements one approach to reading a 7-segment display from an image. There are many ways this *could* be done - and this is almost certainly overkill - but this one is quite fun and works nicely :)

The approach taken here is to synthetically generate a large number of images of displays showing different values and use them to learn a digit-recognition model. The [`SyntheticData.ipynb`](SyntheticData.ipynb) notebook shows some examples of this synthetic data generation process

The digit detection framework builds heavily on top of [this tutorial on object detection](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#torchvision-object-detection-finetuning-tutorial). 
The steps to **predict** the meter value from the image are:

1. Run our digit detection model to find all of the digits in the display
2. Sort the digits from left-to-right
3. Combine and convert to an actual value (given that we know the last digit will be following a decimal point)

This process can be seen in the [`EstimateMeterReading.ipynb`](EstimateMeterReading.ipynb) notebook
