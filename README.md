Vanishing Point(VP) Locations via Expectation-Maximisation Algorithm
==============

The algorithm applies the Expectation-Maximisation algorithm to estimate the location of vanishing points in an image, and assign each pixel to one of the vanishing points at the same time. The input images are assumed to satisfy the Manhattan world assumption.
# Requirements
- opencv
- numpy
- matplotlib
- scipy
- argparse

# Data detail
- RGB building images where the scenes fulfill the Manhattan assumption
- The camera intrinsic properties which include focal length, principle point and size of the sensor pixels

# Objective
- Find the vanishing point (VP) locations in these input images
- Find the assignment of each pixel to the vanishing points 



# Implementation detail
Below are some examples showing how to run the <code>main.py</code> on sample images located in this repository.
<code>$ python main.py --input-image ../images/building1.jpg --output-image ../output/building1.jpg --VP-image ../output/building1_VP.jpg</code>
![Input screenshot](/images/building1.jpg?raw=true)
![VP screenshot](/output/building1_VP.jpg?raw=true)
![Assignment screenshot](/output/building1.jpg?raw=true)