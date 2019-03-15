#CarLicensePlateDetector
<br/>
The project has two parts:
<ol>
    <li>A computer vision based algorithm to localize the license plate in the input image and extract lp characters</li>
    <li>A convolutional neural network (CNN) to classify extracted characters from the license plate</li>
</ol>

#Dependencies
<ul>
    <li>Python 3</li>
    <li>Tensorflow (CPU)</li>
    <li>Matplotlib</li>
    <li>Numpy</li>
    <li>tkinter</li>
    <li>Requests</li>
</ul>

#How to Use
<!--
There exist two different modes:
<ol>
    <li>Run the code to extract a char dataset to be used in training the char classifier (CNN)</li>
    <li>Run the code to train the char classifier and detect the presented LPs in the input images</li>
</ol>
###Extract Char Dataset
-->
Run the code by following command:

`python3 ./Coordinator.py -d ./InputFrames/carDetects --verbose --showimages --url 127.0.0.1 --token fake_token`

Use `--verbose` to print information about each step and `--showimages' to show the images of lp_candidate_region and extracted_char_region at each step.

#License
Licensed to Amirkabir University of Technology (AUT)

