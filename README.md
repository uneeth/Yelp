Multi-class classification is a popular research problem that has many real-world applications. This project focuses on Multi-class image classification on real time images given by Yelp Corporation.

Developed a Convoluted Neural Network model on 206949 images to classify indoor, outdoor, food, drink and menu images.

The dataset has been downloaded from the Yelp dataset challenge website https://www.yelp.com/dataset/challenge.

The Python library-Keras was used to build the Convoluted Neural Network. Several pre-processing techniques such as rotate, flip, rescale etc were used. Grid search for parameter tuning was used to get better results.

Language Used: Python

Libraries Used:

h5py: For storing the image data.

simplejson: For processing the input json file.

keras: For preprocessing of image data and applying deep learning methodologies.

PIL (Python Image Library): For processing Images in Python.

sklearn: For splitting data set, applying grid search and Stratified-K fold.

Numpy & Pandas for processing data.

Evaluation: To evaluate the models and understand the results, Loss and Score metrics were used.

Results: The overall model loss was 20%, giving an effective accuracy of around 80%.
