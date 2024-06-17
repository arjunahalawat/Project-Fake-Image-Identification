# Project:- "Fake Images Identification"  

Nowadays, fake images are becoming very common and is an important problem in the computer vision field and hence to avoid these types of false representations, which are leading to privacy threats and frauds, identification of such fake images is very important.

Convolutional Neural Network (CNN), is a state-of-the-art image classifier model primarily used for the purpose of fake image identifications. CNN involves training the model network on fake and real image datasets to identify the fake images by recognizing and extracting patterns in the images which can be used to differentiate the real and fake images. CNN primarily uses the concept of layers for the purpose of classification of dataset images. It works on multiple layers such as the input layers, convolutional layers for applying filters, pooling layers which down samples the image for easy computations, dense layers and finally the output layer. 

Convolutional Neural Network based model can be efficiently and effectively implemented using python TensorFlow. TensorFlow is an open-source library for machine learning and is commonly used as a platform to create and train ML models, prepare and process data using provided tools and to implement many more ML techniques. CNN can be implemented efficiently using TensorFlow as it provides various inbuilt functions and pre-existing models to create convolutional base to add sequential layers and enables the use of  Keras, which is an high level API for TensorFlow.  

This project involves a machine learning based training model for differentiating and predicting the fake and real images using CNN algorithm. The step wise proposed methodology is:

•	Collection of a large dataset consisting of both fake and real images in order to train the model and answer the research questions.

•	Preparing the data by processing the dataset of images by standardizing shape and size for easy extraction of patterns.

•	Selecting a machine learning model algorithm for fake images detection, such as CNN, which can be effectively used for training the model for image classifications as fake or real.

•	Defining the architecture of the model using the selected CNN algorithm by selecting a platform (such as TensorFlow) for proper implementation.

•	Construct the CNN convolutional layer for extracting the features from the images.

•	Apply the activation function, to simplify the dataset by removing unwanted features, such as Rectified Linear Unit (ReLU).

•	Downsample the size using pooling layer to make the computations easy and modify the output layers.

•	Predict whether the image is a match or not as output by using flattening process i.e. converting it to fully connected network by first converting the data into single column neural input.

•	Apply the SoftMax function to get a categorical distribution so that the output appears effectively.

•	Testing and deploying the trained model for results and accuracy by testing a dataset on the trained model. 

This project successfully proposed a model for identification of fake images using an approach based on convolutional neural networks and the results of this project and of the studies done so far in the field of fake image identifications using machine learning models shows that CNN algorithm is a very effective and efficient way to train a model for the purpose of fake image identifications by recognizing the patterns and extracting the features from the images.
