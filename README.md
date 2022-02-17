# Binary_Speaker_Classification
A Speaker Classification Project which uses a binary classification method.

# Introduction
Since this project is assumed to be used on a security door lock system, an One-vs-One structure can be quite akward.
If we use an softmax activation function in the output layer and apply a category cross entropy loss function, anything within the label of the dataset can surely be classified.

Simple example: assume we contain features of dog, cat, frog in our dataset. Train our model with this dataset. Now we user a another dog file. The expected predict result should be: dog. If we use a cat file, we should get a cat etc. No problem?
What if now I purposely use a duck file to test the model. But the we don't have class(label) that is duck. What is the result?

Probably the frog class since the sound of the frog is the most similar(least loss) to a duck. a.k.a. This kind of One-vs-One classification cannot regonize things which is not within the dataset.
(Unless, you add a forth class, assume: other, which contians all kinds of other sound of different animals.)
