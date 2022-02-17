# Binary_Speaker_Classification
A Speaker Classification Project which uses a binary classification method.

# Introduction
Since this project is assumed to be used on a security door lock system, an One-vs-One structure can be quite akward.
If we use an softmax activation function in the output layer and apply a category cross entropy loss function, anything within the label of the dataset can surely be classified.

Simple example: 
Assume we contain features of dog, cat, frog in our dataset. Train our model with this dataset. Now we user a another dog file. The expected predict result should be: dog. If we use a cat file, we should get a cat etc. No problem?
What if now I purposely use a duck file to test the model. But the we don't have class(label) that is duck. What is the result?

Probably the frog class since the sound of the frog is the most similar(least loss) to a duck. a.k.a. This kind of One-vs-One classification cannot regonize things which is not within the dataset.
(Unless, you add a forth class, assume: other, which contians all kinds of other sounds of different animals.)

# Method
In a security system, You definitely want to recognize the outsiders. So in this project, I try do change it to a One-vs-All(or One-vs-Rest) method.
We can use multiple binary classification model to identify outsiders and also find the correct class.

Back to the dog, cat, frog example:
Now we supposedly have three binary model for each classes(labels). First we use a dog file, it should work like:
  Is it a dog? Yes -> Is it a cat? No -> Is it a frog? No. ANS: dog.
Now if use the duck file, it should work likes this:
  Is it a dog? No -> Is it a cat? No -> Is it a frog? No. ANS: Belongs to none of the classes.

# Result
Show out a binary classifier can be quite succes and suitable for my security door lock system. 

However, for now, I have to manually change the cureent user variable to make the correct One-vs-All dataset format. Might use the os.path.walk function in python to improve the automation flow.
