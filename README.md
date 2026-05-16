# My GPT implementation from Scratch

This work is the implementation of GPT from scratch, helper.py contians all the model components, model.py has the test code to make sure the model is behaving how it is supposed to do, and lastly, on train.py, data loader and the training loop is contained.

The loss starts at around 10, and goes down to sub 3 at the end of the training.

To test, pytorch environment is required with the modules to install at requirements.txt.

To start training, use the code

> python train.py

To make changes to the config, edit the lines in train.py, from line 39 to line 45.