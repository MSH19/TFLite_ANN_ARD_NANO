# Creating ANN model and converting it to TFLite 

The regression_NN_1.ipynb creates a neural network model that predicts the output of a sine function. Then, it trains the model and evalutes it aginst a validation set.       

The trained model is then converted to TFLite models (float and int8). These converted models are exported as header files to be included in the Arduino sketch for using them on Arduino NANA 33 BLE module.    
