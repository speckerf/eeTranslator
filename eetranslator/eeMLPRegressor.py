import ee
import sys
import io
from typing import List
from loguru import logger

from sklearn.neural_network import MLPRegressor

class eeMLPRegressor:
    def __init__(self, model: MLPRegressor, prediction_name: str = 'prediction') -> None:
        self.model = model
        self.prediction_name = prediction_name
        if(prediction_name == 'prediction'):
            logger.warning(f"Using default prediction name: {prediction_name}. Consider using the true trait name in the Constructor. ")
        # if predicting on an array
        self.ee_array_weights = [ee.Array(weights.tolist()) for weights in self.model.coefs_]
        self.ee_array_biases = [ee.Array(biases.tolist()).reshape([1,-1]) for biases in self.model.intercepts_]
        # if predicting on an image
        self.ee_image_weights = [ee.Image(array) for array in self.ee_array_weights]
        self.ee_image_biases = [ee.Image(array) for array in self.ee_array_biases]

        if(self.model.out_activation_ != 'identity'):
            raise NotImplementedError
        
    def __repr__(self) -> str:
        # print method
        # Capture the output of the print method of MLPRegressor
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        print(self.model)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        string = "eeTranslator.eeMLPRegressor:"
        # Concatenate the custom string to the beginning of the captured output
        combined_output = string + "\n" + output.strip() # Strip removes the newline added by the print
        return combined_output 
        
    def _apply_activation_function(self, x: ee.Image | ee.Array, function: str) -> ee.Image | ee.Array:
        if(function == 'tanh'):
            return x.tanh()
        elif(function == 'identity'):
            return x
        elif(function == 'softmax'):
            raise NotImplementedError(f"Activation function: {function} not implemented in eeMLPRegressor.")
        elif(function == 'logistic' or function == 'sigmoid'):
            return x.multiply(-1).exp().add(1).pow(-1) # 1 / (1 + exp(-x))
        elif(function.lower() == 'relu'):
            return x.gt(0).multiply(x) # max(0, x)
        else:
            raise ValueError
    
    def _forward_pass_array(self, ee_X: ee.Array) -> ee.Array:
        # this method is supposed to work with a array where the rows correspond to the number of rows, while the column cooresponds to the number of bands
        # n_samples, _ = np.array(ee_X.getInfo()).shape
        x = ee_X # dim: (n_samples, b_bands)
        for i in range(self.model.n_layers_ - 1):
            x = x.matrixMultiply(self.ee_array_weights[i])
            x = x.add(self.ee_array_biases[i].repeat(axis = 0, copies = n_samples))
            if i != self.model.n_layers_ - 2:
                x = self._apply_activation_function(x, self.model.activation)
        # need to add output activation sk_model.out_activation was not identity
        # apply output activation
        x = self._apply_activation_function(x, self.model.out_activation_)
        return x
    
    def _forward_pass_image(self, ee_X: ee.Image) -> ee.Image:
        # # convert input image to arrayImage where each pixel holds an array of shape (1, n_bands)
        # array_X_1D = ee_X.toArray() # dim: (n_bands)
        # array_X_2D = array_X_1D.toArray(1) # dim: (n_bands, 1)
        # array_X_2D = array_X_2D.arrayTranspose() # dim: (1, n_bands)
        x = ee_X.toArray().toArray(1).arrayTranspose() # dim: (1, n_bands)
        
        for i in range(self.model.n_layers_ - 1):
            x = x.matrixMultiply(self.ee_image_weights[i]) # 1st iteration: dim (1, n_bands) x (n_bands, n_nodes)
            x = x.add(self.ee_image_biases[i])  # dim (1, n_nodes)
            
            if i != self.model.n_layers_ - 2:
                x = self._apply_activation_function(x, self.model.activation) # dim (1, n_nodes)
        
        # apply output activation
        x = self._apply_activation_function(x, self.model.out_activation_) # dim (1, n_nodes)
        # convert back to image with single band with output trait value
        output_image = x.arrayProject([0]).arrayFlatten([[self.prediction_name]])
        return output_image
    
    def classify(self, image: ee.Image | ee.Array) -> ee.Image:
        if isinstance(image, ee.Image):
            return self._forward_pass_image(image)
        elif isinstance(image, ee.Array):
            array = image
            return self._forward_pass_array(array)
        else:
            raise TypeError
