import ee
from typing import List
from loguru import logger

from sklearn.preprocessing import StandardScaler

class eeStandardScaler:
    def __init__(self, scaler: StandardScaler, feature_names: List[str] | str | None = None):
        self.scaler_ = scaler    
        self.mean_ = scaler.mean_
        self.scale_ = scaler.scale_
        self.ee_mean_ = ee.Array(self.mean_.tolist()) # dim: should be 1D: (n_bands)
        self.ee_scale_ = ee.Array(self.scale_.tolist()) # dim: should be 1D: (n_bands) 
        self.feature_names_ = feature_names
        self.n_features_in_ = scaler.n_features_in_
        self.n_samples_seen_ = scaler.n_samples_seen_
        
        if(self.feature_names_ is None):
            logger.warning(f"No feature_names provided; they will be set to: {None}")
        else:
            if not len(self.feature_names_) == len(self.mean_):
                logger.error(f"Length of feature_names: {len(self.feature_names_)} does not match length of mean: {len(self.mean_)}")
                raise ValueError

    def transform_imageCollection(self, image: ee.ImageCollection) -> ee.ImageCollection:
        raise NotImplementedError

    def transform_image(self, image: ee.Image) -> ee.Image:   
        
        band_names = image.bandNames()
        
        image = image.toArray() # dim: (n_bands)
        
        image = image.subtract(self.ee_mean_) # dim: (n_bands)  
        image = image.divide(self.ee_scale_) # dim: (n_bands)
        image = image.arrayFlatten([band_names]) # convert back to image with original band names
        return image    
    

    def inverse_transform(self, image: ee.Image) -> ee.Image:
        # TODO: Implement for multiple columns, so far only one column can be backtansformed. 
        raise NotImplementedError
        image = image.multiply(self.ee_scale_.get([0]))
        image = image.add(self.ee_mean_.get([0]))
        try: 
            image.getInfo()
        except:
            raise ValueError
        return image
    
    def inverse_transform_column(self, image: ee.Image, column: str) -> ee.Image:
        if not column in self.feature_names_:
            raise ValueError(f"Column {column} not in feature names: {self.feature_names_}")
        
        # get index of column in feature_names
        column_index = self.feature_names_.index(column)
        
        image = image.multiply(self.ee_scale_.get([column_index]))
        image = image.add(self.ee_mean_.get([column_index]))

        return image
