'''create a template for the rasters that the predictor class(es) will write to'''
import rasterio
import numpy as np


class RasterTemplate():
    def __init__(self, projBounds, cellWidthX, cellHeightY, crs, 
                 rasterDTypes='float32', transform='default', driver='GTiff'):
        '''
        initialize a raster template that will be used by the predictor class for creating the prediction and distance rasters
        '''
        self.projBounds:pd.DataFrame = projBounds
        self.cellWidthX:np.number = cellWidthX
        self.cellHeightY:np.number = cellHeightY
        self.minX = int(self.projBounds['minx'][0]) // self.cellWidthX * self.cellWidthX
        self.maxX = int(self.projBounds['maxx'][0]) // self.cellWidthX * self.cellWidthX
        self.minY = int(self.projBounds['miny'][0]) // self.cellHeightY * self.cellHeightY
        self.maxY = int(self.projBounds['maxy'][0]) // self.cellHeightY * self.cellHeightY
        self.xDim = self.maxX - self.minX
        self.yDim = self.maxY - self.minY
        self.nColsX = int(self.xDim / self.cellWidthX)
        self.nRowsY = int(self.yDim / self.cellHeightY)
        self.crs = crs
        self.dtypes = rasterDTypes
        if isinstance(transform, str):
            if transform == 'default':
                self.transform = rasterio.transform.from_origin(self.minX, 
                                                                self.maxY, 
                                                                self.cellWidthX, 
                                                                self.cellHeightY)
            else:
                raise ValueError("currently only default transforms are supported. you can pass your own transform in a rasterio.transform format")
        else:
            self.transform = transform
        self.driver = driver

    def initializeEmptyRaster(self):
        '''
        create an empty raster and set to self.data based on paramaters passed)
        '''

        return np.full((self.nRowsY, self.nColsX), np.nan, dtype=self.dtypes)