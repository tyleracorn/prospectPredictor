'''create a template for the rasters that the predictor class(es) will write to'''
import rasterio
import numpy as np


class RasterTemplate():
    '''
    initialize a raster template that will be used by the predictor class for creating the prediction and distance rasters
    '''
    def __init__(self, projBounds, cellWidthX, cellHeightY, crs, rasterDTypes='float32', transform='default', driver='GTiff'):
        self.projBounds:DataFrame = projBounds
        self.cellWidthX:np.number = cellWidthX
        self.cellHeightY:np.number = cellHeightY
        self.crs = crs
        self.dtypes = rasterDTypes
        if isinstance(transform, str):
            if transform == 'default':
                minX = int(self.projBounds['minx'][0]) // self.cellWidthX * self.cellWidthX
                maxY = int(self.projBounds['maxy'][0]) // self.cellHeightY * self.cellHeightY
                self.transform = rasterio.transform.from_origin(minX, maxY, cellWidthX, cellHeightY)
            else:
                raise ValueError("currently only default transforms are supported. you can pass your own transform in a rasterio.transform format")
        else:
            self.transform = transform
        self.data = None
        #self._createEmptyRaster()
        self.driver = driver

    def initializeEmptyRaster(self):
        '''
        create an empty raster and set to self.data based on paramaters passed)
        '''
        minX = int(self.projBounds['minx'][0]) // self.cellWidthX * self.cellWidthX
        maxX = int(self.projBounds['maxx'][0]) // self.cellWidthX * self.cellWidthX
        minY = int(self.projBounds['miny'][0]) // self.cellHeightY * self.cellHeightY
        maxY = int(self.projBounds['maxy'][0]) // self.cellHeightY * self.cellHeightY

        xDim = maxX - minX
        yDim = maxY - minY
        nColsX = int(xDim / self.cellWidthX)
        nRowsY = int(yDim / self.cellHeightY)

        return np.full((nRowsY, nColsX), np.nan, dtype=self.dtypes)