'''create a template for the rasters that the predictor class(es) will write to'''
import rasterio
import numpy as np

from .utils import *

class RasterTemplate():
    def __init__(self, projBounds, cellWidthX:Union[int, float], cellHeightY:Union[int, float], 
                 crs, rasterDTypes:Union[str, np.dtype]='float32', transform='default',
                 driver:str='GTiff'):
        '''
        initialize a raster template that will be used by the predictor class for 
        creating the prediction and distance rasters

        Parameters
        ----------
        projBounds:DataFrame
            projBoundary from prepShapes.projBounds. pd.Dataframe with ['minx', 'miny', 'maxx', 'maxy'] columns
        cellWidthX:Union[int, float]
            how wide the cells will be (in crs units)
        cellHeightY:Union[int, float]
            how high the cells will be (in crs units)
        crs:Union[str, crs]
            the cordinate reference system that will be used in the transform
        rasterDTypes:Union[str, np.dtype]
            dataType for the raster (numpy.array())
        transform:Union[str, Transform]='default'
            the transform used to transform raster index to coordinates
        driver:str='GTiff'
            what rasterIO driver to use for saving and loading data. Only tested with 'GTiff'

        '''
        self.projBounds = projBounds
        self.cellWidthX = cellWidthX
        self.cellHeightY = cellHeightY
        self.minX = int(self.projBounds['minx'][0]) // self.cellWidthX * self.cellWidthX
        self.maxX = int(self.projBounds['maxx'][0]) // self.cellWidthX * self.cellWidthX
        self.minY = int(self.projBounds['miny'][0]) // self.cellHeightY * self.cellHeightY
        self.maxY = int(self.projBounds['maxy'][0]) // self.cellHeightY * self.cellHeightY
        self.xDim = self.maxX - self.minX
        self.yDim = self.maxY - self.minY
        self.nColsX = int(self.xDim / self.cellWidthX)
        self.nRowsY = int(self.yDim / self.cellHeightY)
        if isinstance(crs, str):
            self.crs = rasterio.crs.CRS.from_string(crs)
        else:
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
        returns an empty (with np.nan) raster based on this raster template
        '''

        return np.full((self.nRowsY, self.nColsX), np.nan, dtype=self.dtypes)
