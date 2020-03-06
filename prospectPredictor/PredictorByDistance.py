'''predictor class for prediction based on distance to different shape files'''
import pathlib, math, os, tempfile
import rasterio
import numpy as np
import shapely

import typing
#for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union

def get_tmp_file(dir:Union[pathlib.Path, str]=None):
    '''Create and return a tmp filename, optionally at a specific path.
        `os.remove` when done with it.'''
    with tempfile.NamedTemporaryFile(delete=False, dir=dir) as f: return f.name


class PredictorByDistance():
    def __init__(self, preppedShape, rasterTemplate, architect:str='varriogram', archType:str='exponential',
                 archRange:float=10000, modelDir:Union[pathlib.Path, str]='models'):
        '''
        distance based predictor. takes the distance from 2 or more shapes units and calculates
        a prospectivity value between 0 (unlikely) and 1 (likely). uses a pseudo exponential
        varriogram weighting to calculates prospectivity weights based on max range

        Parameters
        ----------
        preppedShape: 
            an initialized prepShape() class
        rasterTemplate:
            an initialized rasterTemplate
        architect:str (default 'varriogram')
            currently only 1 weighting predictor architect is set up and that is a pseudo varriogram
        archType:str (default 'exponential')
            currently only one style of varriogram model is set up and that is the exponential varriogram
        archRange:float (default 10000)
            the range used in the varriogram calculations. sets max range for your predictor
        modelDir:Union[pathlib.Path, str] (default'models')
            path to save / load prediction and distance raster to / from
        '''
        self.shapes = preppedShape.dictOfProjShapes
        self.shapeKeys = self.shapes.keys()
        self.rasterTemplate = rasterTemplate
        self.rasterTransform = rasterTemplate.transform
        self.predictRaster = rasterTemplate.initializeEmptyRaster()
        self.distRasters = dict()
        for key in preppedShape.dictOfProjShapes.keys():
            self.distRasters.update({key: rasterTemplate.initializeEmptyRaster()})
        self.architect = architect
        self.archType = archType
        self.archRange = archRange
        if isinstance(modelDir, str): modelDir = pathlib.Path(modelDir)
        self.modelDir = modelDir

    def _testWriteablePath(self):
        '''test if you can write to path. makedir if needed'''
        if isinstance(self.modelDir, str):
            self.__dict__.update({'modelDir': pathlib.Path(self.modelDir)})
        try:
            self.modelDir.mkdir(parents=True, exist_ok=True)
            tmpF = get_tmp_file(self.modelDir)
        except OSError as e:
            raise Exception(f"{e}\nCan't write to '{self.modelDir}', set `modelDir` attribute in predictor to a full libpath path that is writable") from None
        os.remove(tmpF)

    def xy(self, row:int, col:int, offset='center'):
        '''Returns the coordinates `(x,y)` of a cell at row and col index. the pixels center is returned by default
        This function is using the transform.xy() function from rasterio
        '''
        return rasterio.transform.xy(self.rasterTransform, rows=row, cols=col, offset=offset)
    
    def updateArchitect(self, architect=False, archType=False, archRange=False):
        ''' 
        update the architect used
        current only 1 architect type and model are implemented... so don't use?!?
        '''

        if architect:
            self.__dict__.update({'architect': architect})
        if archType:
            self.__dict__.update({'archType': archType})
        if archRange:
            self.__dict__.update({'archRange': archRange})

    def distanceMatrix(self, maxRange=None, maxRangeMultiple=None):
        '''
        calculate the distance from center of each cell to the nearest boundary
        of the shape files. self.updateShapes() to change which shapes we are
        calculating distances to.
        
        Note: the varriogram calculator currently predicts 100 percent if all distances
        equal NaN so the maxRange feature here isn't implemented yet
        example

        '''
        if maxRange:
            maxRange = maxRange
        elif maxRangeMultiple:
            maxRange = self.archRange * maxRangeMultiple
        else:
            maxRange = None
        tmpKey = list(self.shapeKeys)[0]
        print(maxRange, tmpKey)
        for (idxR, idxC), _ in np.ndenumerate(self.distRasters[tmpKey]):
            tmpPoint = shapely.geometry.Point(self.xy(idxR, idxC))
            for key in self.shapeKeys:
                dist = self.shapes[key]['dataDissolved'].distance(tmpPoint)[0]
                if maxRange:
                    if dist <= maxRange:
                        self.distRasters[key][idxR][idxC] = dist
                else:
                    self.distRasters[key][idxR][idxC] = dist

    def predict(self, updateDistance:bool=True, distKwargs:Optional[dict]=None, varrioKwargs:Optional[dict]=None):
        '''
        calculates a weighted prediction using the pseudo varriogram. 
        This weights the predictions so that they drop to 0 when the range from either 
        of the shapes reaches the varriogram range.
        
        Parameters
        ----------
        updateDistance:bool (default True)
            by default it updates the distance matrix which is more time consuming then the prediction
            if you are just change range, set to False
        distKwargs:dict
            keyword arguments to pass to self.distanceMatrix
        varrioKwargs:dict
            keyword arguments to pass to self.varriogramExp
        '''
        if updateDistance:
            if distKwargs:
                self.distanceMatrix(**distKwargs)
            else:
                self.distanceMatrix()
        

        for (idxR, idxC), _ in np.ndenumerate(self.predictRaster):
            distances = list()
            for key in self.shapeKeys:
                distances.append(self.distRasters[key][idxR][idxC])
            if varrioKwargs:
                prediction = self.varriogramExp(distances, **varrioKwargs)
            else:
                prediction = self.varriogramExp(distances)
            self.predictRaster[idxR][idxC] = prediction

    def varriogramExp(self, distList:list, varrioRange:Optional[float]='default',
                      distFactor:Optional[float]=1.5, smoothFactor:Optional[float]=1):
        '''
        use a pseudo varriogram for calculating a weighted prediction. this gives you a
        smooth prediction from 1 (most likely) to 0 (least likely) based on the distance to the 
        shapes. It drops off to nearly zero once the distance to any of the shapes reaches the 
        varriogram range. 
        
        Note: works well with 2 distances... more distances will mean it will
        take a little longer to drop to near zero if the distance to only 1 shape reaches the 
        range.

        Parameters
        ----------
        varrioRange:float
            by default uses self.archRange but can change to a new range
        distFactor:float (default 1.5)
            the larger the number the steeper the change
        smoothFactor:float (default 1)
            the larger the number the shallower the change

        Example
        ------- 

        prediction calculated from 2 distances and default values:
        distFactor=1.5, smoothFactor=1

        predict = 1 * math.exp( -( (1.5 * dist1)**2 / varrioRange**2 ) -( (1.5 * dist2)**2 / varrioRange**2) )

        .. math::
            predict = 1 * math.exp( -( (1.5 * dist1)^2 / varrioRange^2 ) - ( (1.5 * dist2)^2 / varrioRange^2 ) )
        '''
        # determine if we are using the default varriogram/architect range or a new range
        if isinstance(varrioRange, str):
            if varrioRange.lower() == 'default':
                vRange = self.archRange
            else:
                raise ValueError("either pass numeric range or `default`")
        else:
            vRange = varrioRange
        # calculate the weighted prediction
        power = list()
        for d in distList:
            power.append(self.calcPower(d, vRange, distFactor))
        power = np.nansum(power)

        return smoothFactor * math.exp(power)

    def calcPower(self, dist:float, archRange:float, distFactor:float):
        '''
        return the distance factor used in varriogram exponential calculation

        Example:
        distFactor=1.5

        power = -( (1.5 * dist1)**2 / archRange**2 )
        '''
        return -((distFactor * dist)**2/archRange**2)

    def _plotSetup(self, ax, figsize, transform):
        '''
        utility function for setting up some common plotting paramaters
        '''
        if isinstance(transform, str):
            if transform.lower() == 'default': 
                transform = self.rasterTransform
            else:
                raise ValueError(f"transform:{transform} - must pass either `default` or a valid rasterio transform")

        if not ax:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        return ax, fig, transform

    def plotPrediction(self, ax=None, figsize=(10,10), cmap='viridis',
                       transform='default', cbar=True, cbarAnchor=None, cbarOrient='vertical',
                       return_cbar=False):
        '''
        simple plotting wrapper that uses rasterio plotting function to plot the prediction raster
        '''
        import rasterio.plot as rioplot
        ax, fig, transform = self._plotSetup(ax, figsize, transform)
        
        ax = rioplot.show(self.predictRaster, ax=ax, cmap=cmap, transform=transform)
        ax.set_title(f'Prediction based on distances to shapes. Varriogram Range:{self.archRange} ({self.rasterTemplate.crs.linear_units})')
        if cbar:
            import matplotlib as mpl
            cmap = mpl.cm.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cbAnchor = [0.85, 0.14, 0.02, 0.7] if not cbarAnchor else cbarAnchor 
            cax = fig.add_axes(cbAnchor)
            cb =  mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation=cbarOrient)
            cax.set_ylabel(f'Prediction 0 (least likely) to 1 (most likely)')
            
            if return_cbar: 
                return ax, cax

        return ax

    def plotDistance(self, rasterKey, ax=None, figsize=(10,10), cmap='viridis',
                     transform='default', cbar=True, cbarAnchor=None, cbarOrient='vertical',
                     return_cbar=False):
        '''
        simple plotting wrapper that uses rasterio plotting function to plot the prediction raster
        '''
        import rasterio.plot as rioplot
        ax, fig, transform = self._plotSetup(ax, figsize, transform)

        ax = rioplot.show(self.distRasters[rasterKey], ax=ax, cmap=cmap, transform=transform)
        ax.set_title(f'Distances to {rasterKey}')
        if cbar:
            import matplotlib as mpl
            cmap = mpl.cm.get_cmap(cmap)
            norm = mpl.colors.Normalize(vmin=0, vmax=int(self.distRasters[rasterKey].max()))
            cbAnchor = [0.85, 0.14, 0.02, 0.7] if not cbarAnchor else cbarAnchor 
            cax = fig.add_axes(cbAnchor)
            cb =  mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation=cbarOrient)
            cax.set_ylabel(f'Distance ({self.rasterTemplate.crs.linear_units})')
            
            if return_cbar: 
                return ax, cax

        return ax

    def saveRaster(self, file:[pathlib.Path,str]=None, bands:str='prediction'):
        '''
        save the prediction and/or distance rasters to file using the GTiff driver
        file:[Path, str] = path to save file to. Will use model directory set in class
        bands:str = options include 'all', 'prediction', or 'distances'

        '''
        if self.modelDir: self._testWriteablePath()
            
        fpath = self.modelDir.joinpath(file)
        if not fpath.suffix: fpath = fpath.with_suffix('.tiff')

        bands = bands.lower()
        targets = dict()
        if bands == 'all':
            targets.update({fpath.with_name(fpath.stem + '_pred' + fpath.suffix): self.predictRaster})
            for key in self.shapeKeys:
                targets.update({fpath.with_name(fpath.stem + '_' + key + fpath.suffix): self.distRasters[key]})
        elif bands == 'prediction':
            targets.update({fpath.with_name(fpath.stem + '_pred' + fpath.suffix): self.predictRaster})
        elif bands == 'distances':
            for key in self.shapeKeys:
                targets.update({fpath.with_name(fpath.stem + '_' + key + fpath.suffix): self.distRasters[key]})
        else:
            raise ValueError("only currently supported save options are bands = 'all', 'predictions' OR 'distances'")

        for flKey in targets.keys():
            with rasterio.open(flKey, 'w', driver='GTiff',
                               height=self.rasterTemplate.nRowsY,
                               width=self.rasterTemplate.nColsX,
                               count=1, dtype=self.rasterTemplate.dtypes,
                               crs=self.rasterTemplate.crs, transform=self.rasterTemplate.transform) as sRaster:
                sRaster.write(targets[flKey], 1)


    def loadRaster(self, path:[pathlib.Path, str], rasterType:str='prediction'):
        '''
        load a prediction raster that has been previously saved.
        '''
        if isinstance(path, str): path = pathlib.Path(path)
        mPath = self.modelDir.joinpath(path)
        rasterType = rasterType.lower()
        if rasterType =='prediction':
            possiblePaths = [mPath,
                             mPath.with_suffix('.tiff'), 
                             mPath.with_name(mPath.stem + '_pred' + mPath.suffix), 
                             mPath.with_name(mPath.stem + '_pred' + mPath.suffix).with_suffix('.tiff'),
                             path, 
                             path.with_suffix('.tiff'), 
                             path.with_name(path.stem + '_pred' + path.suffix), 
                             path.with_name(path.stem + '_pred' + path.suffix).with_suffix('.tiff')]
            idx = 0
            while idx < len(possiblePaths):
                if possiblePaths[idx].exists(): 
                    target = possiblePaths[idx]
                    break
                idx += 1
            else:
                raise ValueError(f"raster path:{path} not found")

            self.predictRaster = rasterio.open(target).read(1)
        else:
            for key in self.shapeKeys:
                shpPath = mPath.with_name(mPath.stem + '_' + key).with_suffix('.tiff')
                self.distRasters.update({key: rasterio.open(shpPath).read(1)})
