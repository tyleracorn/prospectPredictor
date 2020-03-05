'''predictor class for prediction based on distance to different shape files'''
import pathlib, math, os, tempfile
import rasterio
import numpy as np
import shapely

def get_tmp_file(dir:[pathlib.Path, str]=None):
    '''Create and return a tmp filename, optionally at a specific path.
        `os.remove` when done with it.'''
    with tempfile.NamedTemporaryFile(delete=False, dir=dir) as f: return f.name


class PredictorByDistance():
    def __init__(self, preppedShape, rasterTemplate, architect='varriogram', archType='exponential', archRange=10000, modelDir='models'):
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
        '''Returns the coordinates `(x,y)` of a cell at row and col. the pixels center is returned by default
        This functino is using the transform.xy() function from rasterio
        '''
        return rasterio.transform.xy(self.rasterTransform, rows=row, cols=col, offset=offset)
    
    def updateArchitect(self, architect=False, archType=False, archRange=False):

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

    def predict(self, updateDistance:bool=True, distKwargs=None, varrioKwargs=None):
        '''
        updates distance matrix for each shape using self.distanceMatrix()
        Then calculated a weighted prediction using the pseudo varriogram. 
        This weights the predictions so that they drop to 0 when the range from either 
        of the shapes reaches the varriogram range.
        
        distKwargs:dict() = keyword arguments to pass to self.distanceMatrix
        varrioKwargs:dict() = keyword arguments to pass to self.varriogramExp
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

    def varriogramExp(self, distList, varrioRange='default', distFactor=1.5, smoothFactor=1):
        '''
        use a pseudo varriogram for calculating a weighted prediction. this gives you a
        smooth prediction from 1 (most likely) to 0 (least likely) based on the distance to the 
        shapes. It drops off to nearly zero once the distance to any of the shapes reaches the 
        varriogram range. 
        
        Note: works well with 2 distances... more distances will mean it will
        take a little longer to drop to near zero if the distance to only 1 shape reaches the 
        range.

        varrioRange:[int, float]= by default uses self.archRange but can change to a new range.
        distFactor:[int, float]=1.5 - the larger the number the steeper the change
        smoothFactor:[int, float]=1 - the larger the number the shallower the change

        prediction calculated from 2 distances and default values:
        distFactor=1.5, smoothFactor=1

        predict = 1 * math.exp( -((1.5 * dist1)**2/varrioRange**2) -((1.5 * dist2)**2/varrioRange**2) )


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
        #print(distList)
        #print(vRange)
        #print(distFactor)
        for d in distList:
            #print(d, vRange, distFactor)
            power.append(self.calcPower(d, vRange, distFactor))
        power = np.nansum(power)

        return smoothFactor * math.exp(power)

    def calcPower(self, dist, archRange, distFactor):
        '''
        return the distance factor used in varriogram exponential calculation
        '''
        return -((distFactor * dist)**2/archRange**2)

    def saveRaster(self, file:[pathlib.Path,str]=None, bands:str='prediction'):
        '''
        save the prediction and/or distance rasters to file using the GTiff driver
        file:[Path, str] = path to save file to. Will use model directory set in class
        bands:str = options include 'all', 'prediction', or 'distances'

        '''
        if self.modelDir: self._testWriteablePath()
            
        if isinstance(file, [pathlib.Path, str]):
            target = self.modelDir/file 
        else:
            file
            
        bands = bands.lower()
        targets = dict()
        if bands == 'all':
            targets = {'pred_'/target: self.predictRaster}
            for key in self.shapeKeys:
                targets[key/'_'/target] = self.distRasters[key]
        elif bands == 'prediction':
            targets = {'pred_'/target: self.predictRaster}
        elif bands == 'distances':
            targets = dict()
            for key in self.shapeKeys:
                targets[key/'_'/target] = self.distRasters[key]
        else:
            raise ValueError("only currently supported save options are bands = 'all', 'predictions' OR 'distances'")

        for flKey in targets.keys():
            with rasterio.open(flKey, 'w', driver='GTiff',
                               height=self.rasterTemplate.cellHeightY,
                               width=self.rasterTemplate.cellWidthX,
                               count=1, dtype=self.rasterTemplate.dtypes,
                               crs=self.rasterTemplate.crs, transform=self.rasterTemplate.transform) as sRaster:
                sRaster.write(targets[flKey], 1)


    def loadRaster(self, path:[pathlib.Path, str], rasterType:str='prediction'):
        '''
        load a prediction raster that has been previously saved.
        '''
        
        if isinstance(path, str): pathlib.Path(path)
        possiblePaths = [path, path.with_suffix('tiff'), path.with_name('pred_'/path.name), 
                         path.with_name('pred_'/path.name).with_suffix('tiff')]
        idx = 0
        while idx < len(possiblePaths):
            if possiblePaths[idx].exists(): 
                target = possiblePaths[idx]
                break
            idx += 1
        else:
            raise ValueError(f"raster path:{path} not found")

        rasterType = rasterType.lower()
        if rasterType =='prediction':
            self.predictRaster = rasterio.open(target).read(1)
        else:
            raise ValueError("haven't implemented reading in the distance raster(s) yet")