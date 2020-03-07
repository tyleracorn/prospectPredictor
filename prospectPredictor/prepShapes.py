'''
class for prepping shapefiles and selecting the shapes we will use in the predictor class
'''
import pandas as pd
from .utils import *


class PrepShapes():
    def __init__(self, dataOrFileName, shapeNameColDict:Optional[dict]=None, boundaryBuffer:int=12000):
        '''
        Class for prepping shapefiles. Used to select which polygons we want to
        use in our predictor and find project boundaries based on those polygons


        Parameters
        ----------

        data: 
            shapefiles loaded into a geopandas.GeoDataFrame or the filename/path to a file to load into self.data with geopandas.read_file()
        shapeNameColDict:dict
            a dictionary tying the description of a shape to it's column. example {'ultramafic': 'rock_class'}. This is used to group polygons into categories
        boundaryBuffer:int
            a buffer used to find the project boundary. i.e. 10% bigger than your predictor range
        

        Example
        --------

        >>> InputFile = 'data/BedrockP.shp'
        >>> pShapes = PrepShapes(bedrockData)
        '''
        if isinstance(dataOrFileName, (pathlib.Path, str)):
            import geopandas as gpd
            data = gpd.read_file(dataOrFileName)
            self.data = data
        else:
            self.data=dataOrFileName
        self.shapeNameColDict=dict()
        self.projShapes=dict()
        if shapeNameColDict: 
            self.shapeNameColDict.update(shapeNameColDict)
            for key in self.shapeNameColDict.keys():
                col = self.shapeNameColDict[key]
                self.projShapes.update({key:{'cat': col,
                                             'data': self.data.loc[self.data[col] == str(key)]}
                                       })
        self._nShapes=len(self.shapeNameColDict)
        self.boundaryBuffer:int=boundaryBuffer            
        self._dissolved = False
        self._buffered = False
        self.projBounds = False                                        
                                            
    def printUnique(self, include:list=None, exclude:list=['gid', 'upid', 'geometry'],
                    excludeNumeric:bool=True, maxLineLength:int=100):
        '''
        small function to print out unique values in all columns or some columns.
        assumes you are passing it a GeoPandas dataframe so it by default will exclude some columns.

        '''
        if excludeNumeric:
            gdframe = self.data.select_dtypes(exclude='number')
        else:
            gdframe = self.data

        if include:
            catCols = [col for col in gdframe.columns.tolist() if col in include]
        else:
            catCols = [col for col in gdframe.columns.tolist() if col not in exclude]

        categoryDesc = dict()
        for key in catCols:
            categoryDesc[key] = gdframe[key].unique().tolist()

        printDictOfLists(categoryDesc, categoryDesc.keys(), maxLineLength=maxLineLength)

    def printColumns(self, maxLineLength:int=100):
        'print the column names in a slightly easier to read format'

        printLists(self.data.columns.tolist(), maxLineLength=maxLineLength)

    def plotData(self, column:str, ax=None, categorical:bool=True, cmap:str='gist_earth', legend:bool=True,  
                 figsize:tuple=(10,10), kwds:dict=None)-> "matplotlib axes":
        '''
        basic plot for shapes grouped by values in column. This is a wrapper 
        for geopandas.GeoDataFrame.plot() with common defaults.
        you can access the full plotting function at self.data.plot()
        
        Parameters
        ----------
        column: str
            name of the column to color polygons by
        ax: matplotlib.pyplot Artist (default None)
            axes on which to draw the plot
        categorical: bool (default True)
            is the data categorical data?
        cmap: str (default 'gist_earth')
            name of any colormap recognized by matplotlib
        legend: bool (default True)
            whether to plot a legend
        figsize: tuple of integers (default (10, 10))
            Size of the resulting matplotlib.figure.Figure (width, height). If axes is
            passed, then figsize is ignored.
        kwds: dict (default None)
            keyword dictionary to pass onto geopandas.GeoDataFrame.plot()

        Returns
        -------
        ax: matplotlib axes instance
        '''
        test_cmap(cmap)
        if kwds: 
            ax = self.data.plot(column=column, categorical=categorical, 
                                legend=legend, ax=ax, cmap=cmap, figsize=figsize, **kwds)
        else:
            ax = self.data.plot(column=column, categorical=categorical, 
                                legend=legend, ax=ax, cmap=cmap, figsize=figsize)
        return ax
        
    def plotShapes(self, ax=None, color:list=['red', 'orange'], legend:bool=True, legLoc:str='upper left',
                   figsize:tuple=(10,10), polyfill:bool=False, sAlpha:float=None, bAlpha:float=None, 
                   useProjBounds:bool=False, plotBuffer:bool=False, kwds:dict=None):
        '''
        basic plot wrapper that loops through the self.shapeNames and plots the shape outlines relying heavily on defaults.
        uses the geopandas.GeoDataFrame.plot() tied to self.data
        
        Parameters
        ----------
        ax: matplotlib.pyplot Artist (default None)
            axes on which to draw the plot
        color: list (default ['red', 'orange'])
            list of colors. needs to be same length as self.shapeNames
        legend: bool (default True)
            whether to plot a legend
        legLoc:str (default 'upper left')
            the location to plot the legend for the shapes
        figsize: tuple of integers (default (10, 10))
            Size of the resulting matplotlib.figure.Figure (width, height). If axes is
            passed, then figsize is ignored.
        polyfill: bool (default False)
            whether to fill in the polygon using color
        sAlpha: float (default None)
            transparency of the shapeNames polygon fills. should be between 0 (transparent) and 1 (opaque)
        bAlpha: float (default None)
            transparency of the buffer polygon fills. should be between 0 (transparent) and 1 (opaque)
        useProjBounds: bool (default False)
            if True will use self.projBounds to reset the plot axis limits
        plotBuffer: bool (default False)
            to plot buffer shapes or not to plot buffer shapes
        kwds: dict (default None)
            keyword dictionary to pass onto geopandas.GeoDataFrame.plot()

        Returns
        -------
        ax: matplotlib axes instance

        '''
        if legend: 
            from matplotlib.lines import Line2D
            from matplotlib.legend import Legend
            handles = []
            labels = []
        for c, shp in zip(color, self.shapeNameColDict.keys()):
            cpoly = c if polyfill else 'none'
            if kwds:
                ax = self.projShapes[shp]['data'].plot(categorical=True, figsize=figsize,
                                                             ax=ax, facecolor=cpoly, 
                                                             edgecolor=c, alpha=sAlpha, **kwds)
                if plotBuffer: ax = self.projShapes[shp]['buffer'].plot(ax=ax, facecolor=cpoly, 
                                                                              edgecolor=c, alpha=bAlpha, **kwds)
            else:
                ax = self.projShapes[shp]['data'].plot(categorical=True, figsize=figsize,
                                                             ax=ax, facecolor=cpoly, 
                                                             edgecolor=c, alpha=sAlpha) 
                if plotBuffer: ax = self.projShapes[shp]['buffer'].plot(ax=ax, facecolor=cpoly, 
                                                                              edgecolor=c, alpha=bAlpha)
            if legend: 
                handles.append(Line2D([], [], color=c, lw=0, marker='o', markerfacecolor='none'))
                labels.append(shp)
                
        if legend: 
            leg = Legend(ax, handles, labels, loc=legLoc, frameon=True)
            ax.add_artist(leg)

        if useProjBounds:
            ax.set_xlim(self.projBounds['minx'][0], self.projBounds['maxx'][0])
            ax.set_ylim(self.projBounds['miny'][0], self.projBounds['maxy'][0])
            
        return ax
    
    def dissolveData(self, forceDissolve:bool=False):
        '''
        dissolve the many polygons within each shape into a single multipolygon
        
        Parameters
        ----------
        forceDissolve:bool
            force running dissolve again even if you already called it in the past.
        '''
        if forceDissolve or not self._dissolved:
            for key in self.shapeNameColDict.keys():
                column = self.projShapes[key]['cat'] 
                dissolvedShape = self.projShapes[key]['data'][[column, 'geometry']].dissolve(by=column,
                                                                                             aggfunc='first',
                                                                                             as_index=False)
                self.projShapes[key].update({'dataDissolved': dissolvedShape})
            self._dissolved = True
        else:
            print('shapes already dissolved')

    def bufferData(self, buffer:int='default', addPercent=1.1):
        '''
        Create a buffered shape around the project shapes 

        Parameters
        ----------
        buffer:int
            by default it grabs the instance self.buffer variable you set on initialization.
        '''
        if isinstance(buffer, str): 
            if buffer.lower() == 'default':
                buffer = self.boundaryBuffer
            else:
                raise ValueError(f"'default' or a value are the two currently supported buffer values. You passed:{buffer}")
        if addPercent:
            buffer = buffer * 1.1
        for key in self.shapeNameColDict.keys():
            dataToBuffer = 'dataDissolved' if self._dissolved else 'data'
            self.projShapes[key].update({'buffer': self.projShapes[key][dataToBuffer].buffer(buffer)})
        if not self._buffered:
            self._buffered = True

    def setBuffer(self, bufferSize:int=False):
        '''
        update the class boundary buffer variable

        Parameters
        ----------
        bufferSize:int
            size of buffer (in crs units) to used to create a buffer around shapes of interest
        '''
        if bufferSize:
            self.__dict__.update({'boundaryBuffer': bufferSize})
    
    def setShapes(self, shapeNameColDict:dict, reset:bool=True):
        '''
        updates self.shapeNameColDict as well as self.projShapes
        if you want to just update (keep old keys not passed) set reset=False
        
        Parameters
        ----------
        
        shapeNameColDict:dict
            dictionary of category name (i.e. 'ultramafic') and the column the descriptor is found in (i.e. 'rock_class')
        reset:bool (default True)
            will do a reset of the dictionaries (self.shapeNameColDict and self.projShapes) i.e clearing them before updating
        '''
        if reset:
            self.shapeNameColDict.clear()
            self.projShapes.clear()
        
        self.shapeNameColDict.update(shapeNameColDict)
        if len(self.shapeNameColDict) > 0:
            for key in self.shapeNameColDict.keys():
                col = self.shapeNameColDict[key]
                self.projShapes.update({key:{'cat': col,
                                             'data': self.data.loc[self.data[col] == str(key)]}
                                        })

    def setProjectBoundary(self, buffer=True):
        '''
        create the smallest size project boundary based on the the shapes of interest

        Parameters
        ----------
        buffer:bool (default True)
            sets proj boundary based on buffer otherwise will use shapes (which wouldn't be that useful)
        '''
       # for key in self.dictOfProjShapes.keys():
        data = 'buffer' if buffer else 'data'
        minXList = [self.projShapes[key][data].bounds.minx.values for key in self.projShapes]
        maxXList = [self.projShapes[key][data].bounds.maxx.values for key in self.projShapes]
        minYList = [self.projShapes[key][data].bounds.miny.values for key in self.projShapes]
        maxYList = [self.projShapes[key][data].bounds.maxy.values for key in self.projShapes]

        minBounds = {'minx':max(minXList), 'miny':max(minYList), 
                     'maxx':min(maxXList), 'maxy':min(maxYList)}
        self.__dict__.update({'projBounds': pd.DataFrame(minBounds)})

