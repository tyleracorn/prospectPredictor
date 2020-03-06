'''
class for prepping shapefiles and selecting the shapes we will use in the predictor class
'''
import math, pathlib, tempfile, os
import pandas as pd

def printLists(listToPrint:list, maxLineLength:int=100, sep:str='||'):
    '''
    small function for printing list in a way that is a bit easier to read

    Paramaters:
    -----------
    listToPrint:list 
        a list that we want to print
    maxLineLength:int (default 100)
    sep:str (default '||')

    example:
    >>> tmpList = ['ab', 'cd', 'ef']
    ... || ab || cd || ef ||
    '''
    startSep = sep + ' '
    varSep = ' ' + sep + ' '
    lineToPrint = startSep

    for var in listToPrint:
        if len(var) + len(lineToPrint) + len(sep) + 2 <= maxLineLength:
            lineToPrint = lineToPrint + var + varSep
        else:
            print(lineToPrint)
            lineToPrint = startSep + var
    print(lineToPrint)


def printDictOfLists(dict:dict, keys:list, maxLineLength:int=100,
                     varSep:str='||', keySep:str='\n--------'):
    '''
    small function for printing list in a way that is a bit easier to read

    Paramaters:
    -----------
    dict: dict
        dictionary where the values are a bunch of lists to pring
    keys:list 
        the list of keys you want to select from the dictionary to print
    maxLineLength:int (default 100)
        maximum line length to print
    varSep:str (default '||')
        seperator to use between values in the lists
    keySep:str (default '\n--------')
        seperator to use between key and the lists that will be printed
    '''
    for key in keys:
        print(key, keySep)
        printLists(dict[key], maxLineLength=maxLineLength, sep=varSep)
        print('\n')


def printUnique(geodataframe, include:list=None, exclude:list=['gid', 'upid', 'geometry'],
                excludeNumeric:bool=True):
    '''
    small function to print out unique values in all columns or some columns.
    assumes you are passing it a GeoPandas dataframe so it will exclude some columns.

    '''
    if excludeNumeric:
        gdframe = geodataframe.select_dtypes(exclude='number')
    else:
        gdframe = geodataframe

    if include:
        catCols = [col for col in gdframe.columns.tolist() if col in include]
    else:
        catCols = [col for col in gdframe.columns.tolist() if col not in exclude]

    categoryDesc = dict()
    for key in catCols:
        categoryDesc[key] = gdframe[key].unique().tolist()

    printDictOfLists(categoryDesc, categoryDesc.keys())


def get_tmp_file(dir:[pathlib.Path, str]=None)->"tempFile name":
    '''Create and return a tmp filename, optionally at a specific path.
        `os.remove` when done with it.'''
    with tempfile.NamedTemporaryFile(delete=False, dir=dir) as f: return f.name


def bIfAisNone(a:any, b:any)->any:
    "return `a` if `a` is not none, otherwise return `b`."
    return b if a is None else a

class PrepShapes():
    def __init__(self, data, shapeCategories:list, shapeNames:list, boundaryBuffer:int=12000):
        '''
        Class for prepping shapefiles. Used to select which polygons we want to
        use in our predictor and find project boundaries based on those polygons

        Parameters
        ----------
        data: 
            shapefiles loaded into a geopandas.GeoDataFrame
        shapeCategories:list
            the column names (categories) which have the rocktype/unit
            descriptor that we will use for selecting project polygons
        shapeNames:list
            the actual identifier used for the rocktype/unit i.e. "sandstone"
        boundaryBuffer:int
            a buffer used to find the project boundary. i.e. 10% bigger than your predictor range
        
        Example:
        --------
        InputFile = 'data/BedrockP.shp'
        bedrockData = gpd.read_file(InputFile)
        pShapes = PrepShapes(bedrockData)
        '''
        self.data=data
        if len(shapeCategories) > 1:
            if isinstance(shapeCategories, str):
                shapeCategories=list(shapeCategories)
            else:
                if len(shapeCategories) != len(shapeNames):
                    raise ValueError("shapeCategories must by a single cat or same size as shapeNames")
        self.shapeCategories:list=shapeCategories
        self.shapeNames:list=shapeNames
        self._nShapes=len(shapeNames)
        self.boundaryBuffer:int=boundaryBuffer
        self.dictOfProjShapes=dict()
        if len(self.shapeCategories) > 1:
            for cat, key in zip(self.shapeCategories, self.shapeNames):
                self.dictOfProjShapes[key] = {'cat': cat,
                                              'data': self.data.loc[self.data[cat] == key]}
                # self.dictProjShapes[key] = self.data.loc[self.data[cat] == key]
        else:
            for key in shapeNames:
                self.dictOfProjShapes[key] = {'cat': self.shapeCategories,
                                              'data': self.data.loc[self.data[self.shapeCategories] == key]}
        self._dissolved = False
        self._buffered = False
        self.projBounds = False

    def __setstate__(self, k): return getattr(self)

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
        for c, shp in zip(color, self.shapeNames):
            cpoly = c if polyfill else 'none'
            if kwds:
                ax = self.dictOfProjShapes[shp]['data'].plot(categorical=True, figsize=figsize,
                                                             ax=ax, facecolor=cpoly, 
                                                             edgecolor=c, alpha=sAlpha, **kwds)
                if plotBuffer: ax = self.dictOfProjShapes[shp]['buffer'].plot(ax=ax, facecolor=cpoly, 
                                                                              edgecolor=c, alpha=bAlpha, **kwds)
            else:
                ax = self.dictOfProjShapes[shp]['data'].plot(categorical=True, figsize=figsize,
                                                             ax=ax, facecolor=cpoly, 
                                                             edgecolor=c, alpha=sAlpha) 
                if plotBuffer: ax = self.dictOfProjShapes[shp]['buffer'].plot(ax=ax, facecolor=cpoly, 
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
            for key in self.shapeNames:
                column = self.dictOfProjShapes[key]['cat'] 
                dissolvedShape = self.dictOfProjShapes[key]['data'][[column, 'geometry']].dissolve(by=column,
                                                                                                   aggfunc='first',
                                                                                                   as_index=False)
                self.dictOfProjShapes[key].update({'dataDissolved': dissolvedShape})
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
        for key in self.shapeNames:
            dataToBuffer = 'dataDissolved' if self._dissolved else 'data'
            self.dictOfProjShapes[key].update({'buffer': self.dictOfProjShapes[key][dataToBuffer].buffer(buffer)})
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
        minXList = [self.dictOfProjShapes[key][data].bounds.minx.values for key in self.dictOfProjShapes]
        maxXList = [self.dictOfProjShapes[key][data].bounds.maxx.values for key in self.dictOfProjShapes]
        minYList = [self.dictOfProjShapes[key][data].bounds.miny.values for key in self.dictOfProjShapes]
        maxYList = [self.dictOfProjShapes[key][data].bounds.maxy.values for key in self.dictOfProjShapes]

        minBounds = {'minx':max(minXList), 'miny':max(minYList), 
             'maxx':min(maxXList), 'maxy':min(maxYList)}
        self.__dict__.update({'projBounds': pd.DataFrame(minBounds)})

