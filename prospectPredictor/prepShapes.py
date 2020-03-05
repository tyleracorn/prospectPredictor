'''
class for prepping shapefiles and selecting the shapes we will use in the predictor class
'''
import math, pathlib, tempfile, os

def printLists(listToPrint, maxLineLength=120, sep='||'):
    '''
    small function for printing list in a way that is a bit easier to read

    listToPrint {list}
    maxLineLength {int}:120
    sep {str}: '||'
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


def printDictOfLists(dict, keys, maxLineLength=120, varSep='||', keySep='\n--------'):
    '''
    small function for printing list in a way that is a bit easier to read

    dict {dict}: expecting the values to be lists
    keys {list}: pass 1 or more keys in a list
    maxLineLength {int}:120
    varSep {str}: '||'
    keySep {str}: '\n--------'
    '''
    for key in keys:
        print(key, keySep)
        printLists(dict[key], maxLineLength=maxLineLength, sep=varSep)
        print('\n')


def printUnique(geodataframe, include:list=None, exclude:list=['gid', 'upid', 'geometry'], excludeNumeric:bool=True):
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


def get_tmp_file(dir:[pathlib.Path, str]=None):
    '''Create and return a tmp filename, optionally at a specific path.
        `os.remove` when done with it.'''
    with tempfile.NamedTemporaryFile(delete=False, dir=dir) as f: return f.name


def bIfAisNone(a:any, b:any)->any:
    "return `a` if `a` is not none, otherwise return `b`."
    return b if a is None else a

class PrepShapes():
    def __init__(self, data, shapeCategories:list, shapeNames:list, boundaryBuffer:int=10000):
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

    def printUnique(self, include=None, exclude=['gid', 'upid', 'geometry'], excludeNumeric=True, maxLineLength=120):
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

    def printColumns(self, maxLineLength=120):
        'print the column names in a slightly easier to read format'

        printLists(self.data.columns.tolist(), maxLineLength=maxLineLength)

    def dissolveData(self, forceDissolve:bool=False):
        '''
        dissolve the many polygons within each shape into a single multipolygon
        
        forceDissolve:bool = force running dissolve again even if you already called it in the past.
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

    def setBuffer(self, bufferSize=False):
        '''
        update the class boundary buffer variable
        '''
        if bufferSize:
            self.__dict__.update({'boundaryBuffer': bufferSize})

    def setProjectBoundary(self, buffer=True):
        '''
        this will need to be tested. trying to take
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