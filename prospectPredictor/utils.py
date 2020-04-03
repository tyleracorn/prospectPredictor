import pathlib, math, os, tempfile
import typing
#for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union


def get_tmp_file(dir:Union[pathlib.Path, str]=None):
    '''Create and return a tmp filename, optionally at a specific path.
        `os.remove` when done with it.'''
    with tempfile.NamedTemporaryFile(delete=False, dir=dir) as f: return f.name

def test_cmap(cmap):
    if isinstance(cmap, str):
        if cmap.lower() == 'jet':
            raise ValueError("ahh seriously? There are so many better colormaps then Jet. Go back and try again")

def bIfAisNone(a:any, b:any)->any:
    "return `a` if `a` is not none, otherwise return `b`."
    return b if a is None else a

def printLists(listToPrint:list, maxLineLength:int=100, sep:str='||'):
    '''
    small function for printing list in a way that is a bit easier to read

    Paramaters:
    -----------
    listToPrint:list 
        a list that we want to print
    maxLineLength:int (default 100)
    sep:str (default '||')

    Example:
    --------
    >>> tmpList = ['ab', 'cd', 'ef']
    ... || ab || cd || ef ||
    '''
    startSep = sep + ' '
    varSep = ' ' + sep + ' '
    lineToPrint = startSep
    for var in listToPrint:
        if len(var) + len(lineToPrint) + len(varSep) <= maxLineLength:
            lineToPrint = lineToPrint + var + varSep
        else:
            # if count < 1: lineToPrint = lineToPrint + varSep
            print(lineToPrint)
            lineToPrint = startSep + var + varSep
    print(lineToPrint)


def printDictOfLists(printDict:dict, keys:list, maxLineLength:int=100,
                     varSep:str='||', keySep:str='\n--------'):
    '''
    small function for printing list in a way that is a bit easier to read

    Paramaters
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
        printLists(printDict[key], maxLineLength=maxLineLength, sep=varSep)
        print('\n')
