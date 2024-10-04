import sys
from importlib import reload

def import_custom(modules, relative_path = None):
    
    if relative_path is not None:
        sys.path.append(relative_path)
    for module in modules:
        try:
            exec('import ' + module)
        except ImportError:
            print('Module ' + module + ' not found. Check if the module is in the correct path.')
            break
        exec('reload(' + module + ')')
        exec('from ' + module + ' import *')
        
        
    print('All modules imported succesfully.')
    return