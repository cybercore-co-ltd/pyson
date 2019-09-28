import json
import numpy as np
from glob import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import xxhash
import pickle
from functools import partial
#import mmcv
import numpy as np
from six.moves import map, zip
import inspect
try:
    import torch
except:
    pass

print_source = lambda x: print(inspect.getsource(x))

def lib_reload(some_module):
    import importlib
    return importlib.reload(some_module)

def get_paths(directory, input_type='png'):
    """
        Get a list of input_type paths
        params args:
        return: a list of paths
    """
    paths = glob(os.path.join(directory, '*.{}'.format(input_type)))
    assert len(paths) > 0, '\n\tDirectory:\t{}\n\tInput type:\t{} \n num of paths must be > 0'.format(
        dir, input_type)
    print('Found {} files {}'.format(len(paths), input_type))
    return paths

def read_json(path):
    '''Read a json path.
        Arguments: 
            path: string path to json 
        Returns:
             A dictionary of the json file
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    rt =  tuple(map(list, zip(*map_results)))
    rt = list(zip(*rt))
    return rt



def multi_thread(fn, array_inputs, max_workers=None, desc="Executing Pipeline", unit=" Samples", verbose=False):
    def _wraper(x):
        i, input = x
        return {i: fn(input)}
    
    array_inputs = [(i, _) for i, _ in enumerate(array_inputs)]
    if verbose:
        with tqdm(total=len(array_inputs), desc=desc, unit=unit) as progress_bar:
            outputs = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(_wraper, array_inputs):
                    outputs.update(result)
                    progress_bar.update(1)
    else:
        outputs = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_wraper, array_inputs):
                outputs.update(result)
    if verbose:
        print('Finished')
    outputs = list(outputs.values())
    return outputs
    
    
def get_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return np.argtan(dy/dx)*180


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def noralize_filenames(directory, ext='*'):
    paths = glob('{}.{}'.format(directory, ext))
    for i, path in enumerate(paths):
        base_dir, base_name = os.path.split(path)
        name, base_ext = base_name.split('.')
        new_name = '{:0>4d}.{}'.format(i, base_ext)
        new_path = os.path.join(base_dir, new_name)
        print('Rename: {} --> {}'.format(path, new_path))
        os.rename(path, new_path)

def identify(x):
    '''Return an hex digest of the input'''
    return xxhash.xxh64(pickle.dumps(x), seed=0).hexdigest()


        
        
def memoize(func):
    import xxhash
    import pickle
    import os
    from functools import wraps
    '''Cache result of function call on disk
    Support multiple positional and keyword arguments'''

    def print_status(status, func, args, kwargs):
        pass
    

    @wraps(func)
    def memoized_func(*args, **kwargs):
        cache_dir = 'cache'
        try:
            if 'hash_key' in kwargs.keys():
                import inspect                
                func_id = identify((inspect.getsource(func), kwargs['hash_key']))
            else:
                import inspect
                func_id = identify((inspect.getsource(func), args, kwargs))
            cache_path = os.path.join(cache_dir, func_id)
            
            if (os.path.exists(cache_path) and
                    not func.__name__ in os.environ and
                    not 'BUST_CACHE' in os.environ):
                return pickle.load(open(cache_path, 'rb'))
            else:
                result = func(*args, **kwargs)
                os.makedirs(cache_dir, exist_ok=True)
                pickle.dump(result, open(cache_path, 'wb'))
                return result
        except (KeyError, AttributeError, TypeError):
            return func(*args, **kwargs)
    return memoized_func


def show_df(df, path_column=None, max_col_width=-1):
    """
        Turn a DataFrame which has the image_path column into a html table
            with watchable images
        Argument:
            df: the origin dataframe
            path_column: the column name which contains the paths to images
        Return:
            HTML object, a table with images
    """
    assert path_column is not None, 'if you want to show the image then tell me which column contain the path? if not what the point to use this?'
    import pandas
    from PIL import Image
    from IPython.display import HTML
    from io import BytesIO
    import cv2
    import base64
    
    pandas.set_option('display.max_colwidth', max_col_width)

    def get_thumbnail(path):
        img = cv2.imread(path, 0)
        h,w = img.shape[:2]
        f = 48/h
        img = cv2.resize(img, (0,0), fx=f, fy=f)
        return Image.fromarray(img)

    def image_base64(im):
        if isinstance(im, str):
            im = get_thumbnail(im)
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()

    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
    
    return HTML(df.to_html(formatters={path_column: image_formatter}, escape=False))




class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.values = []

    def update(self, new_value, verbose=True):
        if isinstance(new_value, torch.Tensor):
            new_value = new_value.item()
        self.values.append(new_value)

    def __str__(self):
        s = '{}: mean:{:0.3f}, last:{:0.3f}'.format(self.name, np.mean(self.values), self.values[-1])
        return s

    def mean(self):
        return np.mean(self.values)


if __name__ == '__main__':
    path = 'sample_image/1.png'
    image = read_img(path)
    print('Test show from path')
    show(path)
    print('Test show from np image')
    show(image)

