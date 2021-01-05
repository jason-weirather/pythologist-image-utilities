from tifffile import TiffFile
import numpy as np
import pandas as pd
import sys, hashlib, json
from scipy.ndimage.morphology import binary_dilation
from sklearn.neighbors import NearestNeighbors
#from random import random
"""
A set of functions to help read / modify images

"""
def compare_tiff_contents(path1,path2):
    """
    For two input tif image paths, see if they have the same layer structure and image descriptions

    Args:
        path1 (str): a path to a tif
        path2 (str): a path to a tif
    Returns:
        result (bool): True if they are the same image False if they are not
    """
    stack1 = hash_tiff_contents(path1)
    stack2 = hash_tiff_contents(path2)
    return stack1==stack2

def hash_tiff_contents(path):
    """
    For two input tif image paths, see if they have the same layer structure and image descriptions

    Args:
        path (str): a path to a tif
    Returns:
        result (bool): True if they are the same image False if they are not
    """
    stack = read_tiff_stack(path)
    stack = tuple([(hashlib.sha256(x['raw_meta']['image_description']).hexdigest(),hashlib.sha256(x['raw_image'].tostring()).hexdigest()) for x in stack])
    return hashlib.sha256(json.dumps(stack).encode('utf-8')).hexdigest()


def binary_image_dilation(np_array,steps=1):
    """
    For an input image that gets set to 0 or 1, expand the 1's by the number of steps

    Args:
        np_array (numpy.array): a 2d image
        steps (int): number of pixels to expand
    Returns:
        numpy.array: Image with that has been expanded
    """
    img = make_binary_image_array(np_array)
    img = binary_dilation(img,iterations=steps).astype(np.uint8)
    return img

def median_id_coordinates(np_array,exclude_points=None):
    """
    Locate a coordinate near the center of each object in an image

    Args:
        np_array (numpy.array): Take an image where pixels code for the IDs
        exclude_points (list): optional. a list of tuples of 'x','y' coordinates. to exclude from being possible median outputs
    Returns:
        pandas.DataFrame: DataFrame indexed by ID with a near median 'x', and median 'y' for that ID
    """
    nids = map_image_ids(np_array)
    if exclude_points is not None:
        exclude_points = pd.DataFrame(exclude_points,columns=['x','y'])
        exclude_points['exclude'] = 'Yes'
        nids = nids.merge(exclude_points,on=['x','y'],how='left')
        nids = nids.loc[nids['exclude'].isna()].drop(columns='exclude')
    # Get the median of the x dimension
    ngroup = nids.groupby('id').apply(lambda x: pd.Series({'x':list(x['x'])}))
    ngroup['median_x'] = ngroup['x'].apply(lambda x: np.quantile(x,0.5,interpolation='nearest'))
    nids = nids.merge(ngroup[['median_x']],left_on='id',right_index=True)
    # Subset to y values that fall on that x median
    nids = nids.loc[nids['x']==nids['median_x']]
    ngroup = nids.groupby('id').apply(lambda x: pd.Series({'x':list(x['x']),'y':list(x['y'])}))
    nmedian = ngroup.applymap(lambda x: np.quantile(x,0.5,interpolation='nearest'))
    return nmedian

def watershed_image(np_array,starting_points,valid_target_points,steps=1,border=1,fill_value=1,border_fill_value=0):
    """
    A function for expanding a set of pixels in an image from starting_points and into valid_target_points.

    Args:
        np_array (numpy.array): A 2d array of the image where comprised of integer values
        starting_points (list): a list of (x,y) tuples to begin filling out from.  the values of these points
        valid_target_points (list): a list of (x,y) tuples of valid locations to expand into
        steps (int): the number of times to execute the watershed
        border (int): the distance to remain away from the edge of the image
        fill_value (int): The integer value to fill the area in with
        border_fill_value (int): The value to fill the border area in with

    Returns:
        numpy.array: the image with the watershed executed

    """
    output = np_array.copy()
    if len(valid_target_points) > 0 and len(starting_points) > 0:
        nn = NearestNeighbors(n_neighbors=1,radius=steps).\
             fit(starting_points).\
             radius_neighbors(valid_target_points,radius=steps)
        for i,v in enumerate(nn[0]):
            if len(v) == 0: continue
            output[valid_target_points[i][1],valid_target_points[i][0]] = fill_value
    output = _fill_borders(output,border,fill_value=border_fill_value)
    return output

def _fill_borders(img,border_size_px,fill_value):
    if border_size_px == 0: return img.copy()
    _temp = pd.DataFrame(img.copy())
    _temp.iloc[0:,0:border_size_px] = fill_value
    _temp.iloc[0:border_size_px,0:] = fill_value
    _temp.iloc[-1*border_size_px:,0:] = fill_value
    _temp.iloc[0:,-1*border_size_px:] = fill_value
    return np.array(_temp)

def split_color_image_array(np_array):
    if len(np_array.shape) == 2: return [np_array]
    images = []
    for i in range(0,np_array.shape[2]):
        image = np.array([[y[0] for y in x] for x in np_array])
        images.append(image)
    return np.array(images)

def make_binary_image_array(np_array):
    """
    Make a binary (one channel) image from a drawn color image

    Args:
        np_array (numpy.array) a numpy array that came from a color image
    Returns:
        numpy.array: an array that is 1 where something (anything) existed vs 0 where there was nothing
    """
    np_array = np.nan_to_num(np_array)
    if len(np_array.shape) == 2: return np.array([[1 if y > 0 else 0 for y in x] for x in np_array])
    return np.array([[1 if np.nanmax([z for z in y]) > 0 else 0 for y in x] for x in np_array]).astype(np.int8)


def read_tiff_stack(filename):
    """
    Read in a tiff filestack into individual images and their metadata

    Args:
        filename (str): a path to a tiff file

    Returns:
        list: a list of dictionary entries keyed by 'raw_meta' and 'raw_image' for each image in the tiff stack
    """
    data = []
    with TiffFile(filename) as tif:
        image_stack = tif.asarray()
        for page in tif.pages:
            meta = dict((tag.name,tag.value) for tag in page.tags.values())
            data.append({'raw_meta':meta,'raw_image':np.array(page.asarray())})
    return data

def flood_fill(image,x,y,exit_criteria,max_depth=1000,recursion=0,visited=None,border_trim=0):
    """
    There is a flood_fill in scikit-image 0.15.dev0, but it is not faster than this
    for this application.  It may be good to revisit skikit's implemention if it is optimized.

    Args:
        image (numpy.array): a 2d numpy array image
        x (int): x starting coordinate
        y (int): y starting coordinate
        exit_criteria (function): a function for which to exit i.e. ``lambda x: x!=0``
        max_depth (int): a maximum recurssion depth
        recursion (int): not set by user, used to keep track of recursion depth
        visited (list): list of (x,y) tuple representing coordinates that have been visited
        border_trim (int): the size of the border to avoid on the edge
    Returns:
        numpy.array: the filled image
    """
    # return a list of coordinates we fill without visiting twice or hitting an exit condition
    if visited is None: visited = set()
    if len(visited)>=max_depth: return visited
    if recursion > 1000: return visited
    if y < 0+border_trim or y >= image.shape[0]-border_trim: return visited
    if x < 0+border_trim or x >= image.shape[1]-border_trim: return visited
    if (x,y) in visited: return visited
    if exit_criteria(image[y][x]): 
        return visited
    visited.add((x,y))
    # traverse deeper
    if (x,y+1) not in visited:
       visited = flood_fill(image,x,y+1,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    if (x+1,y) not in visited:
        visited = flood_fill(image,x+1,y,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    if (x,y-1) not in visited:
       visited = flood_fill(image,x,y-1,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    if (x-1,y) not in visited:
       visited = flood_fill(image,x-1,y,exit_criteria,max_depth=max_depth,recursion=recursion+1,visited=visited,border_trim=border_trim)
    return visited

def map_image_ids(image,remove_zero=True):
    """
    Convert an image into a list of coordinates and the id (coded by pixel integer value)

    Args:
        image (numpy.array): A numpy 2d array with the integer values representing cell IDs
        remove_zero (bool): If True (default), remove all zero pixels
    Returns:
        pandas.DataFrame: A pandas dataframe with columns shaped as <x><y><id>
    """
    nmap = pd.DataFrame(image.astype(float)).stack().reset_index().\
       rename(columns={'level_0':'y','level_1':'x',0:'id'})
    nmap.loc[~np.isfinite(nmap['id']),'id'] = 0
    if remove_zero: nmap = nmap[nmap['id']!=0].copy()
    nmap['id'] = nmap['id'].astype(int)
    return nmap[['x','y','id']]


def _test_edge(image,x,y,myid):
    for x_iter in [-1,0,1]:
        xcoord = x+x_iter
        if xcoord >= image.shape[1]-1: continue
        for y_iter in [-1,0,1]:
            ycoord = y+y_iter
            if x_iter == 0 and y_iter==0: continue
            if xcoord <= 0 or ycoord <=0: continue
            if ycoord >= image.shape[0]-1: continue
            if image[ycoord][xcoord] != myid: return True
    return False


def image_edges(image,verbose=False):
    """
    Take an image of cells where pixel intensitiy integer values represent cell ids 
    (fully filled-in) and return just the edges

    Args:
        image (numpy.array): A 2d numpy array of integers coding for cell IDs
        verbose (bool): If true output more details to stderr
    Returns:
        numpy.array: an output image of just edges
    """
    if verbose: sys.stderr.write("Making dataframe of possible neighbors.\n")
    cmap = map_image_ids(image)
    edge_image = np.zeros(image.shape)
    if verbose: sys.stderr.write("Testing for edge.\n")
    # cmap
    #print(cmap.head())
    mod = pd.DataFrame({'mod':[-1,0,1]})
    mod['key'] = 1
    mod = mod.merge(mod,on='key')
    mod['keep'] = mod.apply(lambda x: 1 if abs(x['mod_x'])+abs(x['mod_y'])==1 else 0,1)
    mod = mod[mod['keep']==1].copy()

    full = map_image_ids(image,remove_zero=False)
    attempt = full.rename(columns={'id':'next_id',
                                  'x':'mod_x',
                                  'y':'mod_y'})
    testedge = cmap.copy()
    testedge['key'] = 1
    testedge = testedge.merge(mod,on='key')
    testedge['mod_x'] = testedge['x'].add(testedge['mod_x'])
    testedge['mod_y'] = testedge['y'].add(testedge['mod_y'])
    testedge = testedge.merge(attempt,on=['mod_x','mod_y']).query('id!=next_id')
    testedge = testedge.loc[(testedge['x']>=0)&\
                             (testedge['y']>=0)&\
                             (testedge['x']<=image.shape[1])&\
                             (testedge['y']<=image.shape[0])]
    testedge = testedge[['x','y','key']].drop_duplicates()
    testedge = full.merge(testedge,on=['x','y'],how='left')
    #testedge['edge_id'] = testedge['id']
    testedge['edge_id'] = 0
    testedge.loc[testedge['key']==1,'edge_id'] = testedge.loc[testedge['key']==1,'id']
    #print(testedge.shape)
    #print(testedge.head())

    im2 = np.array(testedge.pivot(columns='x',index='y',values='edge_id').astype(int))
    # Now lets clear the edges
    trim_distance = 0
    #for y in range(0,im2.shape[0]):
    #        for i in range(0,trim_distance):
    #            im2[y][0+i] = 0
    #            im2[y][im2.shape[1]-1-i] = 0
    #for x in range(0,im2.shape[1]):
    #        for i in range(0,trim_distance):
    #            im2[0+i][x] = 0
    #            im2[im2.shape[0]-1-i][x] = 0


    return im2.copy()

def binary_image_list_to_indexed_image(image_list,overwrite=False,priority=None):
    """
    For a list of binary images (integer coded as 0 or 1), combine them together into an indexed image..
    Where 0 means undefined, 1 means the first image in the list, 2 means the second image in the list etc

    Args:
        image_list (list): a list of binary 2d images
        overwrite (bool): if False require images to be mutually exclusive, if true write according to priority order
        priority (list): List of indecies of image to write as order to write
    Returns:
        numpy.array: Image that is a 2d image integer coded
    """
    def as_binary(myimg):
        return myimg.astype(bool).astype(int)
    if priority is None: priority = list(range(0,len(image_list)))
    tot = np.zeros(image_list[priority[0]].shape)
    accumulate = np.zeros(image_list[priority[0]].shape)
    for i in priority:
        #print(i)
        current = -1*(as_binary(accumulate)-1)
        index = i+1 # we need a 1-indexed layer
        img = image_list[i]
        if img.shape != tot.shape: raise ValueError("images need to be the same shape")
        tot += img
        contents = set(np.unique(img))
        if (contents - set([0,1])) != set():
            raise ValueError("Only binary images can be in input stack\n")
        accumulate = (index*(img&current))+accumulate
    if np.unique(tot).max() > 1 and overwrite is False: 
        raise ValueError("The layers are not mutually exclusive. Use ovewrite True to allow this.")
    return accumulate

def color_indexed_image(indexed_image,color_key):
    """
    For an image that is indexed so that integers represent different parts,
    color it in based on a key.

    Args:
        indexed_image (numpy.array): a 2d image of integers
        color_key (dict): a dictionary keyed by the integer index pointing to a tuple representing the rgb color, or rgba.  These are on decimal scale 0-1.
    Returns:
        numpy.array: Image that is l,w,3 for RGB or l,w,4 for RGBA
    """
    w = len(list(color_key.values())[0])
    oimg = np.ndarray(tuple(list(indexed_image.shape)+[w]))
    for i in range(0,indexed_image.shape[0]):
        for j in range(0,indexed_image.shape[1]):
            oimg[i,j,:]=color_key[indexed_image[i,j]]
    return oimg
