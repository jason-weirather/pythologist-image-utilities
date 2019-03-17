from tifffile import TiffFile
import numpy as np
import pandas as pd
import sys
#from random import random
"""
A set of functions to help read / modify images
"""


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

def watershed_image(np_array,starting_points,valid_target_points,steps=1,border=1):
    """
    A function for expanding a set of pixels in an image from starting_points and into valid_target_points.

    Args:
        np_array (numpy.array): A 2d array of the image where comprised of integer values
        starting_points (list): a list of (x,y) tuples to begin filling out from.  the values of these points
        valid_target_points (list): a list of (x,y) tuples of valid locations to expand into
        steps (int): the number of times to execute the watershed
        border (int): the distance to remain away from the edge of the image

    Returns:
        numpy.array: the image with the watershed executed

    """
    output = np_array.copy()
    for i in range(0,steps):
        used_target_points = valid_target_points.copy()
        output,filled_points = _watershed_image_step(output,starting_points,used_target_points)
        starting_points = filled_points
        valid_target_points = list(set(valid_target_points)-set(filled_points))
    return output

def _watershed_image_step(np_array,starting_points,valid_target_points,border=1):
    #print("START WATERSHED STEP")
    mod = pd.DataFrame({'mod':[-1,0,1]})
    mod['_key'] = 1
    fullids = map_image_ids(np_array,remove_zero=False)
    starting = pd.DataFrame(starting_points,columns=['x','y']).\
        merge(fullids,on=['x','y'])
    starting['_key'] = 1
    n = starting.merge(mod,on='_key').merge(mod,on='_key')
    n['x'] = n['x'].add(n['mod_x'])
    n['y'] = n['y'].add(n['mod_y'])
    n = n.drop(columns=['mod_x','mod_y','_key'])
    targets = pd.DataFrame(valid_target_points,columns=['x','y'])
    #print("HAVE VALID TAERGETS")
    n = n.merge(targets,on=['x','y'])
    if n.shape[0] == 0 :
        return np_array.copy(), []
    #print("SHUFFLE START")
    n = n.sample(frac=1).reset_index(drop=True).\
        groupby(['x','y']).first().reset_index()
    #print("SHUFFLE END")

    
    filled = n.pivot(index='y',columns='x',values='id')
    # now handle border
    filled.iloc[0,0:border] = 0
    filled.iloc[0:border,0] = 0
    filled.iloc[-1*border:,0] = 0
    filled.iloc[0,-1*border:] = 0 
    fids = map_image_ids(filled)
    coords = set(zip(fids['x'],fids['y']))
    start1 = fullids.loc[fullids['id']!=0]
    start1 = set(zip(start1['x'],start1['y']))
    filled_coords = list(coords-start1)
    fids = fids.merge(fullids.rename(columns={'id':'oldid'}),on=['x','y'],how='right')
    fids.loc[fids['id'].isna(),'id'] = fids.loc[fids['id'].isna(),'oldid']
    filled = fids.pivot(index='y',columns='x',values='id')
    return filled, filled_coords

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

def flood_fill(image,x,y,exit_criteria,max_depth=1000,recursion=0,visited=None,border_trim=1):
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
    testedge = testedge.loc[(testedge['x']>0)&\
                             (testedge['y']>0)&\
                             (testedge['x']<image.shape[1])&\
                             (testedge['y']<image.shape[0])]
    testedge = testedge[['x','y','key']].drop_duplicates()
    testedge = full.merge(testedge,on=['x','y'],how='left')
    #testedge['edge_id'] = testedge['id']
    testedge['edge_id'] = 0
    testedge.loc[testedge['key']==1,'edge_id'] = testedge.loc[testedge['key']==1,'id']
    #print(testedge.shape)
    #print(testedge.head())

    im2 = np.array(testedge.pivot(columns='x',index='y',values='edge_id').astype(int))
    # Now lets clear the edges
    trim_distance = 2
    for y in range(0,im2.shape[0]):
            for i in range(0,trim_distance):
                im2[y][0+i] = 0
                im2[y][im2.shape[1]-1-i] = 0
    for x in range(0,im2.shape[1]):
            for i in range(0,trim_distance):
                im2[0+i][x] = 0
                im2[im2.shape[0]-1-i][x] = 0


    return im2.copy()

    #cmap['is_edge'] = cmap.apply(lambda x: _test_edge(image,x['x'],x['y'],x['id']),1)
    #edge_image = np.zeros(image.shape)
    #orig = map_image_ids(edge_image,remove_zero=False)
    #edge_image = orig[['x','y']].merge(cmap[cmap['is_edge']==True],on=['x','y'],how='left').\
    #    pivot(columns='x',index='y',values='id').fillna(0)
    #if verbose: sys.stderr.write("Finished making edge image.\n")
    #return np.array(edge_image)
