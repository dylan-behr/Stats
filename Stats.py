import numpy as np 
from scipy.interpolate import interp1d

# removes outliers from 2D data set x and y. Threshold standard deviations and window for sampling set by 'ns' and 'shoulder' respectively
def remove_outliers(x1,y1,ns = 1.5, width = 15):
    '''Removes outliers from data set x and y. 
    Inputs: 
        x1, y1 : data set arrays
        ns : Threshold standard deviations 
        width : window for sampling 
    Outputs:
        x, y : cleaned arrays
    '''
    x, y = np.copy(x1), np.copy(y1)
    shoulder = int((width)/2)
    ns = 1.5
    means = []
    stds = []
    for i in range(shoulder,len(y1)-shoulder,1):
        subsamp = y1[i-shoulder:i + shoulder + 1]
        std = subsamp.std()
        mean = subsamp.mean()
        if y1[i] > mean + ns*std or y1[i] < mean - ns*std:
            y[i] = 0
            x[i] = 0
            

    y = y[y != 0]
    x = x[x != 0]
    
    return x, y


# rebins data x, y into bins of set 'width' in data points -> most appropriate for uniformly spaced data
def rebin(x,y,width):
    '''rebins data x, y into bins of set 'width' in data points -> most appropriate for uniformly spaced data
    Inputs:
        x, y : data arrays
        width : bin width
    Outputs:
        new_x, new_y : rebinned arrays
    '''
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    if width == 1:
        return x, y
    else:
        shoulder = int((width)/2)
        mask = (1/(2*shoulder + 1))*np.ones(2*shoulder + 1)
        if width%2 == 0:
            mask[0], mask[-1] = mask[0]/2, mask[-1]/2
        new_y = np.empty(len(x) - 2*shoulder)
        new_x = x[shoulder:-shoulder]
        for i in range(len(x))[shoulder:-shoulder]:
            rawsamp = y[i - shoulder: i + shoulder + 1]
            new_y[i - shoulder] = sum(mask*rawsamp)
        return new_x,new_y
    
# interpolate and rebins 2D data  
def smooth(x, y, xrange = 0, npoints = 300, width = 9):
    '''Interpolates and rebins D data to give more continuity.
    Inputs:
        x, y : input arrays
        xrange : range of x values over which to interpolate, list in form [xmin, xmax]
        npoints: number of x points to interpolate over
        width: bin width of rebinning procedure
    Outputs:
        xnew, ynew : smoothed arrays
        '''
    if xrange == 0:
        xdif = x[-1] - x[0]
        xrange = [x[0]+ xdif/100, x[-1] - xdif/100]
    xint = np.linspace(xrange[0], xrange[-1], npoints)
    f = interp1d(x, y)
    yint = f(xint)
    xnew, ynew = rebin(xint, yint, width)
    return xnew, ynew

# find minimum/maximum coordinates of curve/data using 'smooth()' function to produce continuous line, within bound set by 'xrange' parameter
def find_sp(x, y, xrange = 0, npoints = 300, width = 9, mm = 'min', gradient = False):
    '''Finds minimum/maximum coordinates of curve/data using 'smooth()' function to produce continuous line.
    Inputs:
        x, y : Input arrays
        xrange : range of xvalues over which to interpolate, list in form [xmin, xmax]
        npoints: number of xpoints over which to interpolate
        width : bin width for rebinning procedure
        mm : minimum of maximum with options {'min', 'max'}
    Outputs:
        xm, ym : coordinates of min/max
'''
    xs, ys = smooth(x, y, xrange, npoints, width)
    if mm in ['min', 'minimum']:
        if gradient:
            arg = np.abs(np.gradient(ys, xs)[np.gradient(np.gradient(ys, xs), xs)>0]).argmin()
            xs, ys = xs[np.gradient(np.gradient(ys, xs), xs)>0], ys[np.gradient(np.gradient(ys, xs), xs)>0]
        else:
            arg = ys.argmin()
    else:
        if gradient:
            arg = np.abs(np.gradient(ys, xs)[np.gradient(np.gradient(ys, xs), xs)<0]).argmin()
            xs, ys = xs[np.gradient(np.gradient(ys, xs), xs)<0], ys[np.gradient(np.gradient(ys, xs), xs)<0]
        else:
            arg = ys.argmax()
    xm, ym = xs[arg], ys[arg]
    return xm, ym

# find coordinates of maximal descent/ascent in curve in analagous method to 'find_sp()'
def find_max_grad(x, y, xrange = 0, npoints = 300, width = 9, trend = 'd'):
    '''Finds coordinates extremal gradient of curve/data using 'smooth()' function to produce continuous line.
    Inputs:
        x, y : Input arrays
        xrange : range of xvalues over which to interpolate, list in form [xmin, xmax]
        npoints: number of xpoints over which to interpolate
        width : bin width for rebinning procedure
        trend : positive or negative gradient with options {'d'/'down', 'u'/'up'}
    Outputs:
        xm, ym : coordinates of min/max gradient
'''
    xs, ys = smooth(x, y, xrange, npoints, width)
    if trend in ['d', 'down']:
        arg = np.gradient(ys, xs).argmin()
    else:
        arg = np.gradient(ys, xs).argmax()
    xm, ym = xs[arg], ys[arg]
    return xm, ym