import numpy as np
from scipy import signal
from scipy.interpolate import make_interp_spline

def get_layer_boundary(idx, gap, min_thick):
    """
    Get contiguous layer boundary indices.
    ----------
    idx: np.array(int)
        Indices of points.
    gap: int
        If a gap larger than this exists, they are separated into different layers
    min_thick: int
        Minimum thickness of a layer.

    Returns
    ----------
    layer_start: np.ndarray(nLayers)
        Start index for each layer.
    layer_end: np.ndarray(nLayers)
        End index for each layer.
    """

    # Split idx into layers
    Layers = np.split(idx, np.where(np.diff(idx) > gap)[0]+1)
    nLayers = len(Layers)

    # Create layer_start, layer_end arrays
    layer_start = np.full(nLayers, -99, dtype=int)
    layer_end = np.full(nLayers, -99, dtype=int)

    if nLayers > 0:
        # Loop over each layer
        for iLayer in range(0, nLayers):

            # Calculate layer thickness
            zb = Layers[iLayer][0]
            zt = Layers[iLayer][-1]
            dz = zt - zb
            # A layer must be thicker than min_thick here:
            if (dz > min_thick):
                layer_start[iLayer] = zb
                layer_end[iLayer] = zt
        
        # Exclude negative (i.e., thickness <= min_thick)
        layer_start = layer_start[layer_start >= 0]
        layer_end = layer_end[layer_end >= 0]

    return layer_start, layer_end

def get_growth_rate1(x_in, y_in, duration, Wn):
    idx = y_in > 0
    x = x_in[idx]
    y = y_in[idx]
    
    # Set up number of interpolated samples
    ns_interp = (duration*4).astype(int)

    # Perform spline fitting interpolation
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), ns_interp)
    Y_ = X_Y_Spline(X_)

    # Perform low-pass filter
    b, a = signal.butter(1, Wn, analog=False)
    Y_s = signal.filtfilt(b, a, Y_)

    # Calculate area growth/decay rate (km2/hour)
    dY = np.diff(Y_s) / np.diff(X_)
    
    return (X_, Y_, Y_s, dY)

def get_growth_rate2(x_in, y_in, duration, Wn):
    idx = y_in > 0
    x = x_in[idx]
    y = y_in[idx]
    
    # Set up number of interpolated samples
    ns_interp = (duration*4).astype(int)
    
    # Perform low-pass filter
    b, a = signal.butter(1, Wn, analog=False)
    y_s = signal.filtfilt(b, a, y)

    # Perform spline fitting interpolation
    X_Y_Spline = make_interp_spline(x, y_s)
    X_ = np.linspace(x.min(), x.max(), ns_interp)
    Y_ = X_Y_Spline(X_)

    # Calculate area growth/decay rate (km2/hour)
    dY = np.diff(Y_) / np.diff(X_)
    
    return (X_, Y_, y_s, dY)

def define_growth_stage(X, dY, min_rate_percent, gap=1, min_dur=0):
    
    growth_flag = np.zeros(len(dY), dtype=int)
    
    # Get the percentile for positive area change rate
    # dY_thres = np.quantile(dY[dY > 0], min_rate_percent)
    
    # Get the percentile for absolute area change rate
    dY_thres = np.quantile(np.abs(dY), min_rate_percent)
    
    # Get positive dY indices
    idx_p = np.where(dY > dY_thres)[0]
    # Get negative dY indices
    idx_n = np.where(dY < dY_thres)[0]
    
    # Proceed if there is sufficient sample to define layers
    if (len(idx_p) > min_dur) & (len(idx_n) > min_dur):
        # Get growth period indices
        gidx_s, gidx_e = get_layer_boundary(idx_p, gap, min_dur)
        g_dur = gidx_e - gidx_s    
        # Get decay period indices
        didx_s, didx_e = get_layer_boundary(idx_n, gap, min_dur)
        d_dur = didx_e - didx_s

        # print('Growth:', gidx_s)
        # print('Decay:', didx_s)
        # Combine growth and decayse period start/end indices, and sort them
        idx_s = np.sort(np.concatenate((gidx_s, didx_s)))
        idx_e = np.sort(np.concatenate((gidx_e, didx_e)))
        # print('Combine:', idx_s)
        # Find growth indices in the combine index array
        junk, g1, junk = np.intersect1d(idx_s, gidx_s, return_indices=True)
        junk, d1, junk = np.intersect1d(idx_s, didx_s, return_indices=True)
        # print('Match growth index:', g1)
        # print('Match decay index:', d1)
        # Set growth period to 1, decay period to -1
        stages = np.zeros(len(idx_s), dtype=int)
        stages[g1] = 1
        stages[d1] = -1
        # print('Stages:', stages)

        # Proceed if first stage is growth
        if stages[0] == 1:
            # Set periods of first stage to 1
            growth_flag[idx_s[0]:idx_e[0]] = 1

            # Compute stage differences
            stage_diff = np.diff(stages)
            # print(stage_diff)
            # Find indices of consecutive growth/decay stages (diff==0)
            idx_consec = np.where(stage_diff == 0)[0]
            if len(idx_consec) > 0:
                # Loop over each consecutive stage
                for p in range(len(idx_consec)):
                    idxp = idx_consec[p]
                    # For the first stage
                    if idxp == 0:
                        # If it is a growth stage, set the next stage flag as growth
                        if stages[idxp] == 1:
                            growth_flag[idx_s[idxp+1]:idx_e[idxp+1]] = 1
                    # For subsequent stages
                    else:
                        # If both previous and current stage are growth, 
                        # set the next stage flag as growth
                        if (stages[idxp-1] == 1) & (stages[idxp] == 1):
                            growth_flag[idx_s[idxp+1]:idx_e[idxp+1]] = 1
    
    return growth_flag, dY_thres