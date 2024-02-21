import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import pandas as pd


def Perror(n, full=False, down=False):
    '''
    Calculate the asymmetric Poisson error, using Eqn 7
    and Eqn 12 in Gehrels 1986 ApJ, 3030, 336
    Parameters
    ----------
    n
    full
    Returns
    -------
    '''

    err_up = err_dn = np.sqrt(n + 0.75) + 1.0 # this is the default behavior for N=0

    xn = np.where((n > 0))[0]
    if np.size(xn) > 0:
        err_dn[xn] = np.abs(n[xn] * (1.-1./(9. * n[xn])-1./(3.*np.sqrt(n[xn])))**3.-n[xn])
        err_up[xn] = n[xn] + np.sqrt(n[xn] + 0.75) + 1.0 - n[xn]
    # else:
    #     err_up = np.sqrt(n + 0.75) + 1.0
    #     err_dn = err_up
    #     # err_up = err_dn = np.nan

    if full is True:
        return err_dn, err_up
    else:
        if down is True:
            return err_dn
        else:
            return err_up


def FINDflare(flux, error, N1=3, N2=1, N3=3,
              avg_std=False, std_window=7,
              returnbinary=False, debug=False):
    '''
    The algorithm for local changes due to flares defined by
    S. W. Chang et al. (2015), Eqn. 3a-d
    http://arxiv.org/abs/1510.01005
    Note: these equations originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.
    Note: this algorithm originally ran over sections without "changes" as
    defined by Change Point Analysis. May have serious problems for data
    with dramatic starspot activity. If possible, remove starspot first!
    
    Parameters
    ----------
    flux : numpy array
        data to search over
    error : numpy array
        errors corresponding to data.
    N1 : int, optional
        Coefficient from original paper (Default is 3)
        How many times above the stddev is required.
    N2 : int, optional
        Coefficient from original paper (Default is 1)
        How many times above the stddev and uncertainty is required
    N3 : int, optional
        Coefficient from original paper (Default is 3)
        The number of consecutive points required to flag as a flare
    avg_std : bool, optional
        Should the "sigma" in this data be computed by the median of
        the rolling().std()? (Default is False)
        (Not part of original algorithm)
    std_window : float, optional
        If avg_std=True, how big of a window should it use?
        (Default is 25 data points)
        (Not part of original algorithm)
    returnbinary : bool, optional
        Should code return the start and stop indicies of flares (default,
        set to False) or a binary array where 1=flares (set to True)
        (Not part of original algorithm)
        
    Returns
    -------
    
    '''

    med_i = np.nanmedian(flux)

    if debug is True:
        print("DEBUG: med_i = {}".format(med_i))

    if avg_std is False:
        sig_i = np.nanstd(flux) # just the stddev of the window
    else:
        # take the average of the rolling stddev in the window.
        # better for windows w/ significant starspots being removed
        sig_i = np.nanmedian(pd.Series(flux).rolling(std_window, center=True).std())
    if debug is True:
        print("DEBUG: sig_i = ".format(sig_i))

    ca = flux - med_i
    cb = np.abs(flux - med_i) / sig_i
    cc = np.abs(flux - med_i - error) / sig_i

    if debug is True:
        print("DEBUG: N0={}, N1={}, N2={}".format(sum(ca>0),sum(cb>N1),sum(cc>N2)))

    # pass cuts from Eqns 3a,b,c
    ctmp = np.where((ca > 0) & (cb > N1) & (cc > N2))

    cindx = np.zeros_like(flux)
    cindx[ctmp] = 1

    # Need to find cumulative number of points that pass "ctmp"
    # Count in reverse!
    ConM = np.zeros_like(flux)
    # this requires a full pass thru the data -> bottleneck
    for k in range(2, len(flux)):
        ConM[-k] = cindx[-k] * (ConM[-(k-1)] + cindx[-k])

    # these only defined between dl[i] and dr[i]
    # find flare start where values in ConM switch from 0 to >=N3
    istart_i = np.where((ConM[1:] >= N3) &
                        (ConM[0:-1] - ConM[1:] < 0))[0] + 1

    # use the value of ConM to determine how many points away stop is
    istop_i = istart_i + (ConM[istart_i] - 1)

    istart_i = np.array(istart_i, dtype='int')
    istop_i = np.array(istop_i, dtype='int')

    if returnbinary is False:
        return istart_i, istop_i
    else:
        bin_out = np.zeros_like(flux, dtype='int')
        for k in range(len(istart_i)):
            bin_out[istart_i[k]:istop_i[k]+1] = 1
        return bin_out

    
def iflare(x, y, yerr, win, npass=5, dtlim=0.1, dx=1, N1=3, N2=1, N3=3, returnsmooth=True):
    '''
    This does npass iterations of SAVGOL filter, run FINDflare, creates a Mask array 
    of possible Flares. You'll need to parse the final output to get individual 
    start/stops for flares.
    
    returns Flare Mask array, same length as x,y,yerr
    '''
    # start w/ no flares
    FM = np.zeros_like(y)
    
    for k in range(npass):
        # select all points w/o a possible flare
        ok = np.where((FM == 0))[0]

        # use SAVGOL filter for now, b/c is fast/easy
        # goal could be to use starspot GP (like Spencer did)
        yhat = savgol_filter(y[ok], win, 3)
        
        # keep FINDflare params fixed
        FM_k = FINDflare(y[ok] - yhat, yerr[ok], N1=N1, N2=N2, N3=N3, returnbinary=True)

        # find gaps
        xk1 = np.append(x[ok], np.repeat(x[ok].max(), dx))
        # finds the point before (left of) a gap
        bd1 = np.array(((xk1[dx:] - xk1[:-dx]) <= dtlim), dtype=int)
        
        # mask out any "flares" near the gaps (left and right)
        for j in range(sum(bd1 < 1)):            
            FM_k[np.abs(x[ok] - x[ok][np.where((bd1 < 1))[0]][j]) <= dtlim] = 0
            FM_k[np.abs(x[ok] - x[ok][1+np.where((bd1 < 1))[0]][j]) <= dtlim] = 0

        # mask out any flares at the start/end
        FM_k[np.where((np.abs(x[ok]-np.nanmin(x[ok])) <= dtlim))[0]] = 0
        FM_k[np.where((np.abs(x[ok]-np.nanmax(x[ok])) <= dtlim))[0]] = 0


        # add new flares in
        FM[ok] = FM[ok] + FM_k
        
    if returnsmooth:
        return FM, x[ok], yhat
    else:
        # return array of [0,1]'s for flares
        return FM


def flaresplit(FM, dlim=2):
    ''' 
    given an array of [0,1] for flares, find all 
    individual events that are separated by dlim
    
    return start/stop indicies
    '''
    fl = np.where((FM > 0))[0]

    dfl = np.diff(fl)
    
    i1 = np.where((dfl > dlim))[0]
    i0 = i1 + 1

    i1 = np.append(fl[i1], fl[-1])
    i0 = fl[np.append(0, i0)]

    return i0, i1


def EquivDur(x,y,yerr, xhat,yhat, istart, istop):
    '''
    compute the equiv duration for a given flare (need to wrap in a loop!)
    '''
    
    durk = x[istop] - x[istart]
    c1 = np.where((xhat >= (x[istart] - (durk*1.5))) & 
                  (xhat <= (x[istart] - (durk*0.5)))
                 )[0]
    c2 = np.where((xhat <= (x[istop] + (durk*2.))) & 
                  (xhat >= (x[istop] + (durk)))
                 )[0]

    if len(c1) < 2:
        c1a = np.argmin(np.abs(xhat - x[istart]))
        c1 = np.array([c1a-2, c1a-1])
    if len(c2) < 2:
        c2a = np.argmin(np.abs(xhat - x[istop]))
        c2 = np.array([c2a+1, c2a+2])


    fok = np.where((x >= (x[istart] - (durk*1.5))) & 
                   (x <= (x[istop] + (durk*2.))))[0]
    fok2 = np.where((x >= (x[istart] - (durk*0.5))) & 
                    (x <= (x[istop] + (durk))))[0]
    
    cs = CubicSpline(np.append(xhat[c1], xhat[c2]), np.append(yhat[c1], yhat[c2]))
    smo = cs(x[fok2])
    
    if len(np.append(xhat[c1], xhat[c2])) < 10:
        coef = np.polyfit(np.append(xhat[c1], xhat[c2]), np.append(yhat[c1], yhat[c2]), 1)
        smo = np.polyval(coef, x[fok2])
    
    ED = np.trapz(y[fok2] - smo, x[fok2]*60*60*24.)
    chisq = np.sum( ((y[fok2] - smo)/yerr[fok2])**2.) / np.size(fok2)
    EDerr = np.sqrt(ED**2. / (np.size(fok2)) / chisq)

    return ED, EDerr



def phase_med(x,y,per,k=55):
    phz1 = np.hstack(((x % per)/per-1, (x % per)/per))
    pfl1 = np.tile(y/np.nanmedian(y), 2)
    ss1 = np.argsort(phz1)
    psmo1 = pd.Series(pfl1[ss1]).rolling(k, center=True).median()

    ind1 = np.hstack((np.arange(len(x)), np.arange(len(x))))
    out1 = np.where((phz1[ss1] >= -0.5) & (phz1[ss1] < 0.5))[0]
    sout1 = np.argsort(ind1[ss1][out1])

    model1 = psmo1[out1].values[sout1] - 1
    return model1