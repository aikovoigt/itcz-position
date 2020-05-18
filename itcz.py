import numpy as np

def reorder_south2north(data, lat):
    # if latitude is not indexed from SP to NP, then reorder
    if lat[0]>lat[1]:
        lat = lat[::-1]
        data  = data[::-1]
    return data, lat

def get_itczposition_adam(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    return np.nansum(lati * areai * pri) / np.nansum(areai * pri)

def test_itczposition_adam(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    # calculate itcz position according to Adam
    itcz = get_itczposition_adam(pr, lat, latboundary, dlat)
    # lat index corresponding to itcz 
    ilati = np.argmin(np.abs(itcz - lati))
    aux1=np.abs(np.nansum((lati[0:ilati+1]-itcz)*pri[0:ilati+1]*areai[0:ilati+1]))
    aux2=np.abs(np.nansum((lati[ilati+1:]-itcz)*pri[ilati+1:]*areai[ilati+1:]))
    return aux1, aux2    
    
def get_itczposition_voigt(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    # area-integrated precip (up to constant factor)
    tot = np.sum(pri*areai)
    # integrated pri from southern latboundary to lati
    pri_int = np.zeros(lati.size) + np.nan
    for j in range(0, lati.size):
        pri_int[j] = np.sum(pri[0:j+1]*areai[0:j+1])
    # itcz is where integrated pri is 0.5 of total area-integrated pri
    return lati[np.argmin(np.abs(pri_int - 0.5*tot))]    

def test_itczposition_voigt(pr, lat, latboundary, dlat):
    pr, lat = reorder_south2north(pr, lat)
    # interpolate lat and pr on dlat grid
    lati  = np.arange(-latboundary, latboundary, dlat)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    # calculate itcz position according to Adam
    itcz = get_itczposition_voigt(pr, lat, latboundary, dlat)
    # lat index corresponding to itcz 
    ilati = np.argmin(np.abs(itcz - lati))
    aux1=np.nansum(pri[0:ilati+1]*areai[0:ilati+1])
    aux2=np.nansum(pri[ilati+1:]*areai[ilati+1:])
    return aux1, aux2