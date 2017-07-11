#!/usr/bin/env python
"""
Module for dealing with file input/output
"""
__author__ = "Alex Drlica-Wagner"
from collections import OrderedDict as odict

import numpy as np
import healpy as hp
import fitsio

def healpix_header_odict(nside,nest=False,ordering='RING',coord=None, partial=True):
    """Mimic the healpy header keywords."""
    hdr = odict([])
    hdr['PIXTYPE']=odict([('name','PIXTYPE'),
                          ('value','HEALPIX'),
                          ('comment','HEALPIX pixelisation')])

    ordering = 'NEST' if nest else 'RING'
    hdr['ORDERING']=odict([('name','ORDERING'),
                           ('value',ordering),
                           ('comment','Pixel ordering scheme, either RING or NESTED')])
    hdr['NSIDE']=odict([('name','NSIDE'),
                        ('value',nside),
                        ('comment','Resolution parameter of HEALPIX')])
    if coord:
        hdr['COORDSYS']=odict([('name','COORDSYS'), 
                               ('value',coord), 
                               ('comment','Ecliptic, Galactic or Celestial (equatorial)')])
    
    if not partial:
        hdr['FIRSTPIX']=odict([('name','FIRSTPIX'),
                               ('value',0), 
                               ('comment','First pixel # (0 based)')])
        hdr['LASTPIX']=odict([('name','LASTPIX'),
                              ('value',hp.nside2npix(nside)-1),
                              ('comment','Last pixel # (0 based)')])
    hdr['INDXSCHM']=odict([('name','INDXSCHM'),
                           ('value','EXPLICIT' if partial else 'IMPLICIT'),
                           ('comment','Indexing: IMPLICIT or EXPLICIT')])
    hdr['OBJECT']=odict([('name','OBJECT'), 
                         ('value','PARTIAL' if partial else 'FULLSKY'),
                         ('comment','Sky coverage, either FULLSKY or PARTIAL')])
    return hdr


def write_partial_map(filename, data, nside, coord=None, ordering='RING',
                      header=None,dtype=None,extname='PIX_DATA',**kwargs):
    """
    Partial HEALPix maps are used to efficiently store maps of the sky by only
    writing out the pixels that contain data.

    Three-dimensional data can be saved by supplying a distance modulus array
    which is stored in a separate extension.

    Parameters:
    -----------
    filename  : output file name
    pix       : healpix pixels
    data      : dictionary or recarray of data to write
    nside     : healpix nside of data
    coord : 'G'alactic, 'C'elestial, 'E'cliptic
    ordering : 'RING' or 'NEST'
    kwargs   : Passed to fitsio.write

    Returns:
    --------
    None
    """
    # First, convert data to records array
    if isinstance(data,dict):
        if 'PIXEL' not in data.keys():
            msg = "'PIXEL' column not found"
            raise ValueError(msg)

        pix = data.pop('PIXEL')
        names = ['PIXEL']
        arrays= [pix]

        for key,column in data.items():
            if column.shape[0] != len(pix):
                msg = "Length of '%s' (%i) does not match 'PIXEL' (%i)."%(key,column.shape[0],len(pix))
                logger.warning(msg)
             
            if len(column.shape) > 2:
                msg = "Unexpected shape for column '%s'."%(key)
                logger.warning()

            names.append(key)
            arrays.append(column.astype(dtype,copy=False))

        #data = np.rec.fromarrays(arrays,names=names)
        data = np.rec.array(arrays,names=names,copy=False)

        if 'PIXEL' not in data.dtype.names:
            msg = "'PIXEL' column not found"
            raise ValueError(msg)

    hdr = healpix_header_odict(nside=nside,coord=coord,ordering=ordering)
    fitshdr = fitsio.FITSHDR(hdr.values())
    if header is not None:
        for k,v in header.items():
            fitshdr.add_record({'name':k,'value':v})

    fitsio.write(filename,data,extname=extname,header=fitshdr,clobber=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
