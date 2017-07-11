#!/usr/bin/env python
"""
Module for dealing with file input/output
"""
__author__ = "Alex Drlica-Wagner"

def write_partial_map(filename,m,nside,nest=False,coord=None,extra_header=()):
    """Write partial HEALPix map.
    """
    if 'PIXEL' not in m.dtype.names:
        raise Exception("PIXEL column must be specified")

    cols=[]

    for name,fmt in m.dtype.descr:
        cols.append(pyfits.Column(name=name.upper(),
                              format='%s' % fitsfunc.getformat(fmt),
                              array=m[name]
                              ))


    tbhdu = pyfits.BinTableHDU.from_columns(cols)
    # add needed keywords
    tbhdu.header['PIXTYPE'] = ('HEALPIX', 'HEALPIX pixelisation')
    if nest: ordering = 'NESTED'
    else:    ordering = 'RING'
    tbhdu.header['ORDERING'] = (ordering,
                                'Pixel ordering scheme, either RING or NESTED')
    if coord:
        tbhdu.header['COORDSYS'] = (coord,
                                    'Ecliptic, Galactic or Celestial (equatorial)')
    tbhdu.header['EXTNAME'] = ('xtension',
                               'name of this binary table extension')
    tbhdu.header['NSIDE'] = (nside,'Resolution parameter of HEALPIX')

    tbhdu.header['INDXSCHM'] = ('EXPLICIT',
                                'Indexing: IMPLICIT or EXPLICIT')
    tbhdu.header['OBJECT'] = ('PARTIAL',
                              'Sky coverage, either FULLSKY or PARTIAL')

    # FIXME: In modern versions of Pyfits, header.update() understands a
    # header as an argument, and headers can be concatenated with the `+'
    # operator.
    for args in extra_header:
        tbhdu.header[args[0]] = args[1:]

    fitsfunc.writeto(tbhdu, filename)

def write_map(filename,m,nest=False,dtype=np.float32,fits_IDL=False,coord=None,partial=True,column_names=None,column_units=None,extra_header=()):
    """Writes an healpix map into an healpix file.

    Parameters
    ----------
    filename : str
      the fits file name
    m : array or sequence of 3 arrays
      the map to write. Possibly a sequence of 3 maps of same size.
      They will be considered as I, Q, U maps.
      Supports masked maps, see the `ma` function.
    nest : bool, optional
      If True, ordering scheme is assumed to be NESTED, otherwise, RING. Default: RING.
      The map ordering is not modified by this function, the input map array
      should already be in the desired ordering (run `ud_grade` beforehand).
    fits_IDL : bool, optional
      If True, reshapes columns in rows of 1024, otherwise all the data will
      go in one column. Default: True
    coord : str
      The coordinate system, typically 'E' for Ecliptic, 'G' for Galactic or 'C' for
      Celestial (equatorial)
    partial : bool, optional
      If True, fits file is written as a partial-sky file with explicit indexing.
      Otherwise, implicit indexing is used.  Default: False.
    column_names : str or list
      Column name or list of column names, if None we use:
      I_STOKES for 1 component,
      I/Q/U_STOKES for 3 components,
      II, IQ, IU, QQ, QU, UU for 6 components,
      COLUMN_0, COLUMN_1... otherwise
    column_units : str or list
      Units for each column, or same units for all columns.
    extra_header : list
      Extra records to add to FITS header.
    dtype: np.dtype or list of np.dtypes, optional
      The datatype in which the columns will be stored. Will be converted
      internally from the numpy datatype to the fits convention. If a list,
      the length must correspond to the number of map arrays. 
      Default: np.float32.
    """
    if not hasattr(m, '__len__'):
        raise TypeError('The map must be a sequence')

    m = pixelfunc.ma_to_array(m)
    if pixelfunc.maptype(m) == 0: # a single map is converted to a list
        m = [m]

    # check the dtype and convert it
    try:
        fitsformat = []
        for curr_dtype in dtype:
            fitsformat.append(getformat(curr_dtype))
    except TypeError:
        #dtype is not iterable
        fitsformat = [getformat(dtype)] * len(m)

    if column_names is None:
        column_names = standard_column_names.get(len(m), ["COLUMN_%d" % n for n in range(len(m))])
    else:
        assert len(column_names) == len(m), "Length column_names != number of maps"

    if column_units is None or isinstance(column_units, six.string_types):
        column_units = [column_units] * len(m)

    # maps must have same length
    assert len(set(map(len, m))) == 1, "Maps must have same length"
    nside = pixelfunc.npix2nside(len(m[0]))

    if nside < 0:
        raise ValueError('Invalid healpix map : wrong number of pixel')

    cols=[]
    if partial:
        fits_IDL = False
        mask = pixelfunc.mask_good(m[0])
        pix = np.where(mask)[0]
        if len(pix) == 0:
            raise ValueError('Invalid healpix map : empty partial map')
        m = [mm[mask] for mm in m]
        ff = getformat(np.min_scalar_type(-pix.max()))
        if ff is None:
            ff = 'I'
        cols.append(pf.Column(name='PIXEL',
                              format=ff,
                              array=pix,
                              unit=None))

    for cn, cu, mm, curr_fitsformat in zip(column_names, column_units, m, 
                                           fitsformat):
        if len(mm) > 1024 and fits_IDL:
            # I need an ndarray, for reshape:
            mm2 = np.asarray(mm)
            cols.append(pf.Column(name=cn,
                                   format='1024%s' % curr_fitsformat,
                                   array=mm2.reshape(mm2.size//1024,1024),
                                   unit=cu))
        else:
            cols.append(pf.Column(name=cn,
                                   format='%s' % curr_fitsformat,
                                   array=mm,
                                   unit=cu))

    tbhdu = pf.BinTableHDU.from_columns(cols)
    # add needed keywords
    tbhdu.header['PIXTYPE'] = ('HEALPIX', 'HEALPIX pixelisation')
    if nest: ordering = 'NESTED'
    else:    ordering = 'RING'
    tbhdu.header['ORDERING'] = (ordering,
                                'Pixel ordering scheme, either RING or NESTED')
    if coord:
        tbhdu.header['COORDSYS'] = (coord,
                                    'Ecliptic, Galactic or Celestial (equatorial)')
    tbhdu.header['EXTNAME'] = ('xtension',
                               'name of this binary table extension')
    tbhdu.header['NSIDE'] = (nside,'Resolution parameter of HEALPIX')
    if not partial:
        tbhdu.header['FIRSTPIX'] = (0, 'First pixel # (0 based)')
        tbhdu.header['LASTPIX'] = (pixelfunc.nside2npix(nside)-1,
                                   'Last pixel # (0 based)')
    tbhdu.header['INDXSCHM'] = ('EXPLICIT' if partial else 'IMPLICIT',
                                'Indexing: IMPLICIT or EXPLICIT')
    tbhdu.header['OBJECT'] = ('PARTIAL' if partial else 'FULLSKY',
                              'Sky coverage, either FULLSKY or PARTIAL')

    # FIXME: In modern versions of Pyfits, header.update() understands a
    # header as an argument, and headers can be concatenated with the `+'
    # operator.
    for args in extra_header:
        tbhdu.header[args[0]] = args[1:]

    writeto(tbhdu, filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
