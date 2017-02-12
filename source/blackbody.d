/*
 *  blackbody.d
 */

// This file is for every temperature/redshift/blackbody related function
module starless.blackbody;

// TODO: Find replacements for numpy, scipy
//import numpy as np
//import scipy.misc as spm
import std.math,
    std.algorithm,
    std.conv,
	starless.types,
    imaged.image;

import std.array : array;

// Accretion disk log temperature profile (R^{-3/4})
// 3/4 log(3)
enum LOGSHIFT = 0.823959216501;
  
double[]
disktemp(double[] sqrR, double logT0)
{
    // sqrR is an array of R^2
    // logT0 is log temperature (K) of accretion disk at ISCO

    double A = logT0 + LOGSHIFT;

    return sqrR.map!(r => to!double(A - 0.375 * log(r))).array;
}


// Blackbody temperature (abs, not log) -> relative intensity (absolute, not log)
// T is an array of abs temperatures

double[]
intensity(double[] T)
{
    // This is basically planck's law integrated over the visible spectrum,
    // which is assumed infinitesimal. The actual constant could have been
    // computed but it was safer and faster to gnuplot-fit it with a
    // gradient from
    // http://www.vendian.org/mncharity/dir3/blackbody/intensity.html
    foreach (ref t; T)
    {
        if (t < 1.0)
            t = 1.0;
        t = 1.0 / (exp(29622.4 / t) - 1);
    }
    
    return T;
}

RGB[]
getRamp()
{
	RGB[] ramp;
	// TODO: Make sure I'm loading just horizontal pixel data from colourtemp.jpg
	foreach (pix; load("data/colourtemp.jpg"))
		ramp ~= pix / 255.0;

	return ramp;
}

enum rampsz = 3;

double[]
clip(double[] arr, double floor, double ceiling)
{
    double[] outarr = arr;
    foreach (int i, val; arr)
    {
        if (val < floor)
            outarr[i] = floor;
        else if (ceiling < val)
            outarr[i] = ceiling;
        else
            outarr[i] = val;
    }
    return outarr;
}

// Returns array of 3-element arrays of RGB values based on temperatures array given
RGB[]
colour(double[] T)
{
    double[] Tp = T.map!(t => (t - 1000) / 29000.0 * rampsz).array;
    double[] indices = clip(Tp, 0.0, rampsz - 1.0001);
    int[] indicesInt = indices.map!(i => to!int(i)).array;

	RGB[] ramp = getRamp();

    return indicesInt.map!(i => ramp[i]).array;
}
