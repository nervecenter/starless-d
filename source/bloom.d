module starless.bloom;

//from scipy.special import j1
//from scipy.signal import convolve2d
//import numpy as np
import std.math,
	std.range,
	std.conv,
	starless.j1;

import starless.tracer : Vector3;

// "airy disk" function (actually approximate, and a rescaling, but it's ok)
double
airy_disk(double x)
{  
    return pow(2.0 * j1(x) / (x), 2);
}

// generate a (2*size+1,2*size+1) convolution kernel with "radii" scale
// where the function above is assumed to have "radius" one
// scale is a 3-vector for RGB
def
generate_kernel(Vector3 scale, int size)
{
    double[][] xs =
		iota(-size, size + 1)
		.map(n => n.to!double())
		.array
		.repeat
		.take(2*size+1)
		.array;
    double[][] ys =
		iota(-size, size + 1)
		.map(n => n.to!double()
			       .repeat()
			       .take(2*size+1)
			       .array)
		.array;

	double[][] r = xs.dup;
	foreach (int i, ref row; r)
		foreach (int j, ref ele; row)
			ele = sqrt(pow(xs[i][j], 2) + pow(ys[i][j], 2)) + 0.000001;

	double[][][] r_newAxes;
	foreach (int i, row; r)
		foreach (int j, ele; row)
			r_newAxes[i][j] = [ele];

	double[][][] afterDiv;
	foreach (int i, row; r_newAxes)
	{
		foreach (int j, col; row)
		{
			afterDiv[i][j] ~= col[0] / scale.x;
			afterDiv[i][j] ~= col[0] / scale.y;
			afterDiv[i][j] ~= col[0] / scale.z;
		}
	}
	
	double[xs.length][xs[0].length][3] kernel;
	foreach (int i, ref row; kernel)
		foreach (int j, ref col; row)
			foreach (int k, ref ele; col)
				ele = airy_disk(afterDiv[i][j][k]);

	//normalization
    kernel /= kernel.sum(axis = (0, 1))[np.newaxis, np.newaxis, :];

	return kernel;
}

// computed from approximate position of red green and blue in the spectrum
// it's a brutal approximation, but it vaguely looks like the real thing
enum Vector3 SPECTRUM = Vector3(1.0, 0.86, 0.61);


// convolve a 2D RGB array with three airy kernels with radius of
// red channel = radius and the other two rescaled as of above
// the kernel pixel size is fixed by kernel_radius
def
airy_convolve(array, radius, kernel_radius = 25)
{
    kernel = generate_kernel(radius * SPECTRUM , kernel_radius);

    output = np.zeros((array.shape[0], array.shape[1], 3));
    foreach (i; 0..3)
	{
        output[:, :, i] = convolve2d(array[:, :, i], kernel[:, :, i], mode = 'same', boundary = 'symm');
	}

    return output;
}

