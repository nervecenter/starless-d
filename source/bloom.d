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
double[][][3]
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

	double[xs.length][xs[0].length] r;
	foreach (int i, ref row; r)
		foreach (int j, ref ele; row)
			ele = sqrt(pow(xs[i][j], 2) + pow(ys[i][j], 2)) + 0.000001;

	/*double[][][] r_newAxes;
	foreach (int i, row; r)
		foreach (int j, ele; row)
		    r_newAxes[i][j] = [ele];*/

	double[r.length][r[0].length][3] afterDiv;
	foreach (int i, row; r)
	{
		foreach (int j, ele; row)
		{
			afterDiv[i][j][0] = ele / scale.x;
			afterDiv[i][j][1] = ele / scale.y;
			afterDiv[i][j][2] = ele / scale.z;
		}
	}
	
	double[xs.length][xs[0].length][3] kernel;
	foreach (int i, ref row; kernel)
		foreach (int j, ref col; row)
			foreach (int k, ref ele; col)
				ele = airy_disk(afterDiv[i][j][k]);

	double[3] kernelSum = [0.0, 0.0, 0.0];
	foreach (block; kernel)
	{
		foreach (row; block)
		{
			kernelSum[0] += row[0];
			kernelSum[1] += row[1];
			kernelSum[2] += row[2];
		}
	}

	foreach (ref block; kernel)
	{
		foreach (ref row; kernel)
		{
			row[0] /= kernelSum[0];
			row[1] /= kernelSum[1];
			row[2] /= kernelSum[2];
		}
	}

	return kernel;
}

// computed from approximate position of red green and blue in the spectrum
// it's a brutal approximation, but it vaguely looks like the real thing
enum Vector3 SPECTRUM = Vector3(1.0, 0.86, 0.61);


// convolve a 2D RGB array with three airy kernels with radius of
// red channel = radius and the other two rescaled as of above
// the kernel pixel size is fixed by kernel_radius
double[][][3]
airy_convolve(double[][][] array, double radius, int kernel_radius = 25)
{
	Vector3 scale = Vector3(SPECTRUM.x * radius,
							SPECTRUM.y * radius,
							SPECTRUM.z * radius);

    double[][][] kernel = generate_kernel(scale, kernel_radius);

    //output = np.zeros((array.shape[0], array.shape[1], 3));
	double[array.length][array[0].length][3] output;
	
    foreach (i; 0..3)
	{
        output[:, :, i] = convolve2d(array[:, :, i],
									 kernel[:, :, i],
									 mode = 'same',
									 boundary = 'symm');
	}

    return output;
}

