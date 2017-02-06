module starless.bloom;

//from scipy.special import j1
//from scipy.signal import convolve2d
//import numpy as np
import std.math,
	std.range,
	std.conv,
	starless.types,
	starless.j1;

// "airy disk" function (actually approximate, and a rescaling, but it's ok)
double
airy_disk(double x)
    return pow(2.0 * j1(x) / (x), 2);

RGB
airy_disk(RGB p)
	return RGB(airy_disk(p.r), airy_disk(p.g), airy_disk(p.b));

// generate a (2*size+1,2*size+1) convolution kernel with "radii" scale
// where the function above is assumed to have "radius" one
// scale is a 3-vector for RGB
RGB[][]
generate_kernel(RGB scale, int size)
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

	RGB[r.length][r[0].length] afterDiv;
	foreach (int i, row; r)
		foreach (int j, ele; row)
			afterDiv[i][j] = ele / scale;
	
	RGB[xs.length][xs[0].length] kernel;
	foreach (int i, ref row; kernel)
		foreach (int j, ref pix; row)
			pix = airy_disk(afterDiv[i][j]);

	RGB kernelSum = RGB(0.0, 0.0, 0.0);
	foreach (row; kernel)
		foreach (pixel; row)
			kernelSum += pixel;

	foreach (ref row; kernel)
		foreach (ref pix; row)
			pix /= kernelSum;

	return kernel;
}

// computed from approximate position of red green and blue in the spectrum
// it's a brutal approximation, but it vaguely looks like the real thing
enum RGB SPECTRUM = RGB(1.0, 0.86, 0.61);

// convolve a 2D RGB array with three airy kernels with radius of
// red channel = radius and the other two rescaled as of above
// the kernel pixel size is fixed by kernel_radius
RGB[][]
airy_convolve(RGB[][] array, double radius, int kernel_radius = 25)
{
	RGB scale = SPECTRUM * radius;

    RGB[][] kernel = generate_kernel(scale, kernel_radius);

    //output = np.zeros((array.shape[0], array.shape[1], 3));
	RGB[array.length][array[0].length] output;
	
	foreach (int i, ref row; output)
		foreach (int j, ref pix; row)
			pix = convolve2d(array[i][j], kernel[i][j], "same", "symm");
	/*output[:, :, i] = convolve2d(array[:, :, i],
									 kernel[:, :, i],
									 mode = 'same',
									 boundary = 'symm');*/

    return output;
}

