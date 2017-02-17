module starless.functions;

import starless.logger,
	starless.types,
	core.stdc.math;

double
norm(Vector3 vec)
{
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

// convert from linear rgb to srgb
RGB[]
rgbtosrgb(RGB[] arr)
{
	//see https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
	//Logger.instance.log("RGB -> sRGB...");
	
	RGB[] result;
	foreach (int i, pix; arr)
	foreach (int j, val; pix)
	{
		if (val > 0.0031308)
			result[i][j] = (core.stdc.math.pow(val, (1.0 / 2.4)) * 1.055) - 0.055;
		else
			result[i][j] = val * 12.92;
	}
	
	return result;
}

RGB[][]
rgbtosrgb(RGB[][] im)
{
	

// convert from srgb to linear rgb
RGB[]
srgbtorgb(RGB[] arr)
{
	Logger.instance.log("sRGB -> RGB...");
	
	RGB[] result;
	foreach (int i, pix; arr)
	foreach (int j, val; pix)
	{
		if (val > 0.04045)
			result[i][j] = core.stdc.math.pow(((val + 0.055) / 1.055), 2.4);
		else
			result[i][j] = val / 12.92;
	}

	return arr;
	// mask = arr > 0.04045;
	// arr[mask] += 0.055;
	// arr[mask] /= 1.055;
	// arr[mask] **= 2.4;
	// arr[-mask] /= 12.92;
}
