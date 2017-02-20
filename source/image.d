module starless.image;

import
    starless.types,
	starless.logger,
	std.conv,
	imaged;

Image toComplexImage(RGB[][] simpleImage)
{
	Image im = new Image(simpleImage.length, simpleImage[0].length);
	
	foreach(int i, row; simpleImage)
	foreach(int j, pix; row)
	{
		im.setPixel(i, j, Pixel(pix.r, pix.g, pix.b));
	}

	return im;
}

RGB[][] toSimpleImage(Image im)
{
	Pixel pixel;
	RGB[][] result = new RGB[][](im.width(), im.height());
	for (int i = 0; i < im.width(); i++)
	for (int j = 0; j < im.height(); j++)
	{
		pixel = im.getPixel(i, j);
		result[i][j] = RGB(pixel.r.to!double(),
						   pixel.g.to!double(),
						   pixel.b.to!double());
	}

	return result;
}

RGB[][] resizeSimpleImage(RGB[][] simpleImage, uint newWidth, uint newHeight)
{
	Image complexImage = simpleImage.toComplexImage();
	complexImage.resize(newWidth, newHeight, ResizeAlgo.BICUBIC);
	return complexImage.toSimpleImage();
}

RGB[][] loadImage(string fileloc)
{
	Image loadedImage;
	try
	{
		Decoder dcd = getDecoder(fileloc);
		dcd.parseFile(fileloc);
		loadedImage = dcd.image();
	}
	catch (Exception e)
		Logger.instance.error("Error loading image "
							  ~ fileloc ~ ": " ~ e.msg);

	return toSimpleImage(loadedImage);
}

