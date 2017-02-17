module starless.image;

import
    starless.types,
	starless.logger,
	std.conv,
	imaged;

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

	Pixel pixel;
	RGB[][] result = new RGB[][](loadedImage.width(), loadedImage.height());
	for (int i = 0; i < loadedImage.width(); i++)
	for (int j = 0; j < loadedImage.height(); j++)
	{
		pixel = loadedImage.getPixel(i, j);
		result[i][j] = RGB(pixel.r.to!double(),
						   pixel.g.to!double(),
						   pixel.b.to!double());
	}

	return result;
}
