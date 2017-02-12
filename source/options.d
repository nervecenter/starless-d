module starless.options;

import starless.logger;

enum Method { LEAPFROG, RK4 }

enum SkyTexture { None, Texture, Final }

SkyTexture
parseSkyTextureMode(string input)
{
	if (input == "none")
		return SkyTexture.None;
	else if (input == "texture")
		return SkyTexture.Texture;
	else if (input == "final")
		return SkyTexture.Final;
	else
	{
		Logger.instance.error("Error: "
							  ~ input
							  ~ " is not a valid sky rendering mode.");
	}
}

enum DiskTexture { None, Texture, Solid, Grid, Blackbody }

DiskTexture
parseDiskTextureMode(string input)
{
	if (input == "none")
		return DiskTexture.None;
	else if (input == "texture")
		return DiskTexture.Texture;
	else if (input == "solid")
		return DiskTexture.Solid;
	else if (input == "grid")
		return DiskTexture.Grid;
	else if (input == "blackbody")
		return DiskTexture.Blackbody;
	else
	{
		Logger.instance.error("Error: "
							  ~ input
							  ~ " is not a valid accretion disc rendering mode.");
	}
}

struct Geometry
{
	Vector3 cameraPos = Vector3(0.0, 1.0, -10.0);
	double tanFieldOfView = 1.5;
	Vector3 lookAt = Vector3(0.0, 0.0, 0.0);
	Vector3 upVector;
	double diskInner = 1.5;
	double diskOuter = 4;
	int distort = 1;
}

struct Materials
{
	int horizonGrid = 1;
	DiskTexture diskTexture;
	SkyTexture skyTexture;
	double skyDiskRatio;
	int fogDo = 1;
	double fogMult = 0.02;
	double diskMultiplier = 100.0;
	int diskIntensityDo = 1;
	int gain = 1;
	int normalize = -1;
	int blurDo = 1;
	int redShift = 1;
	int sRGBOut = 1;
	int sRGBIn = 1;
	double bloomCut = 2.0;
	int airyBloom = 1;
	double airyRadius = 1.0;
}

struct Options {
	bool lofi = false;
	bool disableDisplay = false;
	bool disableShuffling = false;
	int nThreads = 4;
	bool drawGraph = true;
	bool overrideRes = false;
	string sceneFname = "scenes/default.scene";
	int chunkSize = 9000;
	Resolution resolution = Resolution(160, 120);
	int iterations = 1000;
	double stepSize = 0.02;
	Materials materials;
	Geometry geometry;
}
