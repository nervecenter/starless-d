module starless.options;

import
    starless.logger,
	starless.functions,
	starless.types,
	std.file,
	std.stdio,
	std.conv,
	std.json,
	std.array;

import std.algorithm : map;
import std.format : format;
import std.string : strip, removechars;

enum Method { Leapfrog, RK4 }

enum SkyTexture { None, Texture, Final }

enum DiskTexture { None, Texture, Solid, Grid, Blackbody }

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
		assert(0);
	}
}

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
		assert(0);
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
	DiskTexture diskTexture = DiskTexture.Texture;
	SkyTexture skyTexture = SkyTexture.Texture;
	double skyDiskRatio = 0.05;
	int fogDo = 1;
	double fogMult = 0.02;
	double diskMultiplier = 100.0;
	int diskIntensityDo = 1;
	double gain = 1.0;
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
	// string sceneFname = "scenes/blackbody.json";
	string sceneFname;
	int chunkSize = 9000;
	Resolution resolution = Resolution(160, 120);
	int iterations = 1000;
	double stepSize = 0.02;
	Materials materials;
	Geometry geometry;
	int fogSkip = 1;
	Method method = Method.RK4;
}

JSONValue
readSceneJSON(string filename)
{
	auto jsonfile = File("scenes/" ~ filename, "r");
	auto jsonstr =
		jsonfile.byLine()
		.map!strip
		.map!(l => l.removechars(" "))
		.map!(l => l.removechars("\t"))
		.join();
	return parseJSON(jsonstr);
}

Options
parseOptions(string[] argsTail)
{
	auto logger = Logger.instance;
	Options options;

	string[] flags;
	if (argsTail[$-1][0] != '-')
	{
		options.sceneFname = argsTail[$-1];
		flags = argsTail[0..$-1];
	}
	else
		flags = argsTail;

	// TODO: Overhaul CLI option parsing
	foreach (flag; flags)
	{
		// -d               Use low fidelity options from scene.
		if (flag == "-d")
			options.lofi = true;

		// --no-graph       Don't render a schematic diagram..
		else if (flag == "--no-graph")
			options.drawGraph = false;

		else if (flag == "--no-display")
			options.disableDisplay = true;

		else if (flag == "--no-shuffle")
			options.disableShuffling = true;

		else if ((flag == "-o") || (flag == "--no-bs"))
		{
			options.drawGraph = false;
			options.disableDisplay = true;
			options.disableShuffling = true;
		}

		else if (flag[0..2] == "-c")
			options.chunkSize = to!int(flag[2..$]);

		else if (flag[0..2] == "-j")
			options.nThreads = to!int(flag[2..$]);

		else if (flag[0..2] == "-r")
		{
			int[] res = flag[2..$].split("x").map!(n => parse!int(n)).array;
			if (res.length != 2)
			{
				logger.error("Resolution \"" ~ flag[2..$] ~ "\" unreadable.\n"
					~ "Please format resolution correctly (e.g.: -r640x480).");
			}
			options.resolution = res.toResolution();
			options.overrideRes = true;
		}

		else if (flag[0] == '-')
			logger.error("Unrecognized option: " ~ flag);
	}

	if (!exists("scenes/" ~ options.sceneFname))
	{
		logger.log("Scene file \"" ~ options.sceneFname ~ "\" does not exist");
		logger.log("Using defaults.");
		return options;
	}

	JSONValue config;
	logger.log("Reading scene " ~ options.sceneFname ~ "...");
	try
		config = readSceneJSON(options.sceneFname);
	catch (JSONException e)
		logger.error(e.msg);
	
	//this section works, but only if the .scene file is good
	//if there's anything wrong, it's a trainwreck
	//must rewrite
	try
	{
		if (!options.overrideRes)
		{
			auto loficonf = config["lofi"];
			options.resolution =
				loficonf["Resolution"].array
				.map!(n => n.integer.to!int()).array
				.toResolution();
			options.iterations = loficonf["Iterations"].integer.to!int();
			options.stepSize = loficonf["Stepsize"].floating;
		}
	}
    catch (JSONException e)
	{
		logger.log("Error reading scene file: Insufficient data in \"lofi\" section.");
		logger.log("Using defaults.");
	}

	if (!options.lofi)
	{
		try
		{
			if (!options.overrideRes)
			{
				auto hificonf = config["hifi"];
				options.resolution =
					hificonf["Resolution"].array
					.map!(n => n.integer.to!int()).array
					.toResolution();
				options.iterations = hificonf["Iterations"].integer.to!int();
				options.stepSize = hificonf["Stepsize"].floating;
			}
		}
		catch(JSONException e)
		{
			logger.log("Error reading scene file: Insufficient data in \"hifi\" section.");
			logger.log("Using lofi/defaults.");
		}
	}

	try
	{
		auto geoconf = config["geometry"];
		Geometry geo;
		
		geo.cameraPos =
			geoconf["Cameraposition"].array
			.map!(f => f.floating).array
			.toVector3();
		geo.tanFieldOfView = geoconf["Fieldofview"].floating;
		geo.lookAt =
			geoconf["Lookat"].array
			.map!(f => f.floating).array
			.toVector3();
		geo.upVector =
			geoconf["Upvector"].array
			.map!(f => f.floating).array
			.toVector3();
		geo.distort = geoconf["Distort"].integer.to!int();
		geo.diskInner = geoconf["Diskinner"].floating;
		geo.diskOuter = geoconf["Diskouter"].floating;

		options.geometry = geo;
	}
	catch (JSONException e)
	{
		logger.log("Error reading scene file: "
				   ~ "Insufficient data in geometry section.\n"
				   ~ "Using defaults.");
	}

	try
	{
		//options for 'blackbody' disktexture
		auto matconf = config["materials"];
		Materials mats;

		mats.diskMultiplier = matconf["Diskmultiplier"].floating;
		//DISK_ALPHA_MULTIPLIER = float(cfp.get('materials','Diskalphamultiplier'))
		mats.diskIntensityDo = matconf["Diskintensitydo"].integer.to!int();
		mats.redShift = matconf["Redshift"].integer.to!int();

		mats.gain = matconf["Gain"].floating;
		mats.normalize = matconf["Normalize"].integer.to!int();

		mats.bloomCut = matconf["Bloomcut"].floating;
		mats.horizonGrid = matconf["Horizongrid"].integer.to!int();
		mats.diskTexture = matconf["Disktexture"].str.parseDiskTextureMode();
		mats.skyTexture = matconf["Skytexture"].str.parseSkyTextureMode();
		mats.skyDiskRatio = matconf["Skydiskratio"].floating;
		mats.fogDo = matconf["Fogdo"].integer.to!int();
		mats.blurDo = matconf["Blurdo"].integer.to!int();
		mats.airyBloom = matconf["Airy_bloom"].integer.to!int();
		mats.airyRadius = matconf["Airy_radius"].floating;
		mats.fogMult = matconf["Fogmult"].floating;
		//perform linear rgb->srgb conversion
		mats.sRGBOut = matconf["sRGBOut"].integer.to!int();
		mats.sRGBIn = matconf["sRGBIn"].integer.to!int();

		options.materials = mats;
	}
	catch (JSONException e)
	{
		logger.log("Error reading scene file: "
				   ~ "Insufficient data in materials section.\n"
				   ~ "Using defaults.");
	}

	logger.log(format("Resolution: %dx%d", options.resolution.x, options.resolution.y));

	//ensure the observer's 4-velocity is timelike
	//since as of now the observer is schwarzschild stationary, we just need to check
	//whether he's outside the horizon.
	if (norm(options.geometry.cameraPos) <= 1.0)
	{
		logger.error("The observer's 4-velocity is not timelike.\n"
					 ~ "(Try placing the observer outside the event horizon.)");
	}
	
	return options;
}
