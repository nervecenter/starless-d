/*import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndim
import scipy.misc as spm
import random,sys,time,os
import datetime

import multiprocessing as multi
import ctypes

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import blackbody as bb
import bloom

import gc
import curses*/

import std.file,
	std.conv,
	core.stdc.math,
	starless.types,
	starless.logger,
	ggplotd.aes,
	ggplotd.ggplotd,
	ggplotd.geom,
	toml.d;

enum Method { LEAPFROG, RK4 }

enum ST { NONE, TEXTURE, FINAL }

ST[string] st_dict = ["none"      : ST.NONE,
					  "texture"   : ST.TEXTURE,
					  "final"     : DT.FINAL];

enum DT { NONE, TEXTURE, SOLID, GRID, BLACKBODY }

DT[string] dt_dict = ["none"      : DT.NONE,
					  "texture"   : DT.TEXTURE,
					  "solid"     : DT.SOLID,
					  "grid"      : DT.GRID,
					  "blackbody" : DT.BLACKBODY];

struct Options {
	bool LOFI = false;

	bool DISABLE_DISPLAY = false;
	bool DISABLE_SHUFFLING = false;

	int NTHREADS = 4;

	bool DRAWGRAPH = true;

	bool OVERRIDE_RES = false;

	string SCENE_FNAME = "scenes/default.scene";

	int CHUNKSIZE = 9000;

	Resolution RESOLUTION;
}

double
norm(Vector3 vec)
	return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);


void main(string[] args)
{
	auto logger = new Logger();
	
	Options options = Options();
	
	foreach (arg; args[1..$])
	{
		if (arg == "-d")
			options.LOFI = true;

		else if (arg == "--no-graph")
			options.DRAWGRAPH = false;

		else if (arg == "--no-display")
			options.DISABLE_DISPLAY = true;

		else if (arg == "--no-shuffle")
			options.DISABLE_SHUFFLING = true;

		else if ((arg == "-o") || (arg == "--no-bs"))
		{
			options.DRAWGRAPH = false;
			options.DISABLE_DISPLAY = true;
			options.DISABLE_SHUFFLING = true;
		}

		else if (arg[0..2] == "-c")
			options.CHUNKSIZE = to!int(arg[2..$]);

		else if (arg[0..2] == "-j")
			options.NTHREADS = to!int(arg[2..$]);

		else if (arg[0..2] == "-r")
		{
			int[] res = arg[2..$].split('x').map(n => to!int(n)).array;
			if (len(res) != 2)
			{
				logger.error("Resolution \"" ~ arg[2..$] ~ "\" unreadable.\n"
					~ "Please format resolution correctly (e.g.: -r640x480).");
			}
			options.RESOLUTION = Resolution(res[0], res[1]);
			options.OVERRIDE_RES = true;
		}

		else if (arg[0] == '-')
			logger.error("Unrecognized option: " ~ arg);

		else
			options.SCENE_FNAME = arg;
	}

	if (!exists(options.SCENE_FNAME))
		logger.error("Scene file \"" ~ options.SCENE_FNAME ~ "\" does not exist");


	auto defaults = [
		"Distort":"1",
		"Fogdo":"1",
		"Blurdo":"1",
		"Fogmult":"0.02",
		"Diskinner":"1.5",
		"Diskouter":"4",
		"Resolution":"160,120",
		"Diskmultiplier":"100.",
		"Gain":"1",
		"Normalize":"-1",
		"Blurdo":"1",
		"Bloomcut":"2.0",
		"Airy_bloom":"1",
		"Airy_radius":"1.",
		"Iterations":"1000",
		"Stepsize":"0.02",
		"Cameraposition":"0.,1.,-10",
		"Fieldofview":1.5,
		"Lookat":"0.,0.,0.",
		"Horizongrid":"1",
		"Redshift":"1",
		"sRGBOut":"1",
		"Diskintensitydo":"1",
		"sRGBIn":"1",
	];

	logger.debug("Reading scene %s...", SCENE_FNAME);
	auto config = parseFile(SCENE_FNAME);

	bool FOGSKIP = true;

	Method METHOD = Method.RK4;

	//enums to avoid per-iteration string comparisons


	//this section works, but only if the .scene file is good
	//if there's anything wrong, it's a trainwreck
	//must rewrite
	try
	{
		if (!options.OVERRIDE_RES)
		{
			auto loficonf = config["lofi"];
			int[2] res =
				loficonf["Resolution"].str.split(',')
				.map(n => parse!int(n));
			options.RESOLUTION = Resolution(res[0], res[1]);
			options.NITER = config["lofi"]["Iterations"].int;
			options.STEP = config["lofi"]["Stepsize"].str.parse!double();
		}
	}
    catch (TOMLException e)
	{
		logger.debug("Error reading scene file: Insufficient data in \"lofi\" section");
		logger.debug("Using defaults.");
	}

	if (!options.LOFI)
	{
		try
		{
			if (!options.OVERRIDE_RES)
			{
				auto hificonf = config["hifi"];
				int[2] res =
					hificonf["Resolution"].str.split(',')
					.map(n => parse!int(n));
				options.RESOLUTION = Resolution(res[0], res[1]);
				options.NITER = hificonf["Iterations"].int;
				options.STEP = hificonf["Stepsize"].str.parse!double();
			}
		}
		catch(TOMLException e)
		{
			logger.debug("No data in hifi section. Using lofi/defaults.");
		}
	}

	try
	{
		auto geoconf = config["geometry"];
		Geometry geo;
		
		double[3] cam =
			geoconf["Cameraposition"].str.split(',')
			.map(f => parse!double(f));
		geo.CAMERA_POS = Vector3(cam[0], cam[1], cam[2]);
		geo.TANFOV = geoconf["Fieldofview"].str.parse!double();
		double[3] look =
			geoconf["Lookat"].str.split(',')
			.map(f => parse!double(f));
		geo.LOOKAT = Vector3(look[0], look[1], look[2]);
		double[3] up =
			geoconf["Upvector"].str.split(',')
			.map(f => parse!double(f));
		geo.UPVEC = Vector3(up[0], up[1], up[2]);
		geo.DISTORT = geoconf["Distort"].int;
		geo.DISKINNER = geoconf["Diskinner"].str.parse!double();
		geo.DISKOUTER = geoconf["Diskouter"].str.parse!double();

		options.geometry = geo;
	}
	catch (TOMLException e)
	{
		logger.debug("Error reading scene file: Insufficient data in geometry section");
		logger.debug("Using defaults.");
	}


	try
	{
		//options for 'blackbody' disktexture
		auto matconf = config["materials"];
		Materials mats;

		mats.DISK_MULTIPLIER = matconf["Diskmultiplier"].str.parse!double();
		//DISK_ALPHA_MULTIPLIER = float(cfp.get('materials','Diskalphamultiplier'))
		mats.DISK_INTENSITY_DO = matconf["Diskintensitydo"].int;
		mats.REDSHIFT = matconf["Redshift"].str.parse!double();

		mats.GAIN = matconf["Gain"].str.parse!double();
		mats.NORMALIZE = matconf["Normalize"].str.parse!double();

		mats.BLOOMCUT = matconf["Bloomcut"].str.parse!double();
		mats.HORIZON_GRID = matconf["Horizongrid"].int;
		mats.DISK_TEXTURE = matconf["Disktexture"].str;
		mats.SKY_TEXTURE = matconf["Skytexture"].str;
		mats.SKYDISK_RATIO = matconf["Skydiskratio"].str.parse!double();
		mats.FOGDO = matconf["Fogdo"].int;
		mats.BLURDO = matconf["Blurdo"].int;
		mats.AIRY_BLOOM = matconf["Airy_bloom"].int;
		mats.AIRY_RADIUS = matconf["Airy_radius"].str.parse!double();
		mats.FOGMULT = matconf["Fogmult"].str.parse!double();
		//perform linear rgb->srgb conversion
		mats.SRGBOUT = matconf["sRGBOut"].int;
		mats.SRGBIN = matconf["sRGBIn"].int;

		options.materials = mats;
	}
	catch (TOMLException e)
	{
		logger.debug("Error reading scene file: Insufficient data in materials section.");
		logger.debug("Using defaults.");
	}


	// converting mode strings to mode ints
	try
	{
		options.materials.DISK_TEXTURE_INT = dt_dict[DISK_TEXTURE];
	}
	catch (KeyError e)
	{
		logger.debug("Error: %s is not a valid accretion disc rendering mode", DISK_TEXTURE);
		return;
	}

	try
	{
		options.materials.SKY_TEXTURE_INT =
			dt_dict[options.materialsSKY_TEXTURE];
	}
	catch (KeyError e)
	{
		logger.debug("Error: %s is not a valid sky rendering mode",
					 options.materials.SKY_TEXTURE);
		return;
	}


	logger.debug("%dx%d", options.RESOLUTION.w, options.RESOLUTION.h);

	//ensure the observer's 4-velocity is timelike
	//since as of now the observer is schwarzschild stationary, we just need to check
	//whether he's outside the horizon.
	if (norm(options.geometry.CAMERA_POS) <= 1.0)
	{
		logger.debug("Error: the observer's 4-velocity is not timelike.");
		logger.debug("(Try placing the observer outside the event horizon)");
		return;
	}


	double DISKINNERSQR =
		options.geometry.DISKINNER * options.geometry.DISKINNERDISKINNER;
	double DISKOUTERSQR =
		options.geometry.DISKOUTER * options.geometry.DISKOUTER;

	//ensuring existence of tests directory
	if (!exists("tests"))
		mkdir("tests");

	//GRAPH
	if (options.DRAWGRAPH)
	{
		logger.debug("Drawing schematic graph...");
		auto gg = GGPlotD();

		//plt.Circle(centerpt, radius, fc=facecolor)
		//g_diskout       = plt.Circle((0,0),DISKOUTER, fc='0.75');
		gg = aes!("x", "y", "size", "colour")
			(0.0, 0.0, options.geometry.DISKOUTER, 0.75)
			.geomEllipse().putIn(gg);

		//g_diskin        = plt.Circle((0,0),DISKINNER, fc='white');
		gg = aes!("x", "y", "size", "colour")
			(0.0, 0.0, options.geometry.DISKINNER, "white")
			.geomEllipse().putIn(gg);

		
		//g_photon        = plt.Circle((0,0),1.5,ec='y',fc='none');
		gg = aes!("x", "y", "size", "colour", "fill")
			(0.0, 0.0, 1.5, "yellow", 0.0)
			.geomEllipse().putIn(gg);

		//g_horizon       = plt.Circle((0,0),1,color='black');
		gg = aes!("x", "y", "size", "colour")
			(0.0, 0.0, 1.0, "black")
			.geomEllipse().putIn(gg);

		Vector3 cam_pos = options.geometry.CAMERA_POS;
		
		//g_cameraball    = plt.Circle((CAMERA_POS[2],CAMERA_POS[0]),0.2,color='black');
		gg = aes!("x", "y", "size", "colour")
			(cam_pos.x, cam_pos.z, 0.2, "black")
			.geomEllipse().putIn(gg);
		
		double gscale = 1.1 * norm(cam_pos);
		gg.put(xaxisRange(-gscale, gscale));
		gg.put(yaxisRange(-gscale, gscale));
		
		// draw line from camera to lookat point
		Vector3 lookat = options.geometry.LOOKAT;
		auto aes =
			Aes!(double[], "x", double[], "y", double, "colour")
			([cam_pos.x, lookat.x], [cam_pos.z, lookat.z], 0.05);
		/*ax.plot([CAMERA_POS[2],LOOKAT[2]],
				[CAMERA_POS[0],LOOKAT[0]],
				color='0.05',
				linestyle='-');*/
		
		logger.debug("Saving diagram...");
		gg.save('tests/graph.png');
	}

	// these need to be here
	// convert from linear rgb to srgb
	void
	rgbtosrgb(RGB[] arr)
	{
		//see https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
		logger.debug("RGB -> sRGB...");

		foreach (ref pix; arr)
		{
			foreach (ref val; pix)
			{
				if (val > 0.0031308)
					val = (core.std.math.pow(val, (1.0 / 2.4)) * 1.055) - 0.055;
				else
					val *= 12.92;
			}
		}
	}

	// convert from srgb to linear rgb
	void
	srgbtorgb(RGB[] arr)
	{
		logger.debug("sRGB -> RGB...");

		foreach (ref pix; arr)
		{
			foreach (ref val; pix)
			{
				if (val > 0.04045)
					val = core.std.math.pow(((val + 0.055) / 1.055), 2.4);
				else
					val /= 12.92;
			}
		}
		mask = arr > 0.04045;
		arr[mask] += 0.055;
		arr[mask] /= 1.055;
		arr[mask] **= 2.4;
		arr[-mask] /= 12.92;
	}

	logger.debug("Loading textures...");
	if (options.materials.SKY_TEXTURE == 'texture')
	{
		auto texarr_sky = spm.imread('textures/bgedit.jpg');
		// must convert to float here so we can work in linear colour
		texarr_sky = texarr_sky.astype(float);
		texarr_sky /= 255.0;

		// must do this before resizing to get correct results
		if (options.materials.SRGBIN)
			srgbtorgb(texarr_sky);
	
		if (!options.LOFI)
		{
			//   maybe doing this manually and then loading is better.
			logger.debug("(Zooming sky texture...)");
			texarr_sky = spm.imresize(texarr_sky,2.0,interp='bicubic');
			// imresize converts back to uint8 for whatever reason
			texarr_sky = texarr_sky.astype(float);
			texarr_sky /= 255.0;
		}
	}

texarr_disk = None
if DISK_TEXTURE == 'texture':
    texarr_disk = spm.imread('textures/adisk.jpg')
if DISK_TEXTURE == 'test':
    texarr_disk = spm.imread('textures/adisktest.jpg')
if texarr_disk is not None:
    // must convert to float here so we can work in linear colour
    texarr_disk = texarr_disk.astype(float)
    texarr_disk /= 255.0
    if SRGBIN:
        srgbtorgb(texarr_disk)


			//defining texture lookup
def lookup(texarr,uvarrin): //uvarrin is an array of uv coordinates
    uvarr = np.clip(uvarrin,0.0,0.999)

    uvarr[:,0] *= float(texarr.shape[1])
    uvarr[:,1] *= float(texarr.shape[0])

    uvarr = uvarr.astype(int)

    return texarr[  uvarr[:,1], uvarr[:,0] ]



logger.debug("Computing rotation matrix...")

			// this is just standard CGI vector algebra

FRONTVEC = (LOOKAT-CAMERA_POS)
FRONTVEC = FRONTVEC / np.linalg.norm(FRONTVEC)

LEFTVEC = np.cross(UPVEC,FRONTVEC)
LEFTVEC = LEFTVEC/np.linalg.norm(LEFTVEC)

NUPVEC = np.cross(FRONTVEC,LEFTVEC)

viewMatrix = np.zeros((3,3))

viewMatrix[:,0] = LEFTVEC
viewMatrix[:,1] = NUPVEC
viewMatrix[:,2] = FRONTVEC


			//array [0,1,2,...,numPixels]
pixelindices = np.arange(0,RESOLUTION[0]*RESOLUTION[1],1)

			//total number of pixels
numPixels = pixelindices.shape[0]

logger.debug("Generated %d pixel flattened array.", numPixels)

			//useful constant arrays
ones = np.ones((numPixels))
ones3 = np.ones((numPixels,3))
UPFIELD = np.outer(ones,np.array([0.,1.,0.]))

			//random sample of floats
ransample = np.random.random_sample((numPixels))

//returns a constant 3-vector array (don't use for varying vectors)
def vec3a(vec):
	return np.outer(ones,vec)

def vec3(x,y,z):
    return vec3a(np.array([x,y,z]))

def norm(vec):
    // you might not believe it, but this is the fastest way of doing this
    // there's a stackexchange answer about this
    return np.sqrt(np.einsum('...i,...i',vec,vec))

def normalize(vec):
    //Return vec/ (np.outer(norm(vec),np.array([1.,1.,1.])))
    return vec / (norm(vec)[:,np.newaxis])

		// An efficient way of computing the sixth power of r
		// much faster than pow!
		// np has this optimization for power(a,2)
		// but not for power(a,3)!

def sqrnorm(vec):
    return np.einsum('...i,...i',vec,vec)

def sixth(v):
    tmp = sqrnorm(v)
    return tmp*tmp*tmp


def RK4f(y,h2):
    f = np.zeros(y.shape)
    f[:,0:3] = y[:,3:6]
    f[:,3:6] = - 1.5 * h2 * y[:,0:3] / np.power(sqrnorm(y[:,0:3]),2.5)[:,np.newaxis]
    return f


		// this blends colours ca and cb by placing ca in front of cb
def blendcolors(cb,balpha,ca,aalpha):
            #* np.outer(aalpha, np.array([1.,1.,1.])) + \
				//return  ca + cb * np.outer(balpha*(1.-aalpha),np.array([1.,1.,1.]))
    return  ca + cb * (balpha*(1.-aalpha))[:,np.newaxis]


				// this is for the final alpha channel after blending
def blendalpha(balpha,aalpha):
    return aalpha + balpha*(1.-aalpha)


def saveToImg(arr,fname):
    logger.debug(" - saving %s...", fname)
		//copy
    imgout = np.array(arr)
		//clip
    imgout = np.clip(imgout,0.0,1.0)
		//rgb->srgb
    if SRGBOUT:
        rgbtosrgb(imgout)
			//unflattening
    imgout = imgout.reshape((RESOLUTION[1],RESOLUTION[0],3))
    plt.imsave(fname,imgout)

			// this is not just for bool, also for floats (as grayscale)
def saveToImgBool(arr,fname):
    saveToImg(np.outer(arr,np.array([1.,1.,1.])),fname)


		//for shared arrays

def tonumpyarray(mp_arr):
    a = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
    a.shape = ((numPixels,3))
    return a




		//PARTITIONING

		//partition viewport in contiguous chunks

		//CHUNKSIZE = 9000
if not DISABLE_SHUFFLING:
    np.random.shuffle(pixelindices)
chunks = np.array_split(pixelindices,numPixels/CHUNKSIZE + 1)

NCHUNKS = len(chunks)

logger.debug("Split into %d chunks of %d pixels each", NCHUNKS, chunks[0].shape[0])

total_colour_buffer_preproc_shared = multi.Array(ctypes.c_float, numPixels * 3)
total_colour_buffer_preproc = tonumpyarray(total_colour_buffer_preproc_shared)

		//open preview window

if not DISABLE_DISPLAY:
    logger.debug("Opening display...")

    plt.ion()
    plt.imshow(total_colour_buffer_preproc.reshape((RESOLUTION[1],RESOLUTION[0],3)))
    plt.draw()


		//shuffle chunk list (does very good for equalizing load)

random.shuffle(chunks)


		//partition chunk list in schedules for single threads

schedules = []

		//from http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
q,r = divmod(NCHUNKS, NTHREADS)
indices = [q*i + min(i,r) for i in range(NTHREADS+1)]

for i in range(NTHREADS):
    schedules.append(chunks[ indices[i]:indices[i+1] ]) 



logger.debug("Split list into %d schedules with %s chunks each", NTHREADS, ", ".join([str(len(s)) for s in schedules]))




		// global clock start
start_time = time.time()

itcounters = [0 for i in range(NTHREADS)]
chnkcounters = [0 for i in range(NTHREADS)]

		//killers

killers = [False for i in range(NTHREADS)]


		// command line output

class Outputter:
    def name(self,num):
        if num == -1:
            return "M"
        else:
            return str(num)

    def __init__(self):
        self.message = {}
        self.queue = multi.Queue()
        self.stdscr = curses.initscr()
        curses.noecho()

        for i in range(NTHREADS):
            self.message[i] = "..."
        self.message[-1] = "..."

    def doprint(self):
        for i in range(NTHREADS + 1):
            self.stdscr.addstr(
                i, 0, self.name(i - 1) + "] " + self.message[i - 1])
        self.stdscr.refresh()

    def parsemessages(self):
        doref = False
        while not self.queue.empty():
            i,m = self.queue.get()
            self.setmessage(m, i)
            doref = True

        if doref:
            self.doprint()

    def setmessage(self,mess,i):
        self.message[i] = mess.ljust(60)
			//self.doprint()

    def __del__(self):
        try:
            curses.echo()
            curses.endwin()
            print('\n'*(NTHREADS+1))
        except:
            pass

output = Outputter()

def format_time(secs):
    if secs < 60:
        return "%d s"%secs
    if secs < 60*3:
        return "%d m %d s"%divmod(secs,60)
    return "%d min"%(secs/60)

def showprogress(messtring,i,queue):
    global start_time

    elapsed_time = time.time() - start_time
    progress = float(itcounters[i])/(len(schedules[i])*NITER)

    try:
        ETA = elapsed_time / progress * (1-progress)
    except ZeroDivisionError:
        ETA = 0

    mes = "%d%%, %s remaining. Chunk %d/%d, %s"%(
            int(100*progress), 
            format_time(ETA),
            chnkcounters[i],
            len(schedules[i]),
            messtring.ljust(30)
                )
    queue.put((i,mes))


			//def showprogress(m,i):
			//    pass

def raytrace_schedule(i,schedule,total_shared,q): # this is the function running on each thread
			//global schedules,itcounters,chnkcounters,killers

    if len(schedule) == 0:
        return

    total_colour_buffer_preproc = tonumpyarray(total_shared)

			//schedule = schedules[i]

    itcounters[i] = 0
    chnkcounters[i]= 0

    for chunk in schedule:
				  //if killers[i]:
				  //    break
        chnkcounters[i]+=1

			//number of chunk pixels
        numChunk = chunk.shape[0]

			//useful constant arrays 
        ones = np.ones((numChunk))
        ones3 = np.ones((numChunk,3))
        UPFIELD = np.outer(ones,np.array([0.,1.,0.]))
        BLACK = np.outer(ones,np.array([0.,0.,0.]))

			//arrays of integer pixel coordinates
        x = chunk % RESOLUTION[0]
        y = chunk / RESOLUTION[0]

        showprogress("Generating view vectors...",i,q)

			//the view vector in 3D space
        view = np.zeros((numChunk,3))

        view[:,0] = x.astype(float)/RESOLUTION[0] - .5
        view[:,1] = ((-y.astype(float)/RESOLUTION[1] + .5)*RESOLUTION[1])/RESOLUTION[0] #(inverting y coordinate)
        view[:,2] = 1.0

        view[:,0]*=TANFOV
        view[:,1]*=TANFOV

			//rotating through the view matrix

        view = np.einsum('jk,ik->ij',viewMatrix,view)

			//original position
        point = np.outer(ones, CAMERA_POS)

        normview = normalize(view)

        velocity = np.copy(normview)


			// initializing the colour buffer
        object_colour = np.zeros((numChunk,3))
        object_alpha = np.zeros(numChunk)

			//squared angular momentum per unit mass (in the "Newtonian fantasy")
			//h2 = np.outer(sqrnorm(np.cross(point,velocity)),np.array([1.,1.,1.]))
        h2 = sqrnorm(np.cross(point,velocity))[:,np.newaxis]

        pointsqr = np.copy(ones3)

        for it in range(NITER):
            itcounters[i]+=1

            if it%150 == 1:
                if killers[i]:
                    break
                showprogress("Raytracing...",i,q)

						// STEPPING
            oldpoint = np.copy(point) #not needed for tracing. Useful for intersections

            if METHOD == METH_LEAPFROG:
	//leapfrog method here feels good
                point += velocity * STEP

                if DISTORT:
	//this is the magical - 3/2 r^(-5) potential...
                    accel = - 1.5 * h2 *  point / np.power(sqrnorm(point),2.5)[:,np.newaxis]
                    velocity += accel * STEP

            elif METHOD == METH_RK4:
                if DISTORT:
	//simple step size control
                    rkstep = STEP

						// standard Runge-Kutta
                    y = np.zeros((numChunk,6))
                    y[:,0:3] = point
                    y[:,3:6] = velocity
                    k1 = RK4f( y, h2)
                    k2 = RK4f( y + 0.5*rkstep*k1, h2)
                    k3 = RK4f( y + 0.5*rkstep*k2, h2)
                    k4 = RK4f( y +     rkstep*k3, h2)

                    increment = rkstep/6. * (k1 + 2*k2 + 2*k3 + k4)
                    
                    velocity += increment[:,3:6]

                point += increment[:,0:3]


						//useful precalcs
            pointsqr = sqrnorm(point)
						//phi = np.arctan2(point[:,0],point[:,2])    #too heavy. Better an instance wherever it's needed.
						//normvel = normalize(velocity)              #never used! BAD BAD BAD!!


						// FOG

            if FOGDO and (it%FOGSKIP == 0):
                phsphtaper = np.clip(0.8*(pointsqr - 1.0),0.,1.0)
                fogint = np.clip(FOGMULT * FOGSKIP * STEP / pointsqr,0.0,1.0) * phsphtaper
                fogcol = ones3

                object_colour = blendcolors(fogcol,fogint,object_colour,object_alpha)
                object_alpha = blendalpha(fogint, object_alpha)


					// CHECK COLLISIONS
					// accretion disk

            if DISK_TEXTURE_INT != DT_NONE:

                mask_crossing = np.logical_xor( oldpoint[:,1] > 0., point[:,1] > 0.) #whether it just crossed the horizontal plane
					mask_distance = np.logical_and((pointsqr < DISKOUTERSQR), (pointsqr > DISKINNERSQR))  //whether it's close enough

                diskmask = np.logical_and(mask_crossing,mask_distance)

                if (diskmask.any()):
                    
                    #actual collision point by intersection
                    lambdaa = - point[:,1]/velocity[:,1]
                    colpoint = point + lambdaa[:,np.newaxis] * velocity
                    colpointsqr = sqrnorm(colpoint)

                    if DISK_TEXTURE_INT == DT_GRID:
                        phi = np.arctan2(colpoint[:,0],point[:,2])
                        theta = np.arctan2(colpoint[:,1],norm(point[:,[0,2]]))
                        diskcolor =     np.outer(
                                np.mod(phi,0.52359) < 0.261799,
                                            np.array([1.,1.,0.])
                                                ) +  \
                                        np.outer(ones,np.array([0.,0.,1.]) )
                        diskalpha = diskmask

                    elif DISK_TEXTURE_INT == DT_SOLID:
                        diskcolor = np.array([1.,1.,.98])
                        diskalpha = diskmask

                    elif DISK_TEXTURE_INT == DT_TEXTURE:

                        phi = np.arctan2(colpoint[:,0],point[:,2])
                        
                        uv = np.zeros((numChunk,2))

                        uv[:,0] = ((phi+2*np.pi)%(2*np.pi))/(2*np.pi)
                        uv[:,1] = (np.sqrt(colpointsqr)-DISKINNER)/(DISKOUTER-DISKINNER)

                        diskcolor = lookup ( texarr_disk, np.clip(uv,0.,1.))
                        #alphamask = (2.0*ransample) < sqrnorm(diskcolor)
                        #diskmask = np.logical_and(diskmask, alphamask )
                        diskalpha = diskmask * np.clip(sqrnorm(diskcolor)/3.0,0.0,1.0)

                    elif DISK_TEXTURE_INT == DT_BLACKBODY:

                        temperature = np.exp(bb.disktemp(colpointsqr,9.2103))

                        if REDSHIFT:
                            R = np.sqrt(colpointsqr)

                            disc_velocity = 0.70710678 * \
                                        np.power((np.sqrt(colpointsqr)-1.).clip(0.1),-.5)[:,np.newaxis] * \
                                        np.cross(UPFIELD, normalize(colpoint))


                            gamma =  np.power( 1 - sqrnorm(disc_velocity).clip(max=.99), -.5)

                            # opz = 1 + z
                            opz_doppler = gamma * ( 1. + np.einsum('ij,ij->i',disc_velocity,normalize(velocity)))
                            opz_gravitational = np.power(1.- 1/R.clip(1),-.5)

                            # (1+z)-redshifted Planck spectrum is still Planckian at temperature T
                            temperature /= (opz_doppler*opz_gravitational).clip(0.1)

                        intensity = bb.intensity(temperature)
                        if DISK_INTENSITY_DO:
                            diskcolor = np.einsum('ij,i->ij', bb.colour(temperature),DISK_MULTIPLIER*intensity)#np.maximum(1.*ones,DISK_MULTIPLIER*intensity))
                        else:
                            diskcolor = bb.colour(temperature)

                        iscotaper = np.clip((colpointsqr-DISKINNERSQR)*0.3,0.,1.)
                        outertaper = np.clip(temperature/1000. ,0.,1.)

                        diskalpha = diskmask * iscotaper * outertaper#np.clip(diskmask * DISK_ALPHA_MULTIPLIER *intensity,0.,1.)


                    object_colour = blendcolors(diskcolor,diskalpha,object_colour,object_alpha)
                    object_alpha = blendalpha(diskalpha, object_alpha)



								// event horizon
            oldpointsqr = sqrnorm(oldpoint)

            mask_horizon = np.logical_and((pointsqr < 1),(sqrnorm(oldpoint) > 1) )

            if mask_horizon.any() :

                lambdaa =  1. - ((1.-oldpointsqr)/((pointsqr - oldpointsqr)))[:,np.newaxis]
                colpoint = lambdaa * point + (1-lambdaa)*oldpoint

                if HORIZON_GRID:
                    phi = np.arctan2(colpoint[:,0],point[:,2])
                    theta = np.arctan2(colpoint[:,1],norm(point[:,[0,2]]))
                    horizoncolour = np.outer( np.logical_xor(np.mod(phi,1.04719) < 0.52359,np.mod(theta,1.04719) < 0.52359), np.array([1.,0.,0.]))
                else:
                    horizoncolour = BLACK#np.zeros((numPixels,3))

                horizonalpha = mask_horizon

                object_colour = blendcolors(horizoncolour,horizonalpha,object_colour,object_alpha)
                object_alpha = blendalpha(horizonalpha, object_alpha)



        showprogress("generating sky layer...",i,q)

        vphi = np.arctan2(velocity[:,0],velocity[:,2])
        vtheta = np.arctan2(velocity[:,1],norm(velocity[:,[0,2]]) )

        vuv = np.zeros((numChunk,2))

        vuv[:,0] = np.mod(vphi+4.5,2*np.pi)/(2*np.pi)
        vuv[:,1] = (vtheta+np.pi/2)/(np.pi)

        if SKY_TEXTURE_INT == DT_TEXTURE:
            col_sky = lookup(texarr_sky,vuv)[:,0:3]

        showprogress("generating debug layers...",i,q)

        //debug color: direction of view vector
        //dbg_viewvec = np.clip(view + vec3(.5,.5,0.0),0.0,1.0)
        //debug color: direction of final ray
        //debug color: grid
        //dbg_grid = np.abs(normalize(velocity)) < 0.1


        if SKY_TEXTURE_INT == ST_TEXTURE:
            col_bg = col_sky
        elif SKY_TEXTURE_INT == ST_NONE:
            col_bg = np.zeros((numChunk,3))
        elif SKY_TEXTURE_INT == ST_FINAL:
            dbg_finvec = np.clip(normalize(velocity) + np.array([.5,.5,0.0])[np.newaxis,:],0.0,1.0)
            col_bg = dbg_finvec
        else:
            col_bg = np.zeros((numChunk,3))


        showprogress("blending layers...",i,q)

        col_bg_and_obj = blendcolors(SKYDISK_RATIO*col_bg, ones ,object_colour,object_alpha)

        showprogress("beaming back to mothership.",i,q)
				// copy back in the buffer
        if not DISABLE_SHUFFLING:
            total_colour_buffer_preproc[chunk] = col_bg_and_obj
        else:
            total_colour_buffer_preproc[chunk[0]:(chunk[-1]+1)] = col_bg_and_obj


				//refresh display
				// NO: plt does not allow drawing outside main thread
				//if not DISABLE_DISPLAY:
				//    showprogress("updating display...")
				//    plt.imshow(total_colour_buffer_preproc.reshape((RESOLUTION[1],RESOLUTION[0],3)))
				//    plt.draw()

        showprogress("garbage collection...",i,q)
        gc.collect()

    showprogress("Done.",i,q)

				// Threading

process_list = []
for i in range(NTHREADS):
    p = multi.Process(target=raytrace_schedule,args=(i,schedules[i],total_colour_buffer_preproc_shared,output.queue))
    process_list.append(p)

logger.debug("Starting threads...")

for proc in process_list:
    proc.start()

try:
    refreshcounter = 0
    while True:
        refreshcounter+=1
        time.sleep(0.1)
    
        output.parsemessages()

        if not DISABLE_DISPLAY and (refreshcounter%40 == 0):
            output.setmessage("Updating display...",-1)
            plt.imshow(total_colour_buffer_preproc.reshape((RESOLUTION[1],RESOLUTION[0],3)))
            plt.draw()

        output.setmessage("Idle.", -1)

        alldone = True
        for i in range(NTHREADS):
            if process_list[i].is_alive():
                alldone = False
        if alldone:
            break
except KeyboardInterrupt:
    for i in range(NTHREADS):
        killers[i] = True
    sys.exit()

del output


logger.debug("Done tracing.")

logger.debug("Total raytracing time: %s", datetime.timedelta(seconds=(time.time() - start_time)))


logger.debug("Postprocessing...")

			//gain
logger.debug("- gain...")
total_colour_buffer_preproc *= GAIN


			// airy bloom
if AIRY_BLOOM:

    logger.debug("-computing Airy disk bloom...")
    
		//blending bloom

		//colour = total_colour_buffer_preproc + 0.3*blurd #0.2*dbg_grid + 0.8*dbg_finvec
    
		//airy disk bloom

    colour_bloomd = np.copy(total_colour_buffer_preproc)
    colour_bloomd = colour_bloomd.reshape((RESOLUTION[1],RESOLUTION[0],3))


		// the float constant is 1.22 * 650nm / (4 mm), the typical diffractive resolution
		// of the human eye for red light. It's in radians, so we rescale using field of view.
    radd = 0.00019825 * RESOLUTION[0] / np.arctan(TANFOV)

		// the user is allowed to rescale the resolution, though
    radd*=AIRY_RADIUS 

		// the pixel size of the kernel:
		// 25 pixels radius is ok for 5.0 bright source pixel at 1920x1080, so...
		// remembering that airy ~ 1/x^3, so if we want intensity/x^3 < hreshold => 
		// => max_x = (intensity/threshold)^1/3
		// so it scales with 
		// - the cube root of maximum intensity
		// - linear in resolution

    mxint = np.amax(colour_bloomd)

    kern_radius = 25 * np.power( np.amax(colour_bloomd) / 5.0 , 1./3.) * RESOLUTION[0]/1920.

    logger.debug("--(radius: %3f, kernel pixel radius: %3f, maximum source brightness: %3f)", radd, kern_radius, mxint)
    
    colour_bloomd = bloom.airy_convolve(colour_bloomd,radd)
 
    colour_bloomd = colour_bloomd.reshape((numPixels,3))


    colour_pb = colour_bloomd
else:
    colour_pb = total_colour_buffer_preproc


		// wide gaussian (lighting dust effect)

if BLURDO:

    logger.debug("-computing wide gaussian blur...")
    
		//hipass = np.outer(sqrnorm(total_colour_buffer_preproc) > BLOOMCUT, np.array([1.,1.,1.])) * total_colour_buffer_preproc
    blurd = np.copy(total_colour_buffer_preproc)

    blurd = blurd.reshape((RESOLUTION[1],RESOLUTION[0],3))

    for i in range(2):
        logger.debug("- gaussian blur pass %d...", i)
        blurd = ndim.gaussian_filter(blurd,int(0.05*RESOLUTION[0]))

    blurd = blurd.reshape((numPixels,3))
    colour = colour_pb + 0.2 * blurd
else:
    colour = colour_pb



		//normalization
if NORMALIZE > 0:
    logger.debug("- normalizing...")
    colour *= 1 / (NORMALIZE * np.amax(colour.flatten()) )




		//final colour
colour = np.clip(colour,0.,1.)


logger.debug("Conversion to image and saving...")

saveToImg(colour,"tests/out.png")
saveToImg(total_colour_buffer_preproc,"tests/preproc.png")
if BLURDO:
    saveToImg(colour_pb,"tests/postbloom.png")
