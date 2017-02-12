module starless.graph;

import starless.logger,
	starless.types,
	starless.options,
	starless.functions,
	ggplotd.aes,
	ggplotd.ggplotd,
	ggplotd.geom;

void
drawGraph(Options options)
{
	if (!options.drawGraph) return;
	
	auto logger = Logger.instance;
	logger.log("Drawing schematic graph...");
	auto gg = GGPlotD();
	
	//plt.Circle(centerpt, radius, fc=facecolor)
	//g_diskout       = plt.Circle((0,0),DISKOUTER, fc='0.75');
	gg = aes!("x", "y", "size", "colour")
		(0.0, 0.0, options.geometry.diskOuter, 0.75)
		.geomEllipse().putIn(gg);
	
	//g_diskin        = plt.Circle((0,0),DISKINNER, fc='white');
	gg = aes!("x", "y", "size", "colour")
		(0.0, 0.0, options.geometry.diskInner, "white")
		.geomEllipse().putIn(gg);
	
	
	//g_photon        = plt.Circle((0,0),1.5,ec='y',fc='none');
	gg = aes!("x", "y", "size", "colour", "fill")
		(0.0, 0.0, 1.5, "yellow", 0.0)
		.geomEllipse().putIn(gg);
	
	//g_horizon       = plt.Circle((0,0),1,color='black');
	gg = aes!("x", "y", "size", "colour")
		(0.0, 0.0, 1.0, "black")
		.geomEllipse().putIn(gg);
	
	Vector3 cam_pos = options.geometry.cameraPos;
	
	//g_cameraball    = plt.Circle((CAMERA_POS[2],CAMERA_POS[0]),0.2,color='black');
	gg = aes!("x", "y", "size", "colour")
		(cam_pos.x, cam_pos.z, 0.2, "black")
		.geomEllipse().putIn(gg);
	
	double gscale = 1.1 * norm(cam_pos);
	gg.put(xaxisRange(-gscale, gscale));
	gg.put(yaxisRange(-gscale, gscale));
	
	// draw line from camera to lookat point
	Vector3 lookat = options.geometry.lookAt;
	auto aes =
		Aes!(double[], "x", double[], "y", double, "colour")
		([cam_pos.x, lookat.x], [cam_pos.z, lookat.z], 0.05);
	/*ax.plot([CAMERA_POS[2],LOOKAT[2]],
	  [CAMERA_POS[0],LOOKAT[0]],
	  color='0.05',
	  linestyle='-');*/
	
	logger.log("Saving diagram...");
	gg.save("tests/graph.png");
}
