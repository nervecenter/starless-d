module starless.logger;

import std.stdio,
	std.file,
	std.datetime;
import std.format : format;

class Logger
{
	private static File logfile;
	private shared static Logger _instance;
	
	this()
	{
		if (!exists("logs"))
			mkdir("logs");
		
		logfile = File("logs/" ~ Clock.currTime().toSimpleString() ~ ".log", "w");
	}

	static @property ref shared(Logger) instance()
	{
		if (_instance is null)
			_instance = new Logger();

		return _instance;
	}
	
	shared void
	log(string msg)
	{
		writeln(msg);
		logfile.writeln(msg);
	}

	shared void
	error(string msg)
	{
		logfile.writeln("ERROR: ", msg);
		throw new Exception(message);
	}
}
		
