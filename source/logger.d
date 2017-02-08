module starless.logger;

import std.stdio,
	std.file,
	std.datetime;
import std.format : format;

class Logger
{
	File logfile;
	
	this()
	{
		if (!exists("logs"))
			mkdir("logs");
		
		logfile = File("logs/" ~ Clock.currTime().toSimpleString() ~ ".log", "w");
	}
	
	void
	debug(Char, A...)(in Char[] fmt, A args)
	{
		string message = format(fmt, args);
		writeln(message);
		logfile.writeln(message);
	}

	void
	error(string msg)
	{
		logfile.writeln("ERROR: ", msg);
		throw new Exception(message);
	}
}
		
