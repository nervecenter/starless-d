module starless.types;

struct Resolution
{
	int w = 800;
	int h = 600;
}

struct Vector3
{
	double x, y, z;

	Vector3 opUnary(string s)() if (s == "-")
	{
		return Vector3(-x, -y, -z);
	}

	Vector3 opBinary(string op)(Vector3 rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			return mixin("Vector3(x "~op~" rhs.x, y "~op~" rhs.y, z "~op~" rhs.z)");
		else
			static assert(0, "Operator "~op~" not implemented for type 'Vector3'");
	}

	Vector3 opBinary(string op)(double rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			return mixin("Vector3(x "~op~" rhs, y "~op~" rhs, z "~op~" rhs)");
		else
			static assert(0, "Operator "~op~" not implemented for type 'Vector3'");
	}

	Vector3 opBinaryRight(string op)(double lhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			return mixin("Vector3(x "~op~" lhs, y "~op~" lhs, z "~op~" lhs)");
		else
			static assert(0, "Operator "~op~" not implemented for type 'Vector3'");
	}

	void opOpAssign(string op)(Vector3 rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			mixin("x = x "~op~" rhs.x; y = y "~op~" rhs.y; z = z "~op~" rhs.z;");
		else
			static assert(0, "Operator "~op~" not implemented for type 'Vector3'");
	}

	void opOpAssign(string op)(double rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			mixin("x = x "~op~" rhs; y = y "~op~" rhs; z = z "~op~" rhs;");
		else
			static assert(0, "Operator "~op~" not implemented for type 'Vector3'");
	}
}

struct RGB
{
	double r, g, b;

	RGB opUnary(string s)() if (s == "-")
	{
		return RGB(-r, -g, -b);
	}

	RGB opBinary(string op)(RGB rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			return mixin("RGB(r "~op~" rhs.r, g "~op~" rhs.g, b "~op~" rhs.b)");
		else
			static assert(0, "Operator "~op~" not implemented for type 'RGB'");
	}

	RGB opBinary(string op)(double rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			return mixin("RGB(r "~op~" rhs, g "~op~" rhs, b "~op~" rhs)");
		else
			static assert(0, "Operator "~op~" not implemented for type 'RGB'");
	}

	RGB opBinaryRight(string op)(double lhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			return mixin("RGB(r "~op~" lhs, g "~op~" lhs, b "~op~" lhs)");
		else
			static assert(0, "Operator "~op~" not implemented for type 'RGB'");
	}

	void opOpAssign(string op)(RGB rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			mixin("r = r "~op~" rhs.r; g = g "~op~" rhs.g; b = b "~op~" rhs.b;");
		else
			static assert(0, "Operator "~op~" not implemented for type 'RGB'");
	}

	void opOpAssign(string op)(double rhs)
	{
		static if (op == "+" || op == "-" || op == "*" || op == "/")
			mixin("r = r "~op~" rhs; g = g "~op~" rhs; b = b "~op~" rhs;");
		else
			static assert(0, "Operator "~op~" not implemented for type 'RGB'");
	}
}

debug
{
	import std.stdio;
	void main()
	{
		RGB color1 = RGB(123.0, 20.0, 200.0);
		RGB color2 = RGB(70.0, 20.0, 225.0);
		double adouble = 10.0;

		writeln(color1 + color2);
		writeln(color1 - color2);
		writeln(color1 * color2);
		writeln(color1 / color2, "\n");

		writeln(color2 + color1);
		writeln(color2 - color1);
		writeln(color2 * color1);
		writeln(color2 / color1, "\n");

		writeln(color1 + adouble);
		writeln(color1 - adouble);
		writeln(color1 * adouble);
		writeln(color1 / adouble, "\n");

		writeln(adouble + color2);
		writeln(adouble - color2);
		writeln(adouble * color2);
		writeln(adouble / color2, "\n");

		color1 += color2;
		color2 += adouble;
		writeln(color1);
		writeln(color2);
	}
}
