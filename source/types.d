module starless.types;

import core.exception : RangeError;

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

	double opIndex(size_t index)
	{
		if (index == 0) return x;
		else if (index == 1) return y;
		else if (index == 2) return z;
		else throw new RangeError("Type 'Vector3' is only indexed from 0 to 2.");
	}

	double opIndexAssign(double value, size_t index)
	{
		if (index == 0) return x = value;
		else if (index == 1) return y = value;
		else if (index == 2) return z = value;
		else throw new RangeError("Type 'Vector3' is only indexed from 0 to 2.");
	}

	int opApply(int delegate(ref const(double)) dg) const {
		dg(x);
		dg(y);
		return dg(z);
    }

	int opApply(int delegate(int, const(double)) dg) const {
		dg(0, x);
		dg(1, y);
		return dg(2, z);
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

	double opIndex(size_t index)
	{
		if (index == 0) return r;
		else if (index == 1) return g;
		else if (index == 2) return b;
		else throw new RangeError("Type 'RGB' is only indexed from 0 to 2.");
	}

	double opIndexAssign(double value, size_t index)
	{
		if (index == 0) return r = value;
		else if (index == 1) return g = value;
		else if (index == 2) return b = value;
		else throw new RangeError("Type 'RGB' is only indexed from 0 to 2.");
	}

	int opApply(int delegate(ref const(double)) dg) const {
		dg(r);
		dg(g);
		return dg(b);
    }

	int opApply(int delegate (int, const(double)) dg) const {
		dg(0, r);
		dg(1, g);
		return dg(2, b);
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
		writeln(color2, "\n");

		color1[1] = 325.0;
		writeln(color1[1]);
		writeln(color1);

		foreach (i, val; color1)
			writeln(i, ", ", val);

		foreach (i, val; color2)
			writeln(i, ", ", val);
	}
}
