/*
 *  j1.d - D implementation of the Bessel function of the first kind
 *  Translated from the Cephes math library by Stephen L. Moshier
 *
 *  translated by Chris Collazo for starless-d
 */

module starless.j1;

import std.math;

static double[4] RP =
	[
		-8.99971225705559398224E8,
		4.52228297998194034323E11,
		-7.27494245221818276015E13,
		3.68295732863852883286E15,
	];
static double[8] RQ =
	[
		6.20836478118054335476E2,
		2.56987256757748830383E5,
		8.35146791431949253037E7,
		2.21511595479792499675E10,
		4.74914122079991414898E12,
		7.84369607876235854894E14,
		8.95222336184627338078E16,
		5.32278620332680085395E18,
	];

static double[7] PP =
	[
		7.62125616208173112003E-4,
		7.31397056940917570436E-2,
		1.12719608129684925192E0,
		5.11207951146807644818E0,
		8.42404590141772420927E0,
		5.21451598682361504063E0,
		1.00000000000000000254E0,
	];
static double[7] PQ =
	[
		5.71323128072548699714E-4,
		6.88455908754495404082E-2,
		1.10514232634061696926E0,
		5.07386386128601488557E0,
		8.39985554327604159757E0,
		5.20982848682361821619E0,
		9.99999999999999997461E-1,
	];
static double[8] QP =
	[
		5.10862594750176621635E-2,
		4.98213872951233449420E0,
		7.58238284132545283818E1,
		3.66779609360150777800E2,
		7.10856304998926107277E2,
		5.97489612400613639965E2,
		2.11688757100572135698E2,
		2.52070205858023719784E1,
	];
static double[7] QQ =
	[
		7.42373277035675149943E1,
		1.05644886038262816351E3,
		4.98641058337653607651E3,
		9.56231892404756170795E3,
		7.99704160447350683650E3,
		2.82619278517639096600E3,
		3.36093607810698293419E2,
	];

static double Z1 = 1.46819706421238932572E1;
static double Z2 = 4.92184563216946036703E1;

// 3*pi/4
double THPIO4 = 2.35619449019234492885;       
// sqrt( 2/pi ) 
double SQ2OPI = 7.9788456080286535587989E-1;  

double polevl(double x, double[] coef, int N)
{
	double ans;
	int i;
	double *p;

	p = coef.ptr;
	ans = *p++;
	i = N;

	do
		ans = ans * x + *p++;
	while(--i);

	return(ans);
}

double p1evl(double x, double[] coef, int N)
{
	double ans;
	double *p;
	int i;
	
	p = coef.ptr;
	ans = x + *p++;
	i = N-1;

	do
		ans = ans * x + *p++;
	while( --i );

	return( ans );
}

double
j1(double x)
{
	double w, z, p, q, xn;

	w = x;
	
	if(x < 0)
		w = -x;

	if(w <= 5.0)
	{
		z = x * x;	
		w = polevl(z, RP, 3) / p1evl(z, RQ, 8);
		w = w * x * (z - Z1) * (z - Z2);
		return(w);
	}

	w = 5.0/x;
	z = w * w;
	p = polevl(z, PP, 6)/polevl(z, PQ, 6);
	q = polevl(z, QP, 7)/p1evl(z, QQ, 7);
	xn = x - THPIO4;
	p = p * cos(xn) - w * q * sin(xn);
	return(p * SQ2OPI / sqrt(x));
}
