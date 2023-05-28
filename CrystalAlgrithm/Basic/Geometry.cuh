/*
Copyright (C) <2023>  <Dezeming>  <feimos@mail.ustc.edu.cn>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any
later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Github site: <https://github.com/feimos32/Crystal>
*/

/*
The following code is partially rewritten from: 
PBRT v3: <https://pbrt.org/>
*/

#ifndef __Geometry_h__
#define __Geometry_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>

//#include <cmath>
//#include <algorithm>

#include "Common.cuh"

// statement
namespace CrystalAlgrithm {
	class Vector3f;
	class Point3f;
	class Normal3f;

	using std::isnan;
	using std::min;
	using std::max;
	using std::abs;
	using std::swap;

}

// Vector3f/Point3f/Normal3f
namespace CrystalAlgrithm {


class Vector3f {
public:
	// Vector3f Public Methods
	HOST_AND_DEVICE Float operator[](int i) const {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}
	HOST_AND_DEVICE Float& operator[](int i) {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}
	HOST_AND_DEVICE Vector3f() { x = y = z = 0; }
	HOST_AND_DEVICE Vector3f(Float x, Float y, Float z) : x(x), y(y), z(z) { }
	HOST bool HasNaNs() const { return isnan(x) || isnan(y) || isnan(z); }
	HOST_AND_DEVICE explicit Vector3f(const Point3f& p);
	HOST_AND_DEVICE Vector3f operator+(const Vector3f& v) const {
		return Vector3f(x + v.x, y + v.y, z + v.z);
	}
	HOST_AND_DEVICE Vector3f& operator+=(const Vector3f& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	HOST_AND_DEVICE Vector3f operator-(const Vector3f& v) const {
		return Vector3f(x - v.x, y - v.y, z - v.z);
	}
	HOST_AND_DEVICE Vector3f& operator-=(const Vector3f& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	HOST_AND_DEVICE bool operator==(const Vector3f& v) const {
		return x == v.x && y == v.y && z == v.z;
	}
	HOST_AND_DEVICE bool operator!=(const Vector3f& v) const {
		return x != v.x || y != v.y || z != v.z;
	}
	HOST_AND_DEVICE Vector3f operator*(Float s) const {
		return Vector3f(s * x, s * y, s * z);
	}
	HOST_AND_DEVICE Vector3f& operator*=(Float s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}
	HOST_AND_DEVICE Vector3f operator/(Float f) const {
		Float inv = (Float)1 / f;
		return Vector3f(x * inv, y * inv, z * inv);
	}

	HOST_AND_DEVICE Vector3f& operator/=(Float f) {
		Float inv = (Float)1 / f;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}
	HOST_AND_DEVICE Vector3f operator-() const { return Vector3f(-x, -y, -z); }
	HOST_AND_DEVICE Float LengthSquared() const { return x * x + y * y + z * z; }
	HOST_AND_DEVICE Float Length() const { return sqrt(LengthSquared()); }
	HOST_AND_DEVICE explicit Vector3f(const Normal3f& n);

	// Vector3f Public Data
	Float x, y, z;
};

class Point3f {
public:
	// Point3f Public Methods
	HOST_AND_DEVICE Point3f() { x = y = z = 0; }
	HOST_AND_DEVICE Point3f(Float x, Float y, Float z) : x(x), y(y), z(z) { }

	HOST_AND_DEVICE Point3f(const Point3f& p)
		: x(p.x), y(p.y), z(p.z) {
	}

	HOST_AND_DEVICE explicit operator Vector3f() const {
		return Vector3f(x, y, z);
	}
	HOST_AND_DEVICE Point3f operator+(const Vector3f& v) const {
		return Point3f(x + v.x, y + v.y, z + v.z);
	}
	HOST_AND_DEVICE Point3f& operator+=(const Vector3f& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	HOST_AND_DEVICE Vector3f operator-(const Point3f& p) const {
		return Vector3f(x - p.x, y - p.y, z - p.z);
	}
	HOST_AND_DEVICE Point3f operator-(const Vector3f& v) const {
		return Point3f(x - v.x, y - v.y, z - v.z);
	}
	HOST_AND_DEVICE Point3f& operator-=(const Vector3f& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	HOST_AND_DEVICE Point3f& operator+=(const Point3f& p) {
		x += p.x;
		y += p.y;
		z += p.z;
		return *this;
	}
	HOST_AND_DEVICE Point3f operator+(const Point3f& p) const {
		return Point3f(x + p.x, y + p.y, z + p.z);
	}

	HOST_AND_DEVICE Point3f operator*(Float f) const {
		return Point3f(f * x, f * y, f * z);
	}

	HOST_AND_DEVICE Point3f& operator*=(Float f) {
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}

	HOST_AND_DEVICE Point3f operator/(Float f) const {
		Float inv = (Float)1 / f;
		return Point3f(inv * x, inv * y, inv * z);
	}

	HOST_AND_DEVICE Point3f& operator/=(Float f) {
		Float inv = (Float)1 / f;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}
	HOST_AND_DEVICE Float operator[](int i) const {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}

	HOST_AND_DEVICE Float& operator[](int i) {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}
	HOST_AND_DEVICE bool operator==(const Point3f& p) const {
		return x == p.x && y == p.y && z == p.z;
	}
	HOST_AND_DEVICE bool operator!=(const Point3f& p) const {
		return x != p.x || y != p.y || z != p.z;
	}
	HOST bool HasNaNs() const { return isnan(x) || isnan(y) || isnan(z); }
	HOST_AND_DEVICE Point3f operator-() const { return Point3f(-x, -y, -z); }

	// Point3f Public Data
	Float x, y, z;
};

class Normal3f {
public:
	// Normal3f Public Methods
	HOST_AND_DEVICE Normal3f() { x = y = z = 0; }
	HOST_AND_DEVICE Normal3f(Float xx, Float yy, Float zz) : x(xx), y(yy), z(zz) { }
	HOST_AND_DEVICE Normal3f operator-() const { return Normal3f(-x, -y, -z); }
	HOST_AND_DEVICE Normal3f operator+(const Normal3f& n) const {
		return Normal3f(x + n.x, y + n.y, z + n.z);
	}

	HOST_AND_DEVICE Normal3f& operator+=(const Normal3f& n) {
		x += n.x;
		y += n.y;
		z += n.z;
		return *this;
	}
	HOST_AND_DEVICE Normal3f operator-(const Normal3f& n) const {
		return Normal3f(x - n.x, y - n.y, z - n.z);
	}

	HOST_AND_DEVICE Normal3f& operator-=(const Normal3f& n) {
		x -= n.x;
		y -= n.y;
		z -= n.z;
		return *this;
	}
	HOST bool HasNaNs() const { return isnan(x) || isnan(y) || isnan(z); }

	HOST_AND_DEVICE Normal3f operator*(Float f) const {
		return Normal3f(f * x, f * y, f * z);
	}

	HOST_AND_DEVICE Normal3f& operator*=(Float f) {
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	HOST_AND_DEVICE Normal3f operator/(Float f) const {
		Float inv = (Float)1 / f;
		return Normal3f(x * inv, y * inv, z * inv);
	}

	HOST_AND_DEVICE Normal3f& operator/=(Float f) {
		Float inv = (Float)1 / f;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}
	HOST_AND_DEVICE Float LengthSquared() const { return x * x + y * y + z * z; }
	HOST_AND_DEVICE Float Length() const { return sqrt(LengthSquared()); }

	HOST_AND_DEVICE Normal3f(const Normal3f& n) {
		x = n.x;
		y = n.y;
		z = n.z;
	}

	HOST_AND_DEVICE Normal3f& operator=(const Normal3f& n) {
		x = n.x;
		y = n.y;
		z = n.z;
		return *this;
	}

	HOST_AND_DEVICE explicit Normal3f(const Vector3f& v) : x(v.x), y(v.y), z(v.z) {}
	HOST_AND_DEVICE bool operator==(const Normal3f& n) const {
		return x == n.x && y == n.y && z == n.z;
	}
	HOST_AND_DEVICE bool operator!=(const Normal3f& n) const {
		return x != n.x || y != n.y || z != n.z;
	}

	HOST_AND_DEVICE Float operator[](int i) const {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}

	HOST_AND_DEVICE Float& operator[](int i) {
		if (i == 0) return x;
		if (i == 1) return y;
		return z;
	}

	// Normal3f Public Data
	Float x, y, z;
};

// Geometry Inline Functions
HOST_AND_DEVICE
	inline Vector3f::Vector3f(const Point3f& p)
	: x(p.x), y(p.y), z(p.z) {}

HOST_AND_DEVICE
	inline Vector3f operator*(const Vector3f& v, const Vector3f& w) {
	return Vector3f(v.x * w.x, v.y * w.y, v.z * w.z);
}

HOST_AND_DEVICE
	inline Vector3f operator*(Float s, const Vector3f& v) {
	return v * s;
}

HOST_AND_DEVICE
	inline Vector3f Abs(const Vector3f& v) {
	return Vector3f(abs(v.x), abs(v.y), abs(v.z));
}

HOST_AND_DEVICE
	inline Float Dot(const Vector3f& v1, const Vector3f& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

HOST_AND_DEVICE
	inline Float AbsDot(const Vector3f& v1, const Vector3f& v2) {
	return abs(Dot(v1, v2));
}

HOST_AND_DEVICE
	inline Vector3f Cross(const Vector3f& v1, const Vector3f& v2) {
	double v1x = v1.x, v1y = v1.y, v1z = v1.z;
	double v2x = v2.x, v2y = v2.y, v2z = v2.z;
	return Vector3f((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
		(v1x * v2y) - (v1y * v2x));
}

HOST_AND_DEVICE
	inline Vector3f Cross(const Vector3f& v1, const Normal3f& v2) {
	double v1x = v1.x, v1y = v1.y, v1z = v1.z;
	double v2x = v2.x, v2y = v2.y, v2z = v2.z;
	return Vector3f((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
		(v1x * v2y) - (v1y * v2x));
}

HOST_AND_DEVICE
	inline Vector3f Cross(const Normal3f& v1, const Vector3f& v2) {
	double v1x = v1.x, v1y = v1.y, v1z = v1.z;
	double v2x = v2.x, v2y = v2.y, v2z = v2.z;
	return Vector3f((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
		(v1x * v2y) - (v1y * v2x));
}

HOST_AND_DEVICE
	inline Vector3f Normalize(const Vector3f& v) {
	return v / v.Length();
}
HOST_AND_DEVICE
	inline Float MinComponent(const Vector3f& v) {
	return min(v.x, min(v.y, v.z));
}

HOST_AND_DEVICE
	inline Float MaxComponent(const Vector3f& v) {
	return max(v.x, max(v.y, v.z));
}


HOST_AND_DEVICE
	inline int MaxDimension(const Vector3f& v) {
	return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

HOST_AND_DEVICE
	inline Vector3f Min(const Vector3f& p1, const Vector3f& p2) {
	return Vector3f(min(p1.x, p2.x), min(p1.y, p2.y),
		min(p1.z, p2.z));
}

HOST_AND_DEVICE
	inline Vector3f Max(const Vector3f& p1, const Vector3f& p2) {
	return Vector3f(max(p1.x, p2.x), max(p1.y, p2.y),
		max(p1.z, p2.z));
}

HOST_AND_DEVICE
	inline Vector3f Permute(const Vector3f& v, int x, int y, int z) {
	return Vector3f(v[x], v[y], v[z]);
}

HOST_AND_DEVICE
	inline Float MinComponent(const Point3f& v) {
	return min(v.x, min(v.y, v.z));
}
HOST_AND_DEVICE
	inline Float MaxComponent(const Point3f& v) {
	return max(v.x, max(v.y, v.z));
}

HOST_AND_DEVICE
	inline Float Distance(const Point3f& p1, const Point3f& p2) {
	return (p1 - p2).Length();
}

HOST_AND_DEVICE
	inline Float DistanceSquared(const Point3f& p1, const Point3f& p2) {
	return (p1 - p2).LengthSquared();
}

HOST_AND_DEVICE
	inline Point3f operator*(Float f, const Point3f& p) {
	return p * f;
}

HOST_AND_DEVICE
	inline Point3f Lerp(Float t, const Point3f& p0, const Point3f& p1) {
	return (1 - t) * p0 + t * p1;
}

HOST_AND_DEVICE
	inline Point3f Min(const Point3f& p1, const Point3f& p2) {
	return Point3f(min(p1.x, p2.x), min(p1.y, p2.y),
		min(p1.z, p2.z));
}

HOST_AND_DEVICE
	inline Point3f Max(const Point3f& p1, const Point3f& p2) {
	return Point3f(max(p1.x, p2.x), max(p1.y, p2.y),
		max(p1.z, p2.z));
}

HOST inline Point3f Floor(const Point3f& p) {
	return Point3f(std::floor(p.x), std::floor(p.y), std::floor(p.z));
}

HOST inline Point3f Ceil(const Point3f& p) {
	return Point3f(std::ceil(p.x), std::ceil(p.y), std::ceil(p.z));
}

HOST_AND_DEVICE
	inline Point3f Abs(const Point3f& p) {
	return Point3f(abs(p.x), abs(p.y), abs(p.z));
}

HOST_AND_DEVICE
	inline Point3f Permute(const Point3f& p, int x, int y, int z) {
	return Point3f(p[x], p[y], p[z]);
}

HOST_AND_DEVICE
	inline void CoordinateSystem(const Vector3f& v1, Vector3f* v2,
		Vector3f* v3) {
	if (abs(v1.x) > abs(v1.y))
		*v2 = Vector3f(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
	else
		*v2 = Vector3f(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
	*v3 = Cross(v1, *v2);
}

HOST_AND_DEVICE
	inline Normal3f operator*(Float f, const Normal3f& n) {
	return Normal3f(f * n.x, f * n.y, f * n.z);
}

HOST_AND_DEVICE
	inline Normal3f Normalize(const Normal3f& n) {
	return n / n.Length();
}

HOST_AND_DEVICE
	inline Vector3f::Vector3f(const Normal3f& n)
	: x(n.x), y(n.y), z(n.z) {
}

HOST_AND_DEVICE
	inline Float Dot(const Normal3f& n1, const Vector3f& v2) {
	return n1.x * v2.x + n1.y * v2.y + n1.z * v2.z;
}

HOST_AND_DEVICE
	inline Float Dot(const Vector3f& v1, const Normal3f& n2) {
	return v1.x * n2.x + v1.y * n2.y + v1.z * n2.z;
}

HOST_AND_DEVICE
	inline Float Dot(const Normal3f& n1, const Normal3f& n2) {
	return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}

HOST_AND_DEVICE
	inline Float AbsDot(const Normal3f& n1, const Vector3f& v2) {
	return abs(n1.x * v2.x + n1.y * v2.y + n1.z * v2.z);
}

HOST_AND_DEVICE
	inline Float AbsDot(const Vector3f& v1, const Normal3f& n2) {
	return abs(v1.x * n2.x + v1.y * n2.y + v1.z * n2.z);
}

HOST_AND_DEVICE
	inline Float AbsDot(const Normal3f& n1, const Normal3f& n2) {
	return abs(n1.x * n2.x + n1.y * n2.y + n1.z * n2.z);
}

HOST_AND_DEVICE
	inline Normal3f Faceforward(const Normal3f& n, const Vector3f& v) {
	return (Dot(n, v) < 0.f) ? -n : n;
}

HOST_AND_DEVICE
	inline Normal3f Faceforward(const Normal3f& n, const Normal3f& n2) {
	return (Dot(n, n2) < 0.f) ? -n : n;
}

HOST_AND_DEVICE
	inline Vector3f Faceforward(const Vector3f& v, const Vector3f& v2) {
	return (Dot(v, v2) < 0.f) ? -v : v;
}

HOST_AND_DEVICE
	inline Vector3f Faceforward(const Vector3f& v, const Normal3f& n2) {
	return (Dot(v, n2) < 0.f) ? -v : v;
}

HOST_AND_DEVICE
	inline Normal3f Abs(const Normal3f& v) {
	return Normal3f(abs(v.x), abs(v.y), abs(v.z));
}


}

// Vector2f/Point2f/Normal2f
namespace CrystalAlgrithm {

}







#endif



