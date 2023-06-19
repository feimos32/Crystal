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


#ifndef __Transform_cuh__
#define __Transform_cuh__

#include <cuda.h>
#include <cuda_runtime.h>

#include "Geometry.cuh"
#include "Common.cuh"

namespace CrystalAlgrithm {

struct Matrix4x4;
class Transform;

inline constexpr float Radians(float deg) { return (3.14159265f / 180) * deg; }

// Matrix4x4 Declarations
struct EXPORT_DLL Matrix4x4 {
	// Matrix4x4 Public Methods
	HOST_AND_DEVICE Matrix4x4() {
		m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
		m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
			m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
	}
	HOST_AND_DEVICE Matrix4x4(float mat[4][4]);
	HOST_AND_DEVICE Matrix4x4(float t00, float t01, float t02, float t03, float t10, float t11,
		float t12, float t13, float t20, float t21, float t22, float t23,
		float t30, float t31, float t32, float t33);
	HOST_AND_DEVICE bool operator==(const Matrix4x4& m2) const {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				if (m[i][j] != m2.m[i][j]) return false;
		return true;
	}
	HOST_AND_DEVICE bool operator!=(const Matrix4x4& m2) const {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				if (m[i][j] != m2.m[i][j]) return true;
		return false;
	}
	HOST_AND_DEVICE friend Matrix4x4 Transpose(const Matrix4x4&);
	HOST_AND_DEVICE static Matrix4x4 Mul(const Matrix4x4& m1, const Matrix4x4& m2) {
		Matrix4x4 r;
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
				m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
		return r;
	}
	HOST_AND_DEVICE friend Matrix4x4 Inverse(const Matrix4x4&, bool& flag);
	float m[4][4];
};

class EXPORT_DLL Transform {
public:
	// Transform Public Methods
	HOST_AND_DEVICE Transform() { invFlag = true; }
	HOST_AND_DEVICE Transform(const float mat[4][4]) {
		m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
			mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
			mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
			mat[3][3]);
		mInv = Inverse(m, invFlag);
	}
	HOST_AND_DEVICE Transform(const Matrix4x4& m) : m(m), mInv(Inverse(m, invFlag)) {}
	HOST_AND_DEVICE Transform(const Matrix4x4& m, const Matrix4x4& mInv) : m(m), mInv(mInv) {}
	HOST_AND_DEVICE friend Transform Inverse(const Transform& t) {
		return Transform(t.mInv, t.m);
	}
	HOST_AND_DEVICE friend Transform Transpose(const Transform& t) {
		return Transform(Transpose(t.m), Transpose(t.mInv));
	}
	HOST_AND_DEVICE bool operator==(const Transform& t) const {
		return t.m == m && t.mInv == mInv;
	}
	HOST_AND_DEVICE bool operator!=(const Transform& t) const {
		return t.m != m || t.mInv != mInv;
	}
	HOST_AND_DEVICE bool operator<(const Transform& t2) const {
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j) {
				if (m.m[i][j] < t2.m.m[i][j]) return true;
				if (m.m[i][j] > t2.m.m[i][j]) return false;
			}
		return false;
	}
	HOST_AND_DEVICE bool IsIdentity() const {
		return (m.m[0][0] == 1.f && m.m[0][1] == 0.f && m.m[0][2] == 0.f &&
			m.m[0][3] == 0.f && m.m[1][0] == 0.f && m.m[1][1] == 1.f &&
			m.m[1][2] == 0.f && m.m[1][3] == 0.f && m.m[2][0] == 0.f &&
			m.m[2][1] == 0.f && m.m[2][2] == 1.f && m.m[2][3] == 0.f &&
			m.m[3][0] == 0.f && m.m[3][1] == 0.f && m.m[3][2] == 0.f &&
			m.m[3][3] == 1.f);
	}
	HOST_AND_DEVICE const Matrix4x4& GetMatrix() const { return m; }
	HOST_AND_DEVICE const Matrix4x4& GetInverseMatrix() const { return mInv; }
	HOST_AND_DEVICE bool HasScale() const {
		float la2 = (*this)(Vector3f(1, 0, 0)).LengthSquared();
		float lb2 = (*this)(Vector3f(0, 1, 0)).LengthSquared();
		float lc2 = (*this)(Vector3f(0, 0, 1)).LengthSquared();
#define NOT_ONE(x) ((x) < .999f || (x) > 1.001f)
		return (NOT_ONE(la2) || NOT_ONE(lb2) || NOT_ONE(lc2));
#undef NOT_ONE
	}
	HOST_AND_DEVICE inline Point3f operator()(const Point3f& pt) const;

	HOST_AND_DEVICE inline Vector3f operator()(const Vector3f& v) const;

	HOST_AND_DEVICE inline Normal3f operator()(const Normal3f&) const;
	HOST_AND_DEVICE Transform operator*(const Transform& t2) const;
private:
	// Transform Private Data
	Matrix4x4 m, mInv;
	bool invFlag;
};


HOST_AND_DEVICE inline Vector3f Transform::operator()(const Vector3f& v) const {
	float x = v.x, y = v.y, z = v.z;
	return Vector3f(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
		m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
		m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
}

HOST_AND_DEVICE inline Normal3f Transform::operator()(const Normal3f& n) const {
	float x = n.x, y = n.y, z = n.z;
	return Normal3f(mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z,
		mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z,
		mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z);
}

HOST_AND_DEVICE inline Point3f Transform::operator()(const Point3f& pt) const {
	float x = pt.x, y = pt.y, z = pt.z;
	float xp = (m.m[0][0] * x + m.m[0][1] * y) + (m.m[0][2] * z + m.m[0][3]);
	float yp = (m.m[1][0] * x + m.m[1][1] * y) + (m.m[1][2] * z + m.m[1][3]);
	float zp = (m.m[2][0] * x + m.m[2][1] * y) + (m.m[2][2] * z + m.m[2][3]);
	float wp = (m.m[3][0] * x + m.m[3][1] * y) + (m.m[3][2] * z + m.m[3][3]);
	if (wp == 1.)
		return Point3f(xp, yp, zp);
	else
		return Point3f(xp, yp, zp) / wp;
}

HOST_AND_DEVICE inline Matrix4x4::Matrix4x4(float mat[4][4]) { memcpy(m, mat, 16 * sizeof(float)); }
HOST_AND_DEVICE inline Matrix4x4::Matrix4x4(
	float t00, float t01, float t02, float t03,
	float t10, float t11, float t12, float t13,
	float t20, float t21, float t22, float t23,
	float t30, float t31, float t32, float t33) {
	m[0][0] = t00;
	m[0][1] = t01;
	m[0][2] = t02;
	m[0][3] = t03;
	m[1][0] = t10;
	m[1][1] = t11;
	m[1][2] = t12;
	m[1][3] = t13;
	m[2][0] = t20;
	m[2][1] = t21;
	m[2][2] = t22;
	m[2][3] = t23;
	m[3][0] = t30;
	m[3][1] = t31;
	m[3][2] = t32;
	m[3][3] = t33;
}

HOST_AND_DEVICE inline Matrix4x4 Transpose(const Matrix4x4& m) {
	return Matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
		m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
		m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
		m.m[3][3]);
}

HOST_AND_DEVICE inline Matrix4x4 Inverse(const Matrix4x4& m, bool &flag) {
	int indxc[4], indxr[4];
	int ipiv[4] = { 0, 0, 0, 0 };
	float minv[4][4];
	memcpy(minv, m.m, 4 * 4 * sizeof(float));
	for (int i = 0; i < 4; i++) {
		int irow = 0, icol = 0;
		float big = 0.f;
		// Choose pivot
		for (int j = 0; j < 4; j++) {
			if (ipiv[j] != 1) {
				for (int k = 0; k < 4; k++) {
					if (ipiv[k] == 0) {
						if (abs(minv[j][k]) >= big) {
							big = float(abs(minv[j][k]));
							irow = j;
							icol = k;
						}
					}
					else if (ipiv[k] > 1) {
						// Singular matrix;
						flag = false;
						return Matrix4x4();
					}
				}
			}
		}
		++ipiv[icol];
		// Swap rows _irow_ and _icol_ for pivot
		if (irow != icol) {
			for (int k = 0; k < 4; ++k) swap(minv[irow][k], minv[icol][k]);
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if (minv[icol][icol] == 0.f) {
			// Singular matrix;
			flag = false;
			return Matrix4x4();
		}

		// Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
		float pivinv = 1. / minv[icol][icol];
		minv[icol][icol] = 1.;
		for (int j = 0; j < 4; j++) minv[icol][j] *= pivinv;

		// Subtract this row from others to zero out their columns
		for (int j = 0; j < 4; j++) {
			if (j != icol) {
				float save = minv[j][icol];
				minv[j][icol] = 0;
				for (int k = 0; k < 4; k++) minv[j][k] -= minv[icol][k] * save;
			}
		}
	}
	// Swap columns to reflect permutation
	for (int j = 3; j >= 0; j--) {
		if (indxr[j] != indxc[j]) {
			for (int k = 0; k < 4; k++)
				swap(minv[k][indxr[j]], minv[k][indxc[j]]);
		}
	}
	flag = true;
	return Matrix4x4(minv);
}

HOST_AND_DEVICE inline Transform Transform::operator*(const Transform& t2) const {
	return Transform(Matrix4x4::Mul(m, t2.m), Matrix4x4::Mul(t2.mInv, mInv));
}

HOST_AND_DEVICE inline Transform Translate(const Vector3f& delta) {
	Matrix4x4 m(1, 0, 0, delta.x, 0, 1, 0, delta.y, 0, 0, 1, delta.z, 0, 0, 0,
		1);
	Matrix4x4 minv(1, 0, 0, -delta.x, 0, 1, 0, -delta.y, 0, 0, 1, -delta.z, 0,
		0, 0, 1);
	return Transform(m, minv);
}

HOST_AND_DEVICE inline Transform Scale(float x, float y, float z) {
	Matrix4x4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
	Matrix4x4 minv(1 / x, 0, 0, 0, 0, 1 / y, 0, 0, 0, 0, 1 / z, 0, 0, 0, 0, 1);
	return Transform(m, minv);
}

HOST_AND_DEVICE inline Transform RotateX(float theta) {
	float sinTheta = sin(Radians(theta));
	float cosTheta = cos(Radians(theta));
	Matrix4x4 m(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0,
		0, 0, 0, 1);
	return Transform(m, Transpose(m));
}

HOST_AND_DEVICE inline Transform RotateY(float theta) {
	float sinTheta = sin(Radians(theta));
	float cosTheta = cos(Radians(theta));
	Matrix4x4 m(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0,
		0, 0, 0, 1);
	return Transform(m, Transpose(m));
}

HOST_AND_DEVICE inline Transform RotateZ(float theta) {
	float sinTheta = sin(Radians(theta));
	float cosTheta = cos(Radians(theta));
	Matrix4x4 m(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 1);
	return Transform(m, Transpose(m));
}

HOST_AND_DEVICE inline Transform Rotate(float theta, const Vector3f& axis) {
	Vector3f a = Normalize(axis);
	float sinTheta = sin(Radians(theta));
	float cosTheta = cos(Radians(theta));
	Matrix4x4 m;
	// Compute rotation of first basis vector
	m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
	m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
	m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
	m.m[0][3] = 0;

	// Compute rotations of second and third basis vectors
	m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
	m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
	m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
	m.m[1][3] = 0;

	m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
	m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
	m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
	m.m[2][3] = 0;
	return Transform(m, Transpose(m));
}

}



#endif








