/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, 2022 Università degli Studi di Milano. All rights reserved.
 *                         Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <numeric>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>

// space.h
// 
// This file contains the geometric type definitions and general utility functions
// 
// The mesh representation follows the one adopted by libigl (https://libigl.github.io)
// Buffers are represented as Eigen matrices, with elements by row and columns encoding
// attribute dimensions (e.g. the vertices position are represented as a #V x 3 matrix
// of Scalars, where #V is the number of vertices)
// A face buffer is a #F x 3 matrix of integers indexing vertex buffers, where #F is the
// number of mesh faces.
// 
// In principle, it can be possible to have different index matrices for different
// vertex buffers (e.g. different indices for vertex positions and uvs).
// 
// The Scalar typedef defines the primitive scalar type and defaults
// to double unless the project is built with the USE_SINGLE_PRECISION
// macro defined
//

// geometric types

#ifdef USE_SINGLE_PRECISION
typedef float Scalar;
#else
typedef double Scalar;
#endif

typedef Eigen::Vector2i Vector2i;
typedef Eigen::Vector3i Vector3i;
typedef Eigen::Vector4i Vector4i;
typedef Eigen::VectorXi VectorXi;

typedef Eigen::Matrix<float, 2, 1> Vector2f;
typedef Eigen::Matrix<float, 3, 1> Vector3f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;

typedef Eigen::Matrix<float, 2, 2> Matrix2f;
typedef Eigen::Matrix<float, 3, 3> Matrix3f;
typedef Eigen::Matrix<float, 4, 4> Matrix4f;

typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;

typedef Eigen::Matrix<uint8_t, 2, 1> Vector2u8;
typedef Eigen::Matrix<uint8_t, 3, 1> Vector3u8;
typedef Eigen::Matrix<uint8_t, 4, 1> Vector4u8;

typedef Eigen::Matrix<Scalar, -1, 1> VectorX;
typedef Eigen::Matrix<uint8_t, -1, 1> VectorXu8;
typedef Eigen::Matrix<int8_t, -1, 1> VectorXi8;

typedef Eigen::Matrix<double, 2, 2> Matrix2d;
typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;

typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

typedef Eigen::Matrix<Scalar, 1, 2> RowVector2;
typedef Eigen::Matrix<Scalar, 1, 3> RowVector3;
typedef Eigen::Matrix<Scalar, 1, 4> RowVector4;

typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

typedef Eigen::Matrix<Scalar, -1, -1> MatrixX;
typedef Eigen::Matrix<float, -1, -1> MatrixXf;
typedef Eigen::Matrix<double, -1, -1> MatrixXd;
typedef Eigen::Matrix<int, -1, -1> MatrixXi;
typedef Eigen::Matrix<uint32_t, -1, -1> MatrixXu32;
typedef Eigen::Matrix<uint8_t, -1, -1> MatrixXu8;
typedef Eigen::Matrix<int8_t, -1, -1> MatrixXi8;

typedef Eigen::SparseMatrix<Scalar> SparseMatrix;

// invalid index used to denote ``deleted`` faces
constexpr int INVALID_INDEX = -1;

constexpr Scalar Infinity = std::numeric_limits<Scalar>::infinity();
constexpr float Infinityf = std::numeric_limits<float>::infinity();
constexpr double Infinityd = std::numeric_limits<double>::infinity();

// an edge is represented as a sorted pair of vertex indices
struct Edge : public std::pair<int, int> {
	Edge(int v1, int v2)
		: std::pair<int, int>(std::min(v1, v2), std::max(v1, v2))
	{
	}

	Edge()
		: std::pair<int, int>(INVALID_INDEX, INVALID_INDEX)
	{
	}
};

namespace std
{
	template<> struct hash<Edge>
	{
		std::size_t operator()(const Edge& e) const noexcept
		{
			std::size_t h1 = std::hash<int>{}(e.first);
			std::size_t h2 = std::hash<int>{}(e.second);
			return h1 ^ (h2 << 1);
		}
	};
}

inline bool operator<(const Vector3& v1, const Vector3& v2)
{
	return v1[0] != v2[0] ? (v1[0] < v2[0]) :
		(v1[1] != v2[1] ? (v1[1] < v2[1]) :
			(v1[2] < v2[2]));
}

// the ratio of twice the inradius to the circumradius
//  p = l0 + l1 + l2 (perimeter)
//  r = area / semiperimeter
// 2R = l0 * l1 * l2 / (2 * area)
// then, 2r / R = (16 * area^2) / (p * l0 * l1 * l2)
inline Scalar aspect_ratio(const Vector3& a, const Vector3& b, const Vector3& c)
{
	Scalar doublearea_squared = (b - a).cross(c - a).squaredNorm();
	Scalar l0 = (b - a).norm();
	Scalar l1 = (c - b).norm();
	Scalar l2 = (a - c).norm();
	Scalar p = l0 + l1 + l2;
	return (4 * doublearea_squared) / (p * l0 * l1 * l2);
}

// convert degrees to radians
constexpr Scalar radians(const Scalar deg)
{
	return deg * (M_PI / 180.0);
}

// convert radians to degrees
constexpr Scalar degrees(const Scalar rad)
{
	return rad * (180.0 / M_PI);
}

inline Scalar vector_angle(const Vector3& u, const Vector3& v)
{
	return std::atan2(u.cross(v).norm(), u.dot(v));
}

inline Scalar cot(Scalar angle_in_radians)
{
	return std::cos(angle_in_radians) / std::sin(angle_in_radians);
}

inline Vector3 barycenter(const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	return (p0 + p1 + p2) / 3.0;
}

// given a point and the 3 vertices of a triangle, computes
// the index of the nearest edge. Assumes the point lies on
// the triangle plane
inline int nearest_edge(const Vector3& p, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	Vector3 p10 = (p1 - p0).normalized();
	Vector3 p21 = (p2 - p1).normalized();
	Vector3 p02 = (p0 - p2).normalized();
	Scalar d0 = (p0 + (p - p0).dot(p10) * p10 - p).norm();
	Scalar d1 = (p1 + (p - p1).dot(p21) * p21 - p).norm();
	Scalar d2 = (p2 + (p - p2).dot(p02) * p02 - p).norm();

	if (d0 <= std::min(d1, d2))
		return 0;
	else if (d1 <= std::min(d0, d2))
		return 1;
	else
		return 2;
}

// given a point and the 3 vertices of a triangle, computes
// the index of the nearest vertex.
inline int nearest_vertex(const Vector3& p, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	Scalar d0 = (p0 - p).norm();
	Scalar d1 = (p1 - p).norm();
	Scalar d2 = (p2 - p).norm();

	if (d0 <= std::min(d1, d2))
		return 0;
	else if (d1 <= std::min(d0, d2))
		return 1;
	else
		return 2;
}

template <typename T>
inline T clamp(const T v, const T vmin, const T vmax)
{
	return std::max(vmin, std::min(v, vmax));
}

inline Vector3 clamp(const Vector3& v, const Vector3& vmin, const Vector3& vmax)
{
	Vector3 q;
	q[0] = clamp(v[0], vmin[0], vmax[0]);
	q[1] = clamp(v[1], vmin[1], vmax[1]);
	q[2] = clamp(v[2], vmin[2], vmax[2]);
	return q;
}

inline Vector2 clamp(const Vector2& v, const Vector2& vmin, const Vector2& vmax)
{
	Vector2 q;
	q[0] = clamp(v[0], vmin[0], vmax[0]);
	q[1] = clamp(v[1], vmin[1], vmax[1]);
	return q;
}

// given a point and the 3 vertices of a triangle, computes the
// barycentric coordinates of the point.
inline Vector3 compute_bary_coords(const Vector3& p, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	Scalar area = (p1 - p0).cross(p2 - p0).norm();

	return Vector3(
		(p1 - p).cross(p2 - p).norm(),
		(p2 - p).cross(p0 - p).norm(),
		(p0 - p).cross(p1 - p).norm()) / area;
}

// given a point and the 3 vertices of a triangle, computes
// the projection of the point onto the triangle
inline Vector3 nearest_triangle_point(const Vector3& p, const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	Matrix3 F;
	F.col(0) = p1 - p0;
	F.col(1) = p2 - p0;
	F.col(2) = F.col(0).cross(F.col(1));
	Vector3 bary = F.inverse() * (p - p0);

	Vector3 n = (p1 - p0).cross(p2 - p0).normalized();
	Vector3 proj_p = p - (p - p0).dot(n) * n;
	Vector3 q = p0 + bary[0] * F.col(0) + bary[1] * F.col(1);

	if (bary[0] >= 0 && bary[1] >= 0 && (bary[0] + bary[1]) <= 1) {
		// planar projection inside tri
		return q;
	}
	else {
		// project the planar projection onto the nearest edge
		Vector3 p10n = (p1 - p0).normalized();
		Vector3 proj10 = p0 + p10n * (clamp(p10n.dot(q - p0), Scalar(0), (p1 - p0).norm()));
		Scalar d10 = (q - proj10).norm();

		Vector3 p21n = (p2 - p1).normalized();
		Vector3 proj21 = p1 + p21n * (clamp(p21n.dot(q - p1), Scalar(0), (p2 - p1).norm()));
		Scalar d21 = (q - proj21).norm();

		Vector3 p02n = (p0 - p2).normalized();
		Vector3 proj02 = p2 + p02n * (clamp(p02n.dot(q - p2), Scalar(0), (p0 - p2).norm()));
		Scalar d02 = (q - proj02).norm();

		if (d10 <= std::min(d21, d02))
			return proj10;
		else if (d21 <= std::min(d10, d02))
			return proj21;
		else
			return proj02;
	}
}

// given the 3 vertices of a triangle, computes a local orthonormal frame
inline Matrix3 local_frame(const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	Vector3 p20 = (p2 - p0).normalized();
	Matrix3 F;
	F.row(0) = (p1 - p0).normalized();
	F.row(2) = F.row(0).cross(p20).normalized();
	F.row(1) = F.row(2).cross(F.row(0)).normalized();
	return F;
}

// Axis-angle rotation (Rodrigues formula)
inline Vector3 rotate(const Vector3& p, Vector3 axis, Scalar angle)
{
	axis.normalize();
	Scalar cu = std::cos(angle);
	Scalar su = std::sin(angle);
	return p * cu + (axis.cross(p)) * su + axis * (axis.dot(p)) * (1 - cu);
}

// computes the approximate volume of a prismoid represented as 3 bottom vertices
// and 3 offset vectors connecting bottom and top vertices
inline Scalar prismoid_volume_approximate(const Vector3& v0, const Vector3& v1, const Vector3& v2,
	const Vector3& d0, const Vector3& d1, const Vector3& d2)
{
	return (v1 - v0).cross(v2 - v0).dot((d0 + d1 + d2)) / 6.0;
}

// returns the parameter t such that o + t * d is the projection of p onto the ray
inline Scalar project_onto_ray(const Vector3& p, const Vector3& o, const Vector3& d)
{
	return (p - o).dot(d) / d.squaredNorm();
}

// volume of the tetra divided by the average edge length
// division by 6 omitted
inline Scalar quad_planarity(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d)
{
	Scalar v = std::abs((d - a).dot((b - a).cross(c - a)));
	Scalar l = (b - a).norm() + (c - a).norm() + (d - a).norm() + (b - c).norm() + (b - d).norm() + (c - d).norm();
	return v / l;
}

// given the 4 vertices of a quad, computes the corresponding interior angles
inline Vector4 quad_angles(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d)
{
	return Vector4(
		vector_angle(d - a, b - a),
		vector_angle(a - b, c - b),
		vector_angle(b - c, d - c),
		vector_angle(c - d, a - d)
	);
}

inline Scalar triangle_area(const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	return 0.5 * (p1 - p0).cross(p2 - p0).norm();
}

inline Scalar triangle_area(const Vector2& p0, const Vector2& p1, const Vector2& p2)
{
	return (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p2.x() - p0.x()) * (p1.y() - p0.y());
}

// Finds the minimum t such that area_vector * area_vector(t, d) >= 0
inline Scalar flip_preventing_offset(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& d)
{
	Vector3 p10 = p1 - p0;
	Vector3 p20 = p2 - p0;
	Vector3 area_vector = p10.cross(p20);
	Vector3 dd = d.cross(p10 - p20);
	if (area_vector.dot(dd) >= 0) {
		return 1;
	}
	else {
		Scalar t = -area_vector.squaredNorm() / (area_vector.dot(dd));
		return clamp(t, Scalar(0), Scalar(1));
	}
}

// Finds the minimum t such that n * area_vector(t, d) >= 0
inline Scalar flip_preventing_offset(const Vector3& p0, const Vector3& p1, const Vector3& p2, const Vector3& d, const Vector3& n)
{
	Vector3 p10 = p1 - p0;
	Vector3 p20 = p2 - p0;
	Scalar geq = -n.dot(p10.cross(p20));
	Scalar denom = n.dot(d.cross(p1 - p2));
	Scalar val = geq / denom;
	if (denom >= 0) {
		if (val < 1)
			return 1;
		else
			return 0;
	}
	else {
		// t <= geq / denom
		return clamp(val, Scalar(0), Scalar(1));
	}
}

