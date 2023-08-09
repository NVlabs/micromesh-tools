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

#include "space.h"
#include "adjacency.h"

#include <vector>

#define THIN_QUADRICS

struct Quadric {

#ifdef THIN_QUADRICS
	typedef Eigen::Matrix<Scalar, 6, 1> SymmetricMatrix3;

	SymmetricMatrix3 A;
#else
	Matrix3 A;
#endif

	Vector3 b;
	Scalar c;

	Quadric(const Matrix3& AA, const Vector3& bb, const Scalar& cc)
#ifdef THIN_QUADRICS
		: A(), b(bb), c(cc)
	{
		A(0) = AA(0, 0);
		A(1) = AA(0, 1);
		A(2) = AA(0, 2);
		A(3) = AA(1, 1);
		A(4) = AA(1, 2);
		A(5) = AA(2, 2);
	}
#else
		: A(AA), b(bb), c(cc)
	{
	}
#endif

	Quadric(const Quadric& other)
		: A(other.A), b(other.b), c(other.c)
	{
	}

	Quadric& add(const Quadric& other)
	{
		A += other.A;
		b += other.b;
		c += other.c;
		return *this;
	}

	Quadric& operator+=(const Quadric& other)
	{
		return this->add(other);
	}

	Quadric operator+(const Quadric& other) const
	{
		return Quadric(_get_full_matrix() + other._get_full_matrix(), b + other.b, c + other.c);
	}

	Quadric operator*(Scalar s) const
	{
		return Quadric(s * _get_full_matrix(), s * b, s * c);
	}

	Quadric operator/(Scalar s) const
	{
		return Quadric((1 / s) * _get_full_matrix(), (1 / s) * b, (1 / s) * c);
	}

	Scalar error(const Vector3& q) const
	{
		return q.transpose() * (_get_full_matrix() * q) + (q.dot(b)) + c;
	}

	Vector3 minimizer() const
	{
		return -(2 * _get_full_matrix()).inverse() * b;
	}

	Vector3 gradient(const Vector3& p)
	{
		return 2 * _get_full_matrix() * p + b;
	}

	inline Matrix3 _get_full_matrix() const
	{
#ifdef THIN_QUADRICS
		Matrix3 M;
		M << A(0), A(1), A(2),
		     A(1), A(3), A(4),
		     A(2), A(4), A(5);
		return M;
#else
		return A;
#endif
	}
};

inline Quadric operator*(const Scalar& s, const Quadric& quadric)
{
	return quadric * s;
}

void compute_quadrics_per_vertex(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB, Scalar border_error_scale, std::vector<Quadric>& Q, std::vector<Scalar>& QW);
MatrixX per_vertex_quadric_gradients(const MatrixX& V, const std::vector<Quadric>& Q, const std::vector<Scalar>& QW);
std::pair<Quadric, Scalar> compute_face_quadric(const MatrixX& V, const MatrixXi& F, int fi);
Quadric compute_point_quadric(const Vector3& p);
Quadric compute_plane_quadric(const Vector3& p, const Vector3& n);

// Compute a distance-from-line quadric
// p is a point on the line, d a vector parallel to the line
// Returns a quadric Q such that Q(x) = |p + ((x - p).dot(d) * d) - x|^2
// This is the squared distance of the x from its orthogonal projection onto the line parametrized by p + t * d
Quadric compute_line_quadric(const Vector3& p, Vector3 d);

// maps quadrics of mesh1 onto the vertices of mesh2 using a bidirectional nearest mapping
void map_quadrics_onto_mesh(const MatrixX& V1, const MatrixXi& F1, const VectorXu8& VB1, Scalar input_border_error_scale,
	const MatrixX& V2, const MatrixXi& F2, const VectorXu8 VB2, std::vector<Quadric>& Q, std::vector<Scalar>& QW);

//void quadrics_smoothing(MatrixX& V, const MatrixXi& F, Scalar weight, int iterations, Scalar border_error_scale);

