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
#include "utils.h"

struct Plane {
	Vector3 p;
	Vector3 n;

	Plane(const Vector3& pp, const Vector3& nn)
		: p(pp), n(nn.normalized())
	{
	}
};

// TODO rename to area_vector
inline Vector3 compute_face_normal(int fi, const MatrixX& V, const MatrixXi& F)
{
	Assert(fi < F.rows());
	Assert(F.cols() == 3);
	Vector3 v10 = V.row(F(fi, 1)) - V.row(F(fi, 0));
	Vector3 v20 = V.row(F(fi, 2)) - V.row(F(fi, 0));
	return v10.cross(v20);
}

inline Vector3 compute_area_vector(const Vector3& p0, const Vector3& p1, const Vector3& p2)
{
	return (p1 - p0).cross(p2 - p0);
}

inline Vector3 compute_vertex_normal(int vi, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF)
{
	Assert(vi < V.rows());
	Vector3 n = Vector3::Zero();
	for (const VFEntry& vfe : VF[vi]) {
		int fi = vfe.first;
		n += compute_face_normal(fi, V, F);
	}
	return n;
}

inline Vector3 project(const Vector3& q, const Plane& plane)
{
	return q - (plane.n.dot((q - plane.p)) * plane.n);
}

inline MatrixX compute_vertex_normals(const MatrixX& V, const MatrixXi& F)
{
	MatrixX VN = MatrixX::Constant(V.rows(), 3, 0);

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			Vector3 v10 = (V.row(F(i, 1)) - V.row(F(i, 0)));
			Vector3 v20 = (V.row(F(i, 2)) - V.row(F(i, 0)));
			Vector3 n = v10.cross(v20);
			for (int j = 0; j < 3; ++j) {
				VN.row(F(i, j)) += n;
			}
		}
	}

	for (int i = 0; i < VN.rows(); ++i)
		VN.row(i).normalize();

	return VN;
}

inline MatrixX compute_face_normals(const MatrixX& V, const MatrixXi& F)
{
	MatrixX FN = MatrixX::Constant(F.rows(), 3, 0);

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX)
			FN.row(i) = compute_area_vector(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2))).normalized();
	}

	return FN;
}

inline MatrixX compute_face_area_vectors(const MatrixX& V, const MatrixXi& F)
{
	MatrixX FN = MatrixX::Constant(F.rows(), 3, 0);

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX)
			FN.row(i) = compute_area_vector(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)));
	}

	return FN;
}



inline MatrixX compute_tangent_vectors(const MatrixX& V, const MatrixX& UV, const MatrixXi& F)
{
	MatrixX VT = MatrixX::Constant(V.rows(), 3, 0);

	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX) {
			Vector2 u10 = UV.row(F(fi, 1)) - UV.row(F(fi, 0));
			Vector2 u20 = UV.row(F(fi, 2)) - UV.row(F(fi, 0));
			Vector3 x10 = V.row(F(fi, 1)) - V.row(F(fi, 0));
			Vector3 x20 = V.row(F(fi, 2)) - V.row(F(fi, 0));

			MatrixX A = MatrixX::Constant(6, 6, 0);
			A(0, 0) = u10.x();
			A(0, 1) = u10.y();
			A(1, 2) = u10.x();
			A(1, 3) = u10.y();
			A(2, 4) = u10.x();
			A(2, 5) = u10.y();
			A(3, 0) = u20.x();
			A(3, 1) = u20.y();
			A(4, 2) = u20.x();
			A(4, 3) = u20.y();
			A(5, 4) = u20.x();
			A(5, 5) = u20.y();

			VectorX b(6);
			b(0) = x10.x();
			b(1) = x10.y();
			b(2) = x10.z();
			b(3) = x20.y();
			b(4) = x20.x();
			b(5) = x20.z();

			VectorX e = A.colPivHouseholderQr().solve(b);

			for (int i = 0; i < 3; ++i)
				VT.row(F(fi, i)) += Vector3(e(0), e(2), e(4));

		}
	}

	VT.rowwise().normalize();
	VT.conservativeResize(VT.rows(), 4);
	VT.col(3) = VectorX::Constant(VT.rows(), 1);

	return VT;
}

