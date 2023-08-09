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

#include "smooth.h"
#include "tangent.h"
#include "quadric.h"

void flip_guarded_laplacian_smooth(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB, const MatrixX& FN)
{
	MatrixX VS = V;
	for (int n = 0; n < V.cols(); ++n) {
		VectorX X = VS.col(n);
		laplacian_smooth(X, F, VF, VB);
		VS.col(n) = X;
	}

	for (int i = 0; i < V.rows(); ++i) {
		Vector3 d = VS.row(i) - V.row(i);
		Scalar tmin = 1;
		for (const VFEntry vfe : VF[i]) {
			int v0 = F(vfe.first, vfe.second);
			int v1 = F(vfe.first, (vfe.second + 1) % 3);
			int v2 = F(vfe.first, (vfe.second + 2) % 3);
			Scalar t = flip_preventing_offset(V.row(v0), V.row(v1), V.row(v2), d, FN.row(vfe.first));
			tmin = std::min(t, tmin);
		}
		// if some face constrained the vertex move, be conservative and dampen the offset
		if (tmin < 1)
			tmin *= 0.8;
		V.row(i) = V.row(i) + tmin * d.transpose();
	}
}

void flip_guarded_anisotropic_smoothing(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB, const MatrixX& FN,
	int iterations, Scalar border_error_scale, Scalar weight)
{
	Assert(iterations >= 0);
	if (iterations == 0)
		return;

	Timer t;

	weight = clamp(weight, Scalar(0), Scalar(1));

	std::vector<Scalar> QW;
	std::vector<Quadric> Q;

	compute_quadrics_per_vertex(V, F, VF, VB, border_error_scale, Q, QW);

	std::vector<Scalar> QQW = QW;
	std::vector<Quadric> QQ = Q;

	for (int k = 0; k < iterations; ++k) {
		for (long i = 0; i < V.rows(); ++i) {
			if (!VB[i]) {
				Quadric q(Matrix3::Zero(), Vector3::Zero(), 0);
				Scalar qw = 0;

				for (const VFEntry& vfe : VF[i]) {
					int vnext = F(vfe.first, (vfe.second + 1) % 3);
					if (!VB[vnext]) {
						q.add(Q[vnext]);
						qw += QW[vnext];
					}
				}

				QQ[i] = Q[i] + q;
				QQW[i] = QW[i] + qw;
			}
		}

		Q = QQ;
		QW = QQW;
	}

	for (int i = 0; i < V.rows(); ++i) {
		Quadric qv = compute_point_quadric(V.row(i));
		Quadric qs = Q[i] * (Scalar(1) / QW[i]);
		Quadric q = (1 - weight) * qv + weight * qs;

		Vector3 d = q.minimizer() - V.row(i).transpose();
		Scalar tmin = 1;
		for (const VFEntry vfe : VF[i]) {
			int v0 = F(vfe.first, vfe.second);
			int v1 = F(vfe.first, (vfe.second + 1) % 3);
			int v2 = F(vfe.first, (vfe.second + 2) % 3);
			Scalar t = flip_preventing_offset(V.row(v0), V.row(v1), V.row(v2), d, FN.row(vfe.first));
			tmin = std::min(t, tmin);
		}
		// if some face constrained the vertex move, be conservative and dampen the offset
		if (tmin < 1)
			tmin *= 0.8;
		V.row(i) = V.row(i) + tmin * d.transpose();
	}
}

void laplacian_smooth(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB)
{
	for (int n = 0; n < V.cols(); ++n) {
		VectorX X = V.col(n);
		laplacian_smooth(X, F, VF, VB);
		V.col(n) = X;
	}
}

void laplacian_smooth(VectorX& X, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB)
{
	VectorX XS = VectorX::Constant(X.rows(), 0);
	VectorXi deg = VectorXi::Constant(X.rows(), 0);

	for (int vi = 0; vi < X.rows(); ++vi) {
		for (int vj : vertex_star(vi, F, VF)) {
			if (!VB(vi) || VB(vj)) {
				XS.row(vi) += X.row(vj);
				deg(vi)++;
			}
		}
	}

	for (int vi = 0; vi < X.rows(); ++vi) {
		X.row(vi) = (X.row(vi) + XS.row(vi)) / Scalar(deg(vi) + 1);
	}
}

void tangent_plane_laplacian_smooth(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB)
{
	std::vector<Plane> planes;
	planes.reserve(V.rows());
	{
		MatrixX VN = compute_vertex_normals(V, F);
		for (int i = 0; i < V.rows(); ++i)
			planes.push_back(Plane(V.row(i), VN.row(i)));
	}

	laplacian_smooth(V, F, VF, VB);

	for (int vi = 0; vi < V.rows(); ++vi) {
		V.row(vi) = project(V.row(vi), planes[vi]);
	}
}

