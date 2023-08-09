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

#include "quadric.h"
#include "adjacency.h"
#include "utils.h"
#include "mesh_utils.h"
#include "tangent.h"
#include "bvh.h"
#include "micro.h"

#include <vector>

#include <iostream>
#include <execution>

void compute_quadrics_per_vertex(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB, Scalar border_error_scale, std::vector<Quadric>& Q, std::vector<Scalar>& QW)
{
	int deg = 3;

	Timer t;

	Q.clear();
	QW.clear();

	Q.resize(V.rows(), Quadric(Matrix3::Zero(), Vector3::Zero(), 0));
	QW.resize(V.rows(), 0);

	std::vector<std::pair<Quadric, Scalar>> face_quadrics;
	face_quadrics.resize(F.rows(), std::make_pair(Quadric(Matrix3::Zero(), Vector3::Zero(), 0), 0));

	auto face_quadric_task = [&](int fi) -> void {
		if (F(fi, 0) != INVALID_INDEX)
			face_quadrics[fi] = compute_face_quadric(V, F, fi);
	};

	std::vector<int> face_indices(F.rows());
	std::iota(face_indices.begin(), face_indices.end(), 0);

	auto vertex_quadric_task = [&](int vi) -> void {
		for (const VFEntry& vfe : VF[vi]) {
			const std::pair<Quadric, Scalar>& qw = face_quadrics[vfe.first];
			Q[vi] += qw.second * qw.first;
			QW[vi] += qw.second;
		}
	};

	std::vector<int> vert_indices(V.rows());
	std::iota(vert_indices.begin(), vert_indices.end(), 0);

	std::for_each(std::execution::par_unseq, face_indices.begin(), face_indices.end(), face_quadric_task);
	std::for_each(std::execution::par_unseq, vert_indices.begin(), vert_indices.end(), vertex_quadric_task);

	// accumulate border quadrics
	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			for (int j = 0; j < deg; ++j) {
				int v0 = F(i, j);
				int v1 = F(i, (j + 1) % deg);

				if (VB(v0) && VB(v1)) {
					Flap flap = compute_flap(Edge(v0, v1), F, VF);
					if (flap.size() == 1) {
						Scalar w = face_area(V, F.row(i));
						Vector3 e10 = V.row(v1) - V.row(v0);

						// retain the original area weight also
						//const Scalar wt = w * std::pow(border_error_scale, Scalar(2));
						const Scalar wt = w * 6;
						Quadric quadric_t = wt * compute_line_quadric(V.row(v0), e10);

						Q[v0] += quadric_t;
						QW[v0] += w;

						Q[v1] += quadric_t;
						QW[v1] += w;
					}
				}
			}
		}
	}

	VectorX vertex_areas = compute_voronoi_vertex_areas(V, F);

	Scalar w = 1e-8 * (*std::min_element(QW.begin(), QW.end()));

	for (int vi = 0; vi < V.rows(); ++vi) {
		//Scalar w = 1e-3 * (VB(vi) ? border_error_scale : 1) * vertex_areas(vi);
		Q[vi] += (w * compute_point_quadric(V.row(vi)));
		QW[vi] += w;
	}

	VectorX vertex_scaling_coefficients = VectorX::Constant(V.rows(), 1);

	std::set<int> vnext;
	Scalar next_val = border_error_scale;

	for (int i = 0; i < V.rows(); ++i)
		if (VB(i))
			vnext.insert(i);

	for (int k = 0; k < 20; ++k) {
		for (int vi : vnext) {
			if (vertex_scaling_coefficients(vi) < next_val) {
				vertex_scaling_coefficients(vi) = next_val;
			}
		}
		next_val = next_val * 0.75;
		if (next_val <= 1)
			break;
		std::set<int> vcurr = vnext;
		vnext.clear();
		for (int vi : vcurr) {
			for (const VFEntry& vfe : VF[vi]) {
				for (int j = 0; j < 3; ++j) {
					if (vertex_scaling_coefficients(F(vfe.fi(), j)) < next_val)
						vnext.insert(F(vfe.fi(), j));
				}
			}
		}
	}

	for (int vi = 0; vi < V.rows(); ++vi) {
		if (vertex_scaling_coefficients(vi) > 1) {
			Q[vi] = Q[vi] * vertex_scaling_coefficients(vi);
			//QW[vi] = QW[vi] * vertex_scaling_coefficients(vi);
		}
	}
}


//void compute_quadrics_per_vertex_(const MatrixX& V, const MatrixXi& F, const VectorXu8& VB, std::vector<Quadric>& Q, std::vector<Scalar>& QW)
//{
//	int deg = 3;
//
//	Q.reserve(V.rows());
//	QW.reserve(V.rows());
//
//	for (int vi = 0; vi < V.rows(); ++vi) {
//		Scalar w = 1e-6;
//		Q.push_back(w * compute_point_quadric(V.row(vi)));
//		QW.push_back(w);
//	}
//
//	// init quadrics
//	for (int i = 0; i < F.rows(); ++i) {
//		if (F(i, 0) != INVALID_INDEX) {
//			std::pair<Quadric, Scalar> qw = compute_face_quadric(V, F, i);
//			//Scalar w = qw.second * 0.333333333;
//			Scalar w = qw.second;
//			for (int j = 0; j < F.cols(); ++j) {
//				Q[F(i, j)] = Q[F(i, j)] + (w * qw.first);
//				QW[F(i, j)] = QW[F(i, j)] + w;
//			}
//		}
//	}
//
//	// add border quadrics
//	std::map<Edge, int> em;
//	for (int i = 0; i < F.rows(); ++i) {
//		if (F(i, 0) == INVALID_INDEX)
//			continue;
//		for (int j = 0; j < deg; ++j) {
//			Edge e(F(i, j), F(i, (j + 1) % deg));
//			em[e]++;
//		}
//	}
//
//	for (int i = 0; i < F.rows(); ++i) {
//		if (F(i, 0) != INVALID_INDEX) {
//			for (int j = 0; j < deg; ++j) {
//				int v0 = F(i, j);
//				int v1 = F(i, (j + 1) % deg);
//				auto itedge = em.find(Edge(v0, v1));
//				if (itedge != em.end() && itedge->second == 1) {
//
//					Vector3 e10 = V.row(v1) - V.row(v0);
//					Vector3 e20 = V.row(F(i, (j + 2) % deg)) - V.row(v0);
//					Vector3 n = e10.cross(e20);
//					Vector3 t = n.cross(e10);
//
//					const Scalar wp = 0;
//					const Scalar wt = 2 * e10.norm();
//					const Scalar wn = 0;
//
//					//Scalar wt = v10j.squaredNorm();
//					Quadric quadric_t = wt * compute_plane_quadric(V.row(v0), t);
//					Quadric quadric_n = wn * compute_plane_quadric(V.row(v0), n);
//
//					//Quadric quadric_p0 =  wp * compute_point_quadric(V.row(v0));
//					//Quadric quadric_p1 =  wp * compute_point_quadric(V.row(v1));
//
//					Quadric quadric_e0 =  wp * compute_plane_quadric(V.row(v0), e10);
//					Quadric quadric_e1 =  wp * compute_plane_quadric(V.row(v1), e10);
//
//					//Q[v0] = quadric_t + quadric_n + quadric_e0;
//					//QW[v0] = 1;// wp + wn + wt;
//
//					//Q[v1] = quadric_t + quadric_n + quadric_e1;
//					//QW[v1] = 1;// wp + wn + wt;
//
//					Q[v0] = Q[v0] + quadric_t + quadric_n + quadric_e0;
//					QW[v0] +=  wp + wn + wt;
//
//					Q[v1] = Q[v1] + quadric_t + quadric_n + quadric_e1;
//					QW[v1] += wp + wn + wt;
//
//				}
//
//			}
//		}
//	}
//}

MatrixX per_vertex_quadric_gradients(const MatrixX& V, const std::vector<Quadric>& Q, const std::vector<Scalar>& QW)
{
	MatrixX QG(V.rows(), 3);
	QG.setZero();

	for (int i = 0; i < V.rows(); ++i) {
		Quadric q = Q[i] * (1.0 / QW[i]);
		QG.row(i) = q.gradient(V.row(i));
	}

	return QG;
}

// derived by encoding the plane as point p and normal n
// the quadric measures the distance of a point q to the plane
// as the length of the projection of (q - V0) onto n
std::pair<Quadric, Scalar> compute_face_quadric(const MatrixX& V, const MatrixXi& F, int fi)
{
	Vector3 v10 = V.row(F(fi, 1)) - V.row(F(fi, 0));
	Vector3 v20 = V.row(F(fi, 2)) - V.row(F(fi, 0));
	Vector3 n = v10.cross(v20);
	Scalar area = 0.5 * n.norm();
	n.normalize();
	Vector3 p = V.row(F(fi, 0));
	Scalar pn = p.dot(n);
	return std::make_pair(Quadric(n * n.transpose(), -2 * pn * n, pn * pn), area);
}

Quadric compute_point_quadric(const Vector3& p)
{
	return Quadric(Matrix3::Identity(), -2 * p, p.squaredNorm());
}

Quadric compute_plane_quadric(const Vector3& p, const Vector3& n)
{
	Vector3 nn = n.normalized();
	Scalar pn = p.dot(nn);
	return Quadric(nn * nn.transpose(), -2 * pn * nn, pn * pn);
}

// Returns a quadric Q such that Q(x) = |p + ((x - p).dot(d) * d) - x|^2
// This is the squared distance of the point from its orthogonal projection onto the line
// parametrized by p + t * d
// The derivation is mechanical, nothing special
Quadric compute_line_quadric(const Vector3& p, Vector3 d)
{
	d.normalize();
	Matrix3 I = Matrix3::Identity();
	Matrix3 ddt = d * d.transpose();
	Matrix3 ddt2 = ddt * ddt;
	return Quadric(
		I - 2 * ddt + ddt2,
		p.transpose() * (-2 * I + 4 * ddt - 2 * ddt2),
		p.transpose() * (I - 2 * ddt + ddt2) * p
	);
}

void map_quadrics_onto_mesh(const MatrixX& V1, const MatrixXi& F1, const VectorXu8& VB1, Scalar input_border_error_scale,
	const MatrixX& V2, const MatrixXi& F2, const VectorXu8 VB2, std::vector<Quadric>& Q, std::vector<Scalar>& QW)
{
	Timer t;
	MatrixX VN1 = compute_vertex_normals(V1, F1);
	MatrixX VN2 = compute_vertex_normals(V2, F2);

	MatrixX FN1 = compute_face_normals(V1, F1);
	MatrixX FN2 = compute_face_normals(V2, F2);

	VFAdjacency VF1 = compute_adjacency_vertex_face(V1, F1);

	std::vector<Quadric> Q1;
	std::vector<Scalar> QW1;
	compute_quadrics_per_vertex(V1, F1, VF1, VB1, input_border_error_scale, Q1, QW1);

	std::vector<uint8_t> cleared(V2.rows(), 0);

	auto accumulate_quadric = [&](const Quadric& q, Scalar w, int vi_2) -> void {
		if (!cleared[vi_2]) {
		QW[vi_2] = 1e-6;
		Q[vi_2] = QW[vi_2] * compute_point_quadric(V2.row(vi_2));
		cleared[vi_2] = 1;
		}
		Q[vi_2].add(q);
		QW[vi_2] += w;
	};

	Scalar avg_area1 = average_area(V1, F1);

	// map using V1 -> nearest on Mesh2
	{
		BVHTree bvh2;
		bvh2.build_tree(&V2, &F2, &VN2);


		for (int vi = 0; vi < (int)V1.rows(); ++vi) {
			NearestInfo ni;
			bool found_nearest = bvh2.nearest_point(V1.row(vi), &ni, [&](int fi_2) { return FN2.row(fi_2).dot(VN1.row(vi)) >= 0; });
			if (found_nearest) {
				Vector3 bary = compute_bary_coords(ni.p, V2.row(F2(ni.fi, 0)), V2.row(F2(ni.fi, 1)), V2.row(F2(ni.fi, 2)));
				// distribute the quadric onto the vertices of ni.fi
				for (int k = 0; k < 3; ++k) {
					accumulate_quadric(Q1[vi] * bary(k), QW1[vi] * bary(k), F2(ni.fi, k));
				}
			}
		}
	}

	// map using V2 -> nearest on Mesh1
	{
		BVHTree bvh1;
		bvh1.build_tree(&V1, &F1, &VN1);

		MatrixX FV;
		MatrixXi FF;

		for (int fi = 0; fi < (int)F2.rows(); ++fi) {
			if (F2(fi, 0) != INVALID_INDEX) {
				Scalar f_area = face_area(V2, F2.row(fi));

				int subdivision_level = std::round(std::log2(f_area / avg_area1) / 2.0);
				if (subdivision_level < 0)
					subdivision_level = 0;
				//subdivision_level = std::min(subdivision_level, 3);
				uint8_t subdivision_bits = subdivision_level << 3;
				subdivide_tri(V2.row(F2(fi, 0)), V2.row(F2(fi, 1)), V2.row(F2(fi, 2)), FV, FF, subdivision_bits);

				Scalar ff_area = f_area / (Scalar)FF.rows();

				for (int ffi = 0; ffi < (int)FF.rows(); ++ffi) {
					// compute centroid
					Vector3 c = barycenter(FV.row(FF(ffi, 0)), FV.row(FF(ffi, 1)), FV.row(FF(ffi, 2)));
					
					// lookup nearest
					NearestInfo ni;
					bool found_nearest = bvh1.nearest_point(c, &ni, [&](int fi_1) { return FN1.row(fi_1).dot(FN2.row(fi)) >= 0; });
					if (found_nearest) {
						// compute intersecting quadric
						// TODO precompute the face quadrics ?
						std::pair<Quadric, Scalar> qp = compute_face_quadric(V1, F1, ni.fi);
						// compute barys of centroid
						Vector3 bary = compute_bary_coords(c, V2.row(F2(fi, 0)), V2.row(F2(fi, 1)), V2.row(F2(fi, 2)));
						// distribute quadric
						for (int k = 0; k < 3; ++k) {
							accumulate_quadric(qp.first * ff_area * bary(k), ff_area * bary(k), F2(fi, k));
						}
					}
				}
			}
		}
	}

	std::cout << "map_quadrics_onto_mesh() took " << t.time_elapsed() << " seconds" << std::endl;
}

//void quadrics_smoothing(MatrixX& V, const MatrixXi& F, Scalar weight, int iterations, Scalar border_error_scale)
//{
//	Assert(iterations >= 0);
//	if (iterations == 0)
//		return;
//
//	weight = clamp(weight, Scalar(0), Scalar(1));
//
//	std::vector<Scalar> QW;
//	std::vector<Quadric> Q;
//
//	VectorXu8 VB;
//	VFAdjacency VF = compute_adjacency_vertex_face(V, F);
//	per_vertex_border_flag(V, F, VB);
//
//	compute_quadrics_per_vertex(V, F, VB, border_error_scale, Q, QW);
//
//	std::vector<Scalar> QQW = QW;
//	std::vector<Quadric> QQ = Q;
//
//	for (int k = 0; k < iterations; ++k) {
//		for (long i = 0; i < V.rows(); ++i) {
//			if (!VB[i]) {
//				Quadric q(Matrix3::Zero(), Vector3::Zero(), 0);
//				Scalar qw = 0;
//
//				for (const VFEntry& vfe : VF[i]) {
//					int vnext = F(vfe.first, (vfe.second + 1) % 3);
//					if (!VB[vnext]) {
//						q.add(Q[vnext]);
//						qw += QW[vnext];
//					}
//				}
//
//				QQ[i] = Q[i] + q;
//				QQW[i] = QW[i] + qw;
//			}
//		}
//
//		Q = QQ;
//		QW = QQW;
//	}
//
//	for (long i = 0; i < V.rows(); ++i) {
//		Quadric qv = compute_point_quadric(V.row(i));
//		Quadric qs = Q[i] * (Scalar(1) / QW[i]);
//		Quadric q = (1 - weight) * qv + weight * qs;
//		V.row(i) = q.minimizer();
//	}
//}

