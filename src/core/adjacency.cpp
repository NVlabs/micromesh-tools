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

#include "adjacency.h"
#include "tangent.h"
#include "utils.h"

//// To add: faster VFAdjacency implementation
//struct vf_adjacency {
//
//	VectorXi _vf; // adjacent face index
//	VectorXi8 _vfvi; // face-vertex index (0, 1, 2) 
//
//	MatrixXi _fvf;
//	MatrixXi8 _fvfvi;
//
//	vf_adjacency(const MatrixX& V, const MatrixXi& F)
//	{
//		compute(V, F);
//	}
//
//	void compute(const MatrixX& V, const MatrixXi& F)
//	{
//		_vf = VectorXi::Constant(V.rows(), INVALID_INDEX);
//		_vfvi = VectorXi8::Constant(V.rows(), 0);
//
//		_fvf = MatrixXi::Constant(F.rows(), 3, INVALID_INDEX);
//		_fvfvi = MatrixXi8::Constant(F.rows(), 3, INVALID_INDEX);
//
//		for (int fi = 0; fi < F.rows(); ++fi) {
//			for (int8_t i = 0; i < 3; ++i) {
//				int vi = F(fi, i);
//				if (_vf(vi) != INVALID_INDEX) {
//					_fvf(fi, i) = _vf(vi);
//					_fvfvi(fi, i) = _vfvi(vi);
//				}
//				_vf(vi) = fi;
//				_vfvi(vi) = i;
//			}
//		}
//	}
//			
//	std::vector<VFEntry> adj(int vi) const
//	{
//		Assert(vi >= 0);
//		Assert(vi < _vf.rows());
//		
//		std::vector<VFEntry> vflist;
//		
//		const int* fp = &_vf(vi); // face index
//		const int8_t * fvip = &_vfvi(vi); // face vertex index
//		
//		while (*fp != INVALID_INDEX) {
//			vflist.push_back(VFEntry(*fp, *fvip));
//			int fi = *fp;
//			int fvi = *fvip;
//			fp = &_fvf(fi, fvi);
//			fvip = &_fvfvi(fi, fvi);
//		}
//		
//		return vflist;
//	}
//		
//};


// simple iteration over the face indices
VFAdjacency compute_adjacency_vertex_face(const MatrixX& V, const MatrixXi& F)
{
	VFAdjacency VF;
	VF.resize(V.rows(), {});

	for (int i = 0; i < F.rows(); ++i) {
		for (int j = 0; j < F.cols(); ++j) {
			int vi = F(i, j);
			if (vi >= 0) {
				VF[vi].insert(VFEntry(i, j));
			}
		}
	}

	return VF;
}

EFAdjacency compute_adjacency_edge_face(const MatrixXi& F)
{
	std::map<Edge, std::vector<EFEntry>> EF;

	for (int fi = 0; fi < F.rows(); ++fi) {
		for (int8_t j = 0; j < 3; ++j) {
			Edge e(F(fi, j), F(fi, (j + 1) % 3));
			EF[e].push_back(EFEntry(fi, j));
		}
	}

	return EF;
}

std::set<int> shared_faces(const VFAdjacency& VF, int v1, int v2)
{
	std::set<int> vf1;
	std::set<int> shared;

	for (const VFEntry& vfe : VF[v1]) {
		vf1.insert(vfe.first);
	}

	for (const VFEntry& vfe : VF[v2]) {
		if (vf1.find(vfe.first) != vf1.end())
			shared.insert(vfe.first);
	}

	Assert(shared.size() != 0);

	return shared;
}

std::set<int> vertex_star(int vi, const MatrixXi& F, const VFAdjacency& VF)
{
	std::set<int> vstar;

	for (const VFEntry vfe : VF[vi]) {
		vstar.insert(F(vfe.first, (vfe.second + 1) % F.cols()));
		vstar.insert(F(vfe.first, (vfe.second + 2) % F.cols()));
	}

	return vstar;
}

Flap compute_flap(Edge e, const MatrixXi& F, const VFAdjacency& VF)
{
	// Iterate over the adjacency of e.first, visiting edges from each face
	Flap flap;
	for (const VFEntry vfe : VF[e.first]) {
		for (int i = 0; i < 3; ++i) {
			if (Edge(F(vfe.first, i), F(vfe.first, (i + 1) % 3)) == e) {
				flap.f.push_back(vfe.first);
				flap.e.push_back(i);
			}
		}
	}
	Assert(flap.size() > 0);
	return flap;
}

void per_vertex_border_flag(const MatrixX& V, const MatrixXi& F, VectorXu8& VB)
{
	MatrixXu8 FEB = per_face_edge_border_flag(V, F);
	VB = per_vertex_border_flag(V, F, FEB);
}

void per_vertex_border_flag(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, VectorXu8& VB)
{
	VB = VectorXu8::Constant(V.rows(), 0);

	std::unordered_map<int, int> counter;

	for (int i = 0; i < V.rows(); ++i) {
		counter.clear();
		for (const VFEntry& vfe : VF[i]) {
			counter[F(vfe.first, (vfe.second + 1) % 3)]++;
			counter[F(vfe.first, (vfe.second + 2) % 3)]++;
		}
		for (const auto& p : counter) {
			if (p.second == 1) {
				VB(i) = 1;
				break;
			}
		}
	}
}

VectorXu8 per_vertex_border_flag(const MatrixX& V, const MatrixXi& F, const MatrixXu8& FEB)
{
	VectorXu8 VB = VectorXu8::Constant(V.rows(), 0);

	for (int i = 0; i < F.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			if (FEB(i, j)) {
				VB(F(i, j)) = 1;
				VB(F(i, (j + 1) % 3)) = 1;
			}
		}
	}

	return VB;
}

MatrixXu8 per_face_edge_border_flag(const MatrixX& V, const MatrixXi& F)
{
	MatrixXu8 FEB = MatrixXu8::Constant(F.rows(), 3, 0);
	
	int deg = F.cols();

	std::unordered_map<Edge, int> em;
	em.reserve(1.6 * F.rows());

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) == INVALID_INDEX)
			continue;
		for (int j = 0; j < deg; ++j) {
			Edge e(F(i, j), F(i, (j + 1) % deg));
			em[e]++;
		}
	}

	// all edges seen only once are border edges,
	// set the corresponding vertex flags
	for (int i = 0; i < F.rows(); ++i)
		for (int j = 0; j < 3; ++j)
			if (em[Edge(F(i, j), F(i, (j + 1) % 3))] == 1)
				FEB(i, j) = 1;

	return FEB;
}

Vector4i vertex_ring(const Flap& flap, const MatrixXi& F)
{
	Assert(flap.size() == 2);
	int f0 = flap.f[0];
	int f1 = flap.f[1];
	int e0 = flap.e[0];
	int e1 = flap.e[1];

	int a = F(f0, (e0 + 2) % 3);
	int b = F(f0, e0);
	int c = F(f1, (e1 + 2) % 3);
	int d = F(f1, e1);

	return Vector4i(a, b, c, d);
}

Scalar flap_planarity(const Flap& flap, const MatrixX& V, const MatrixXi& F)
{
	Vector4i ring = vertex_ring(flap, F);
	return quad_planarity(V.row(ring(0)), V.row(ring(1)), V.row(ring(2)), V.row(ring(3)));
}

Scalar flap_normals_angle(const Flap& flap, const MatrixX& V, const MatrixXi& F)
{
	Assert(flap.size() == 2);

	return vector_angle(compute_face_normal(flap.f[0], V, F), compute_face_normal(flap.f[1], V, F));
}

Vector4 flap_angles(const Flap& flap, const MatrixX& V, const MatrixXi& F)
{
	Vector4i ring = vertex_ring(flap, F);
	return quad_angles(V.row(ring(0)), V.row(ring(1)), V.row(ring(2)), V.row(ring(3)));
}

std::vector<EFEntry> compute_face_circle(EFEntry start, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF)
{
	std::vector<EFEntry> face_circle;

	int vi = F(start.fi(), start.ei()); // the 'pivoting vertex' of the circle
	Flap flap = compute_flap(Edge(F(start.fi(), start.ei()), F(start.fi(), start.ei(1))), F, VF);
	Assert(flap.size() == 2);

	EFEntry ef = start;

	face_circle.push_back(ef);
	std::set<int> visited = { ef.fi() };

	while (true) {
		Assert(F(ef.fi(), ef.ei()) == vi);
		Edge e(F(ef.fi(), ef.ei()), F(ef.fi(), ef.ei(2)));

		flap = compute_flap(e, F, VF);
		if (F(flap.f[0], flap.e[0]) == vi)
			ef = EFEntry(flap.f[0], flap.e[0]);
		else
			ef = EFEntry(flap.f[1], flap.e[1]);

		if (visited.find(ef.fi()) != visited.end())
			break;
		else {
			face_circle.push_back(ef);
			visited.insert(ef.fi());
		}
	}

	return face_circle;
}

