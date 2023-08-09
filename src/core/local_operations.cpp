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

#include "local_operations.h"
#include "quadric.h"
#include "tangent.h"
#include "visibility.h"
#include "utils.h"

// -- Local operations --

namespace detail {
	SplitInfo split_edge(const Flap& flap, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, int u, int f12, int f22)
	{
		Assert(flap.size() > 0);

		SplitInfo si = {};

		si.ok = flap.size() <= 2;
		if (si.ok) {

			si.f1 = flap.f[0];

			if (flap.size() == 2)
				si.f2 = flap.f[1];
			else
				si.f2 = INVALID_INDEX;

			si.e = Edge(F(si.f1, flap.e[0]), F(si.f1, (flap.e[0] + 1) % 3));

			int new_faces = si.f2 != INVALID_INDEX ? 2 : 1;

			// disconnect from VF
			{
				std::vector<int> fdel = { si.f1, si.f2 };
				for (int fi : fdel) {
					if (fi != INVALID_INDEX) {
						for (int8_t i = 0; i < 3; ++i) {
							VF[F(fi, i)].erase(VFEntry(fi, i));
						}
					}
				}
			}

			si.u = u;
			si.split_position = Scalar(0.5) * (V.row(si.e.first) + V.row(si.e.second));

			V.row(si.u) = si.split_position;
			VB(si.u) = si.f2 == INVALID_INDEX ? 1 : 0;

			si.f12 = f12;

			int v10 = F(si.f1, flap.e[0]);
			int v11 = F(si.f1, (flap.e[0] + 1) % 3);
			int v12 = F(si.f1, (flap.e[0] + 2) % 3);

			F(si.f1, 0) = si.u;
			F(si.f1, 1) = v12;
			F(si.f1, 2) = v10;
			F(si.f12, 0) = si.u;
			F(si.f12, 1) = v11;
			F(si.f12, 2) = v12;

			if (si.f2 != INVALID_INDEX) {
				si.f22 = f22;

				int v20 = F(si.f2, flap.e[1]);
				int v21 = F(si.f2, (flap.e[1] + 1) % 3);
				int v22 = F(si.f2, (flap.e[1] + 2) % 3);

				F(si.f2, 0) = si.u;
				F(si.f2, 1) = v22;
				F(si.f2, 2) = v20;
				F(si.f22, 0) = si.u;
				F(si.f22, 1) = v21;
				F(si.f22, 2) = v22;
			}

			// update VF
			{
				std::vector<int> fadd = { si.f1, si.f12 };

				if (si.f2 != INVALID_INDEX) {
					fadd.push_back(si.f2);
					fadd.push_back(si.f22);
				}

				for (int fi : fadd) {
					for (int8_t i = 0; i < 3; ++i) {
						VF[F(fi, i)].insert(VFEntry(fi, i));
					}
				}
			}
		}

		return si;
	}
}

CollapseInfo collapse_edge(const Flap& flap, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, const Vector3& vertex_position)
{
	Assert(flap.size() > 0);
	int deg = int(F.cols());

	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));

	CollapseInfo collapse;
	collapse.ok = true;
	collapse.vertex = e.first;
	collapse.collapsed_vertex = e.second;

	V.row(e.first) = vertex_position;

	int v1 = e.first;
	int v2 = e.second;


	// detect edges that expire due to the operation
	// i.e. all edges incident on v2

	for (const VFEntry& vfe : VF[v2]) {
		int fi = vfe.first;
		for (int i = 0; i < deg; ++i) {
			if (F(fi, i) == v2) {
				collapse.expired.insert(Edge(F(fi, i), F(fi, (i + 1) % deg)));
				collapse.expired.insert(Edge(F(fi, (i + deg - 1) % deg), F(fi, i)));
			}
		}
	}

	// remove from VF and invalidate all the collapsed faces

	for (int fi : flap.f) {
		for (int i = 0; i < deg; ++i) {
			int vi = F(fi, i);
			Assert(VF[vi].erase(VFEntry(fi, i)) > 0);
			F(fi, i) = INVALID_INDEX;
		}
	}

	// re-index faces around v2

	for (const VFEntry& vfe : VF[v2]) {
		int fi = vfe.first;
		int fvi = vfe.second;
		Assert(F(fi, 0) != INVALID_INDEX && F(fi, 1) != INVALID_INDEX && F(fi, 2) != INVALID_INDEX);
		Assert(F(fi, fvi) == v2);
		F(fi, fvi) = v1;
		VF[v1].insert(vfe);
	}

	VF[v2].clear();

	VB[v1] = VB[v1] || VB[v2];
	VB[v2] = false;

	return collapse;
}

FlipInfo flip_edge(const Flap& flap, const MatrixX& V, MatrixXi& F, VFAdjacency& VF)
{
	Assert(flap.size() == 2);
	
	FlipInfo flip;

	int f0 = flap.f[0];
	int f1 = flap.f[1];
	int e0 = flap.e[0];
	int e1 = flap.e[1];
	int opp1 = (e0 + 2) % 3;
	int opp2 = (e1 + 2) % 3;

	flip.ok = true;

	Assert(VF[F(f0, e0)].erase(VFEntry(f0, e0)) > 0);
	Assert(VF[F(f1, e1)].erase(VFEntry(f1, e1)) > 0);

	VF[F(f0, opp1)].insert(VFEntry(f1, e1));
	VF[F(f1, opp2)].insert(VFEntry(f0, e0));

	F(f0, e0) = F(f1, opp2);
	F(f1, e1) = F(f0, opp1);

	Assert(F(f0, 0) != F(f0, 1) && F(f0, 0) != F(f0, 2) && F(f0, 1) != F(f0, 2));
	Assert(F(f1, 0) != F(f1, 1) && F(f1, 0) != F(f1, 2) && F(f1, 1) != F(f1, 2));

	return flip;
}

//
//   v10                              v10         
//   | \                              | \         
//   |  \                             |  \        
//   |   \                            |   \       
//   |    \                           | F' \      
//  e| F   > v12    [split(e)] =>   u +-----> v12 
//   |    /                           | F''/      
//   |   /                            |   /       
//   |  /                             |  /        
//   | /                              | /         
//   v11                              v11         
//
SplitInfo split_edge(const Flap& flap, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB)
{
	Assert(flap.size() > 0);

	SplitInfo si = {};

	si.ok = flap.size() <= 2;
	if (si.ok) {

		si.f1 = flap.f[0];

		if (flap.size() == 2)
			si.f2 = flap.f[1];
		else
			si.f2 = INVALID_INDEX;

		si.e = Edge(F(si.f1, flap.e[0]), F(si.f1, (flap.e[0] + 1) % 3));

		int new_faces = si.f2 != INVALID_INDEX ? 2 : 1;

		V.conservativeResize(V.rows() + 1, V.cols());
		F.conservativeResize(F.rows() + new_faces, F.cols());
		VF.resize(VF.size() + 1);
		VB.conservativeResize(VB.rows() + 1);

		// disconnect from VF
		{
			std::vector<int> fdel = { si.f1, si.f2 };
			for (int fi : fdel) {
				if (fi != INVALID_INDEX) {
					for (int8_t i = 0; i < 3; ++i) {
						VF[F(fi, i)].erase(VFEntry(fi, i));
					}
				}
			}
		}

		si.u = V.rows() - 1;
		si.split_position = Scalar(0.5) * (V.row(si.e.first) + V.row(si.e.second));


		V.row(si.u) = si.split_position;
		VB(si.u) = si.f2 == INVALID_INDEX ? 1 : 0;

		si.f12 = F.rows() - new_faces;
		
		int v10 = F(si.f1, flap.e[0]);
		int v11 = F(si.f1, (flap.e[0] + 1) % 3);
		int v12 = F(si.f1, (flap.e[0] + 2) % 3);

		F(si.f1, 0) = si.u;
		F(si.f1, 1) = v12;
		F(si.f1, 2) = v10;
		F(si.f12, 0) = si.u;
		F(si.f12, 1) = v11;
		F(si.f12, 2) = v12;

		if (si.f2 != INVALID_INDEX) {
			si.f22 = F.rows() - new_faces + 1;
			
			int v20 = F(si.f2, flap.e[1]);
			int v21 = F(si.f2, (flap.e[1] + 1) % 3);
			int v22 = F(si.f2, (flap.e[1] + 2) % 3);

			F(si.f2, 0) = si.u;
			F(si.f2, 1) = v22;
			F(si.f2, 2) = v20;
			F(si.f22, 0) = si.u;
			F(si.f22, 1) = v21;
			F(si.f22, 2) = v22;
		}

		// update VF
		{
			std::vector<int> fadd = { si.f1, si.f12 };

			if (si.f2 != INVALID_INDEX) {
				fadd.push_back(si.f2);
				fadd.push_back(si.f22);
			}

			for (int fi : fadd) {
				for (int8_t i = 0; i < 3; ++i) {
					VF[F(fi, i)].insert(VFEntry(fi, i));
				}
			}
		}
	}

	return si;
}

// splits a vertex, substituting two edges with two faces
VertexSplitInfo split_vertex(int vi, Edge e1, Edge e2, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB)
{
	VertexSplitInfo vsi = {};

	if (VB(vi) || e1 == e2) {
		return vsi;
	}
	
	V.conservativeResize(V.rows() + 1, V.cols());
	F.conservativeResize(F.rows() + 2, F.cols());
	VF.resize(VF.size() + 1);
	VB.conservativeResize(VB.rows() + 1);

	vsi.old_vertex = vi;
	vsi.new_vertex = V.rows() - 1;

	int vi_e1 = e1.first == vi ? e1.second : e1.first;
	int vi_e2 = e2.first == vi ? e2.second : e2.first;

	VB(vsi.new_vertex) = false;
	V.row(vsi.new_vertex) = V.row(vi);

	vsi.f1 = F.rows() - 2;
	vsi.f2 = F.rows() - 1;

	EFEntry start(-1, 0);

	Flap flap = compute_flap(e1, F, VF);
	if (F(flap.f[0], flap.e[0]) == vi)
		start = EFEntry(flap.f[0], flap.e[0]);
	else
		start = EFEntry(flap.f[1], flap.e[1]);

	std::vector<EFEntry> face_circle = compute_face_circle(start, V, F, VF);

	vsi.f1_old = start.fi();

	// we started from e1 when inserting faces in face_circle
	// iterate over EFEntry objects until we find e2
	// faces from e1 to e2 are on one side of the divide
	// the others are on the other side of the divide

	bool insert_new_vertex = false;
	for (EFEntry& ef : face_circle) {
		if (!insert_new_vertex && Edge(F(ef.fi(), ef.ei()), F(ef.fi(), ef.ei(1))) == e2) {
			insert_new_vertex = true;
			vsi.f2_old = ef.fi();
		}

		if (insert_new_vertex)
			F(ef.fi(), ef.ei()) = vsi.new_vertex;
	}

	F(vsi.f1, 0) = vi_e1;
	F(vsi.f1, 1) = vi;
	F(vsi.f1, 2) = vsi.new_vertex;

	F(vsi.f2, 0) = vi_e2;
	F(vsi.f2, 1) = vsi.new_vertex;
	F(vsi.f2, 2) = vi;

	Assert(insert_new_vertex);

	vsi.ok = true;

	std::cout << "TODO update mesh adjacency data" << std::endl;

	return vsi;
}

// -- Full mesh pass --

FlipInfo flip_edge(const MatrixX& V, MatrixXi& F, Edge e, VFAdjacency& VF, const Quadric& edge_quadric, Scalar err_threshold)
{
	FlipInfo flip;

	int deg = F.cols();

	int v1 = e.first;
	int v2 = e.second;

	std::set<int> shared = shared_faces(VF, v1, v2);

	auto it = shared.begin();
	int f1 = *it++;
	int f2 = *it;
	int e1 = -1;
	int e2 = -1;
	for (int i = 0; i < deg; ++i) {
		if (Edge(F(f1, i), F(f1, (i + 1) % deg)) == e)
			e1 = i;
		if (Edge(F(f2, i), F(f2, (i + 1) % deg)) == e)
			e2 = i;
	}

	flip.ok = (shared.size() == 2);
	//if (flip.ok)
	//	flip.ok = flip_preserves_topology(F, VF, f1, f2, e1, e2);
	//if (flip.ok)
	//	flip.ok = flip_preserves_geometry_quadric(V, F, edge_quadric, err_threshold, f1, f2, e1, e2);

	if (flip.ok) {
		Assert(VF[F(f1, e1)].erase(VFEntry(f1, e1)) > 0);
		Assert(VF[F(f2, e2)].erase(VFEntry(f2, e2)) > 0);

		int opp1 = (e1 + 2) % deg;
		int opp2 = (e2 + 2) % deg;
		VF[F(f1, opp1)].insert(VFEntry(f2, e2));
		VF[F(f2, opp2)].insert(VFEntry(f1, e1));

		F(f1, e1) = F(f2, opp2);
		F(f2, e2) = F(f1, opp1);

		Assert(F(f1, 0) != F(f1, 1) && F(f1, 0) != F(f1, 2) && F(f1, 1) != F(f1, 2));
		Assert(F(f2, 0) != F(f2, 1) && F(f2, 0) != F(f2, 2) && F(f2, 1) != F(f2, 2));
	}

	return flip;
}

FlipInfo flip_edge(const MatrixX& V, MatrixXi& F, Edge e, VFAdjacency& VF)
{
	FlipInfo flip;

	int deg = F.cols();

	int v1 = e.first;
	int v2 = e.second;

	std::set<int> shared = shared_faces(VF, v1, v2);

	auto it = shared.begin();
	int f1 = *it++;
	int f2 = *it;
	int e1 = -1;
	int e2 = -1;
	for (int i = 0; i < deg; ++i) {
		if (Edge(F(f1, i), F(f1, (i + 1) % deg)) == e)
			e1 = i;
		if (Edge(F(f2, i), F(f2, (i + 1) % deg)) == e)
			e2 = i;
	}

	flip.ok = (shared.size() == 2);
	//if (flip.ok)
	//	flip.ok = flip_preserves_topology(F, VF, f1, f2, e1, e2);
	//if (flip.ok)
	//	flip.ok = flip_preserves_geometry(V, F, f1, f2, e1, e2);

	if (flip.ok) {
		Assert(VF[F(f1, e1)].erase(VFEntry(f1, e1)) > 0);
		Assert(VF[F(f2, e2)].erase(VFEntry(f2, e2)) > 0);

		int opp1 = (e1 + 2) % deg;
		int opp2 = (e2 + 2) % deg;
		VF[F(f1, opp1)].insert(VFEntry(f2, e2));
		VF[F(f2, opp2)].insert(VFEntry(f1, e1));

		F(f1, e1) = F(f2, opp2);
		F(f2, e2) = F(f1, opp1);

		Assert(F(f1, 0) != F(f1, 1) && F(f1, 0) != F(f1, 2) && F(f1, 1) != F(f1, 2));
		Assert(F(f2, 0) != F(f2, 1) && F(f2, 0) != F(f2, 2) && F(f2, 1) != F(f2, 2));
	}

	return flip;
}

// -- Qualitative tests (hard constraints) --

bool collapse_preserves_topology(const Flap& flap, const MatrixXi& F, const VFAdjacency& VF, VectorXu8& VB)
{
	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	int v1 = e.first;
	int v2 = e.second;
	int v_opp = F(flap.f[0], (flap.e[0] + 2) % deg);

	if (flap.size() > 2) {
		return false;
	}
	
	if (VB[v1] && VB[v2]) {
		// if the endpoints are both border vertices, make sure this is a border edge
		// also make sure that at least one face survives the collapse
		if (flap.size() != 1 || VF[v_opp].size() <= 1)
			return false;
	}
		
	std::set<int> nv1;
	std::set<int> nv2;

	for (const VFEntry& vfe : VF[v1]) {
		int fi = vfe.first;
		for (int i = 0; i < deg; ++i)
			nv1.insert(F(fi, i));
	}

	for (const VFEntry& vfe : VF[v2]) {
		int fi = vfe.first;
		for (int i = 0; i < deg; ++i)
			nv2.insert(F(fi, i));
	}

	for (auto it = nv1.begin(); it != nv1.end(); ) {
		if (nv2.find(*it) == nv2.end())
			it = nv1.erase(it);
		else
			it++;
	}

	// to ensure the contraction preserves topology
	// the edge must be manifold, and the only common
	// neighbors between v1 and v2 must be the 2 vertices
	// opposite to the edge itself.
	// nv1.size() == 4 => nv = {v1, v2, opposite_1, opposite_2}

	if (VB[v1] && VB[v2])
		return nv1.size() == 3 && (VF[v1].size() > 1 || VF[v2].size() > 1); // in the border case, also make sure that the surface doesn't 'disappear'
	else
		return nv1.size() == 4;
}

bool collapse_preserves_geometry(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	double min_cos_theta)
{
	Assert(flap.size() > 0);

	auto test_normal = [&](int fi, int vi) -> bool {
		if (fi == flap.f[0] || (flap.size() > 1 && fi == flap.f[1]))
			return true;
		Vector3 v10 = V.row(F(fi, 1)) - V.row(F(fi, 0));
		Vector3 v20 = V.row(F(fi, 2)) - V.row(F(fi, 0));
		Vector3 n1 = v10.cross(v20).normalized();
		Vector3 test0 = F(fi, 0) == vi ? vpos : V.row(F(fi, 0));
		Vector3 test1 = F(fi, 1) == vi ? vpos : V.row(F(fi, 1));
		Vector3 test2 = F(fi, 2) == vi ? vpos : V.row(F(fi, 2));
		Vector3 n2 = (test1 - test0).cross(test2 - test0).normalized();
		return n1.dot(n2) > min_cos_theta;
	};

	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	for (const VFEntry& vfe : VF[v1])
		if (!test_normal(vfe.first, v1))
			return false;

	for (const VFEntry& vfe : VF[v2])
		if (!test_normal(vfe.first, v2))
			return false;
	
	return true;
}

bool collapse_preserves_geometry(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	Scalar orientation_threshold, Scalar border_orientation_threshold, const VectorXu8& VB, const MatrixX& FN)
{
	Assert(flap.size() > 0);

	auto test_normal = [&](int fi, int vi) -> bool {
		if (fi == flap.f[0] || (flap.size() > 1 && fi == flap.f[1]))
			return true;
		Vector3 test0 = F(fi, 0) == vi ? vpos : V.row(F(fi, 0));
		Vector3 test1 = F(fi, 1) == vi ? vpos : V.row(F(fi, 1));
		Vector3 test2 = F(fi, 2) == vi ? vpos : V.row(F(fi, 2));
		Vector3 nt = (test1 - test0).cross(test2 - test0).normalized();
		bool border = VB(F(fi, 0)) || VB(F(fi, 1)) || VB(F(fi, 2));
		Scalar nt_dot_fn = nt.dot(FN.row(fi));
		if (nt_dot_fn > (border ? border_orientation_threshold : orientation_threshold)) {
			// above threshold, OK
			return true;
		}
		else {
			// below threshold, check if the orientation improves
			Vector3 p0 = V.row(F(fi, 0));
			Vector3 p1 = V.row(F(fi, 1));
			Vector3 p2 = V.row(F(fi, 2));
			Vector3 n = (p1 - p0).cross(p2 - p0).normalized();
			return nt_dot_fn >= n.dot(FN.row(fi));
		}
	};

	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	for (const VFEntry& vfe : VF[v1])
		if (!test_normal(vfe.first, v1))
			return false;

	for (const VFEntry& vfe : VF[v2])
		if (!test_normal(vfe.first, v2))
			return false;

	return true;
}

bool collapse_preserves_orientation_topology(const Flap& flap, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB)
{
	Assert(flap.size() > 0);
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % 3));
	int v1 = e.first;
	int v2 = e.second;

	if (VB(v1) == VB(v2))
		return true;

	int v_internal = VB(v1) ? v2 : v1;
	for (const VFEntry& vfe : VF[v_internal]) // test if it is a border face
		if (!(VB(F(vfe.first, 0)) || VB(F(vfe.first, 1)) || VB(F(vfe.first, 2))))
			return false;

	return true;
}

bool collapse_preserves_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	double min_ratio, const VectorX& FR, Scalar tolerance)
{
	auto test_aspect_ratio = [&](int fi, int vi) -> bool {
		// return true if the face disappears after the collapse
		if (fi == flap.f[0] || (flap.size() > 1 && fi == flap.f[1]))
			return true;

		// otherwise, perform the test
		Vector3 v0 = V.row(F(fi, 0));
		Vector3 v1 = V.row(F(fi, 1));
		Vector3 v2 = V.row(F(fi, 2));
		
		Scalar ar = aspect_ratio(v0, v1, v2);
		
		Vector3 test0 = F(fi, 0) == vi ? vpos : V.row(F(fi, 0));
		Vector3 test1 = F(fi, 1) == vi ? vpos : V.row(F(fi, 1));
		Vector3 test2 = F(fi, 2) == vi ? vpos : V.row(F(fi, 2));

		Scalar arnew = aspect_ratio(test0, test1, test2);

		return (arnew >= min_ratio) || (arnew < min_ratio && arnew >= (FR(fi) - tolerance));
	};

	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));

	int v1 = e.first;
	int v2 = e.second;

	for (const VFEntry& vfe : VF[v1])
		if (!test_aspect_ratio(vfe.first, v1))
			return false;

	for (const VFEntry& vfe : VF[v2])
		if (!test_aspect_ratio(vfe.first, v2))
			return false;

	return true;
}

bool collapse_preserves_vertex_ring_normals2(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	double max_vertex_ring_normals_angle)
{
	auto compute_test_area_vector = [&](int fi, int vi) ->Vector3 {
		if (fi == flap.f[0] || (flap.size() > 1 && fi == flap.f[1]))
			return Vector3(0, 0, 0);

		Vector3 p0 = F(fi, 0) == vi ? vpos : V.row(F(fi, 0));
		Vector3 p1 = F(fi, 1) == vi ? vpos : V.row(F(fi, 1));
		Vector3 p2 = F(fi, 2) == vi ? vpos : V.row(F(fi, 2));

		return compute_area_vector(p0, p1, p2);
	};

	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	Vector3 n(0, 0, 0);
	for (const VFEntry& vfe : VF[v1])
		n += compute_test_area_vector(vfe.first, v1);

	for (const VFEntry& vfe : VF[v2])
		n += compute_test_area_vector(vfe.first, v2);

	n.normalize();

	std::set<int> shared = shared_faces(VF, v1, v2);

	std::vector<Vector3> updated_normals;
	for (const VFEntry& vfe : VF[v1])
		if (shared.find(vfe.first) == shared.end())
			updated_normals.push_back(compute_test_area_vector(vfe.first, v1).normalized());
	for (const VFEntry& vfe : VF[v2])
		if (shared.find(vfe.first) == shared.end())
			updated_normals.push_back(compute_test_area_vector(vfe.first, v2).normalized());

	for (uint32_t i = 0; i < updated_normals.size(); ++i)
		for (uint32_t j = i + 1; j < updated_normals.size(); ++j)
			if (vector_angle(updated_normals[i], updated_normals[j]) > max_vertex_ring_normals_angle)
				return false;

	return true;
}

bool collapse_preserves_vertex_ring_normals(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	double max_vertex_ring_normals_angle)
{
	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	auto compute_test_area_vector = [&](int fi) -> Vector3 {
		Assert(fi != flap.f[0] && (flap.size() == 1 || fi != flap.f[1]));

		Vector3 p0 = (F(fi, 0) == v1 || F(fi, 0) == v2) ? vpos : V.row(F(fi, 0));
		Vector3 p1 = (F(fi, 1) == v1 || F(fi, 1) == v2) ? vpos : V.row(F(fi, 1));
		Vector3 p2 = (F(fi, 2) == v1 || F(fi, 2) == v2) ? vpos : V.row(F(fi, 2));

		return compute_area_vector(p0, p1, p2);
	};

	auto compute_original_area_vector = [&](int fi) -> Vector3 {
		//Assert(fi != flap.f[0] && (flap.size() == 1 || fi != flap.f[1]));

		Vector3 p0 = V.row(F(fi, 0));
		Vector3 p1 = V.row(F(fi, 1));
		Vector3 p2 = V.row(F(fi, 2));

		return compute_area_vector(p0, p1, p2);
	};

	std::set<int> vset;

	for (int vi : { v1, v2 }) {
		const std::set<VFEntry> vf = VF[vi];
		for (const VFEntry vfe : vf) {
			for (int j = 0; j < deg; ++j)
				vset.insert(F(vfe.first, j));
		}
	}

	//Assert(vset.count(v1) > 0);
	//Assert(vset.count(v2) > 0);

	std::map<int, std::vector<Vector3>> area_vectors;
	for (int vi : vset) {
		area_vectors[vi].reserve(16);
		for (const VFEntry vfe : VF[vi]) {
			if (vfe.first != flap.f[0] && (flap.size() == 1 || vfe.first != flap.f[1]))
				area_vectors[vi].push_back(compute_test_area_vector(vfe.first));
		}
	}

	area_vectors[v1].insert(area_vectors[v1].end(), area_vectors[v2].begin(), area_vectors[v2].end());
	vset.erase(v2);

	std::set<int> vfail;

	for (int vi : vset)
		for (uint32_t i = 0; i < area_vectors[vi].size(); ++i)
			for (uint32_t j = i + 1; j < area_vectors[vi].size(); ++j)
				if (vector_angle(area_vectors[vi][i], area_vectors[vi][j]) > max_vertex_ring_normals_angle)
					vfail.insert(vi);

	return vfail.size() == 0;

	if (vfail.size() == 0) {
		return true;
	}
	else {
		for (int vi : vfail) {
			std::vector<Vector3> og_area_vectors;
			og_area_vectors.reserve(16);
			for (const VFEntry vfe : VF[vi]) {
				//if (vfe.first != flap.f[0] && (flap.size() == 1 || vfe.first != flap.f[1]))
				og_area_vectors.push_back(compute_original_area_vector(vfe.first));
			}

			// if the vertex already violated the constraint to begin with, keep going
			// otherwise, return false (a flat vertex became sharp)
			bool sharp = false;
			for (uint32_t i = 0; i < og_area_vectors.size() && !sharp; ++i)
				for (uint32_t j = i + 1; j < og_area_vectors.size() && !sharp; ++j)
					if (vector_angle(og_area_vectors[i], og_area_vectors[j]) > max_vertex_ring_normals_angle)
						sharp = true;

			if (!sharp) {
				return false;
			}
		}

		return true;
	}
}

// tests true if the two vertices opposite to the flipping edge are *not* connected
bool flip_preserves_topology(const Flap& flap, const MatrixXi& F, const VFAdjacency& VF)
{
	if (flap.size() != 2)
		return false;

	int f0 = flap.f[0];
	int f1 = flap.f[1];
	int e0 = flap.e[0];
	int e1 = flap.e[1];
	int opp1 = (e0 + 2) % 3;
	int opp2 = (e1 + 2) % 3;

	int v_opp1 = F(f0, (e0 + 2) % 3);
	int v_opp2 = F(f1, (e1 + 2) % 3);

	for (const VFEntry& vfe : VF[v_opp1]) {
		int fi = vfe.first;
		for (int i = 0; i < 3; ++i)
			if (F(fi, i) == v_opp2)
				return false;
	}

	return true;
}

bool flip_preserves_geometry(const Flap& flap, const MatrixX& V, const MatrixXi& F)
{
	int deg = F.cols();

	Assert(flap.size() == 2);

	int f1 = flap.f[0];
	int f2 = flap.f[1];
	int e1 = flap.e[0];
	int e2 = flap.e[1];

	Vector3 v0 = V.row(F(f1, e1));
	Vector3 v1 = V.row(F(f1, (e1 + 1) % deg));
	Vector3 v2 = V.row(F(f1, (e1 + 2) % deg));
	Vector3 v3 = V.row(F(f2, (e2 + 2) % deg));

	Vector3 n1pre = (v1 - v0).cross(v2 - v0).normalized();
	Vector3 n2pre = (v0 - v1).cross(v3 - v1).normalized();

	if (n1pre.dot(n2pre) < 0.8)
		return false;

	Vector3 n1post = (v1 - v3).cross(v2 - v3).normalized();
	Vector3 n2post = (v0 - v2).cross(v3 - v2).normalized();

	if (n1pre.dot(n1post) < 0.3 || n2pre.dot(n2post) < 0.3)
		return false;

	return true;
}

bool flip_preserves_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF)
{
	Vector4i ring = vertex_ring(flap, F);

	Scalar ar0 = aspect_ratio(V.row(ring(0)), V.row(ring(1)), V.row(ring(3)));
	Scalar ar1 = aspect_ratio(V.row(ring(1)), V.row(ring(2)), V.row(ring(3)));
	Scalar ar0_new = aspect_ratio(V.row(ring(0)), V.row(ring(1)), V.row(ring(2)));
	Scalar ar1_new = aspect_ratio(V.row(ring(0)), V.row(ring(2)), V.row(ring(3)));

	return std::min(ar0, ar1) < std::min(ar0_new, ar1_new);
}

bool flip_improves_diagonal_split(const Flap& flap, const MatrixX& V, const MatrixXi& F, Scalar angle_threshold)
{
	Vector4 angles = flap_angles(flap, V, F);
	return (angles(0) >= angles.maxCoeff() || angles(2) >= angles.maxCoeff()) && (angles.maxCoeff() > angle_threshold);
}

// -- Qualitative tests (soft constraints) --

// Returns the pair <Correlation, Weight> for the given face fi and ring vertex vi
// (i.e., it is assumed that fi is adjacent to vi)
static inline std::pair<Scalar, Scalar> compute_correlation(const Flap& flap, const MatrixX&V, const MatrixXi& F, int fi, int vi, const Vector3& vpos)
{
	// if on a collapsed face, simply return zero
	if (fi == flap.f[0] || flap.size() > 1 && fi == flap.f[1]) 
		return std::make_pair(Scalar(0), Scalar(0));

	Vector3 area_vector_pre = compute_area_vector(V.row(F(fi, 0)), V.row(F(fi, 1)), V.row(F(fi, 2)));
	
	Vector3 area_vector_post = compute_area_vector(
		F(fi, 0) == vi ? vpos : V.row(F(fi, 0)),
		F(fi, 1) == vi ? vpos : V.row(F(fi, 1)),
		F(fi, 2) == vi ? vpos : V.row(F(fi, 2))
	);

	Scalar correlation = area_vector_pre.normalized().dot(area_vector_post.normalized());
	Scalar weight = area_vector_post.norm();

	return std::make_pair(correlation * weight, weight);
}

Scalar compute_collapse_normal_correlation(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos)
{
	Assert(flap.size() > 0);

	typedef std::pair<Scalar, Scalar> Correlation; // normal correlation and area weight

	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	Scalar correlation = 0;
	Scalar weight = 0;

	for (const VFEntry& vfe : VF[v1]) {
		std::pair<Scalar, Scalar> cw = compute_correlation(flap, V, F, vfe.first, v1, vpos);
		correlation += cw.first;
		weight += cw.second;
	}

	for (const VFEntry& vfe : VF[v2]) {
		std::pair<Scalar, Scalar> cw = compute_correlation(flap, V, F, vfe.first, v1, vpos);
		correlation += cw.first;
		weight += cw.second;
	}

	return correlation / weight;
}

static inline std::pair<Scalar, Scalar> compute_correlation(const Flap& flap, const MatrixX&V, const MatrixXi& F, int fi, int vi, const Vector3& vpos, const Vector3& fn_pre)
{
	// if on a collapsed face, simply return zero
	if (fi == flap.f[0] || flap.size() > 1 && fi == flap.f[1]) 
		return std::make_pair(Scalar(1), Scalar(0));

	Vector3 area_vector_post = compute_area_vector(
		F(fi, 0) == vi ? vpos : V.row(F(fi, 0)),
		F(fi, 1) == vi ? vpos : V.row(F(fi, 1)),
		F(fi, 2) == vi ? vpos : V.row(F(fi, 2))
	);

	Scalar correlation = fn_pre.dot(area_vector_post.normalized());
	Scalar weight = area_vector_post.norm();

	return std::make_pair(correlation, weight);
}

Scalar compute_collapse_normal_correlation_average(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos, const MatrixX& FN)
{
	Assert(flap.size() > 0);

	typedef std::pair<Scalar, Scalar> Correlation; // normal correlation and area weight

	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	Scalar correlation = 0;
	Scalar weight = 0;

	for (const VFEntry& vfe : VF[v1]) {
		std::pair<Scalar, Scalar> cw = compute_correlation(flap, V, F, vfe.first, v1, vpos, FN.row(vfe.first));
		correlation += cw.first * cw.second;
		weight += cw.second;
	}

	for (const VFEntry& vfe : VF[v2]) {
		std::pair<Scalar, Scalar> cw = compute_correlation(flap, V, F, vfe.first, v1, vpos, FN.row(vfe.first));
		correlation += cw.first * cw.second;
		weight += cw.second;
	}

	return correlation / weight;
}

Scalar compute_collapse_normal_correlation(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos, const MatrixX& FN)
{
	Assert(flap.size() > 0);

	typedef std::pair<Scalar, Scalar> Correlation; // normal correlation and area weight

	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	Scalar correlation = 1;

	for (const VFEntry& vfe : VF[v1]) {
		std::pair<Scalar, Scalar> cw = compute_correlation(flap, V, F, vfe.first, v1, vpos, FN.row(vfe.first));
		correlation = std::min(correlation, cw.first);
	}

	for (const VFEntry& vfe : VF[v2]) {
		std::pair<Scalar, Scalar> cw = compute_correlation(flap, V, F, vfe.first, v1, vpos, FN.row(vfe.first));
		correlation = std::min(correlation, cw.first);
	}

	return correlation;
}

static Scalar compute_resulting_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, int fi, int vi, const Vector3& vpos)
{
	// return 1 if the face disappears after the collapse
	if (fi == flap.f[0] || (flap.size() > 1 && fi == flap.f[1]))
		return 1;

	Vector3 p0 = F(fi, 0) == vi ? vpos : V.row(F(fi, 0));
	Vector3 p1 = F(fi, 1) == vi ? vpos : V.row(F(fi, 1));
	Vector3 p2 = F(fi, 2) == vi ? vpos : V.row(F(fi, 2));

	return aspect_ratio(p0, p1, p2);
}

Scalar compute_collapse_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos)
{
	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));

	int v1 = e.first;
	int v2 = e.second;

	Scalar min_ar = 1;

	for (const VFEntry& vfe : VF[v1])
		min_ar = std::min(compute_resulting_aspect_ratio(flap, V, F, vfe.first, v1, vpos), min_ar);

	for (const VFEntry& vfe : VF[v2])
		min_ar = std::min(compute_resulting_aspect_ratio(flap, V, F, vfe.first, v2, vpos), min_ar);

	return min_ar;
}

static inline std::pair<Scalar, Scalar> compute_resulting_aspect_ratio_and_area(
	const Flap& flap, const MatrixX& V, const MatrixXi& F, int fi, int vi, const Vector3 vpos)
{
	Vector3 p0 = F(fi, 0) == vi ? vpos : V.row(F(fi, 0));
	Vector3 p1 = F(fi, 1) == vi ? vpos : V.row(F(fi, 1));
	Vector3 p2 = F(fi, 2) == vi ? vpos : V.row(F(fi, 2));

	return std::make_pair(aspect_ratio(p0, p1, p2), compute_area_vector(p0, p1, p2).norm());
};

Scalar compute_collapse_area_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos)
{
	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));
	
	int v1 = e.first;
	int v2 = e.second;

	std::pair<Scalar, Scalar> w_ar = std::make_pair(0, 0);

	for (const VFEntry& vfe : VF[v1]) {
		std::pair<Scalar, Scalar> ar = compute_resulting_aspect_ratio_and_area(flap, V, F, vfe.first, v1, vpos);
		w_ar.first += ar.first * ar.second;
		w_ar.second += ar.second;
	}

	for (const VFEntry& vfe : VF[v2]) {
		std::pair<Scalar, Scalar> ar = compute_resulting_aspect_ratio_and_area(flap, V, F, vfe.first, v2, vpos);
		w_ar.first += ar.first * ar.second;
		w_ar.second += ar.second;
	}

	return w_ar.first / w_ar.second;
}

std::pair<Vector3, Scalar> compute_collapse_visibility(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos)
{
	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));

	int v1 = e.first;
	int v2 = e.second;

	std::vector<Vector3> area_vectors;

	for (const VFEntry& vfe : VF[v1])
		if (vfe.first != flap.f[0] && (flap.size() < 2 || vfe.first != flap.f[1]))
			area_vectors.push_back(compute_area_vector(
				F(vfe.first, 0) == v1 ? vpos : V.row(F(vfe.first, 0)),
				F(vfe.first, 1) == v1 ? vpos : V.row(F(vfe.first, 1)),
				F(vfe.first, 2) == v1 ? vpos : V.row(F(vfe.first, 2))).normalized()
			);

	for (const VFEntry& vfe : VF[v2])
		if (vfe.first != flap.f[0] && (flap.size() < 2 || vfe.first != flap.f[1]))
			area_vectors.push_back(compute_area_vector(
				F(vfe.first, 0) == v2 ? vpos : V.row(F(vfe.first, 0)),
				F(vfe.first, 1) == v2 ? vpos : V.row(F(vfe.first, 1)),
				F(vfe.first, 2) == v2 ? vpos : V.row(F(vfe.first, 2))).normalized()
			);

	std::pair<Vector3, Scalar> visibility_data = compute_positive_visibility_from_directions(area_vectors);
	visibility_data.second = clamp<Scalar>(visibility_data.second, 0, 1);

	return visibility_data;
}

Scalar compute_collapse_visibility_approx(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos, const MatrixX& VD, const VectorX& VIS)
{
	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));

	int v1 = e.first;
	int v2 = e.second;

	std::map<int, Vector3> normals;

	for (const VFEntry& vfe : VF[v1])
		if (normals.count(vfe.first) == 0)
			normals.insert(std::make_pair(vfe.first, compute_area_vector(
				F(vfe.first, 0) == v1 ? vpos : V.row(F(vfe.first, 0)),
				F(vfe.first, 1) == v1 ? vpos : V.row(F(vfe.first, 1)),
				F(vfe.first, 2) == v1 ? vpos : V.row(F(vfe.first, 2))).normalized()
			));

	for (const VFEntry& vfe : VF[v2])
		if (normals.count(vfe.first) == 0)
			normals.insert(std::make_pair(vfe.first, compute_area_vector(
				F(vfe.first, 0) == v2 ? vpos : V.row(F(vfe.first, 0)),
				F(vfe.first, 1) == v2 ? vpos : V.row(F(vfe.first, 1)),
				F(vfe.first, 2) == v2 ? vpos : V.row(F(vfe.first, 2))).normalized()
			));

	for (int fi : flap.f)
		normals.erase(fi);

	std::vector<Vector3> v(normals.size());
	for (const auto& entry : normals)
		v.push_back(entry.second);

	std::pair<Vector3, Scalar> visibility_data = compute_positive_visibility_from_directions(v);

	Scalar worst_visibility = 1;

	for (const auto& entry : normals) {
		int fi = entry.first;
		const Vector3& n = entry.second;
		for (int i = 0; i < 3; ++i) {
			if (VIS(F(fi, i)) >= 0) {
				const Vector3& d = (F(fi, i) == v1 || F(fi, i) == v2) ? visibility_data.first : VD.row(F(fi, i));
				worst_visibility = std::min(worst_visibility, n.dot(d));
			}
		}
	}

	return clamp<Scalar>(worst_visibility, 0, 1);
}

Scalar compute_collapse_visibility_approx2(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos, const MatrixX& VD, const VectorX& VIS)
{
	int deg = F.cols();
	Edge e(F(flap.f[0], flap.e[0]), F(flap.f[0], (flap.e[0] + 1) % deg));

	int v1 = e.first;
	int v2 = e.second;

	std::map<int, Vector3> normals;

	for (const VFEntry& vfe : VF[v1])
		if (normals.count(vfe.first) == 0)
			normals.insert(std::make_pair(vfe.first, compute_area_vector(
				F(vfe.first, 0) == v1 ? vpos : V.row(F(vfe.first, 0)),
				F(vfe.first, 1) == v1 ? vpos : V.row(F(vfe.first, 1)),
				F(vfe.first, 2) == v1 ? vpos : V.row(F(vfe.first, 2))).normalized()
			));

	for (const VFEntry& vfe : VF[v2])
		if (normals.count(vfe.first) == 0)
			normals.insert(std::make_pair(vfe.first, compute_area_vector(
				F(vfe.first, 0) == v2 ? vpos : V.row(F(vfe.first, 0)),
				F(vfe.first, 1) == v2 ? vpos : V.row(F(vfe.first, 1)),
				F(vfe.first, 2) == v2 ? vpos : V.row(F(vfe.first, 2))).normalized()
			));

	for (int fi : flap.f)
		normals.erase(fi);

	Scalar worst_visibility = 1;

	for (const auto& entry : normals) {
		int fi = entry.first;
		const Vector3& n = entry.second;

		// test the entire set against both v1 and v2 since they will be collapsed
		if (VIS(v1) > 0)
			worst_visibility = std::min(worst_visibility, n.dot(VD.row(v1)));
		if (VIS(v2) > 0)
			worst_visibility = std::min(worst_visibility, n.dot(VD.row(v2)));

		// also test against the face vertices vertices
		for (int i = 0; i < 3; ++i) {
			if (VIS(F(fi, i)) > 0) {
				worst_visibility = std::min(worst_visibility, n.dot(VD.row(F(fi, i))));
			}
		}
	}

	return clamp<Scalar>(worst_visibility, 0, 1);
}

