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

#include "clean.h"
#include "adjacency.h"
#include "local_operations.h"
#include "mesh_utils.h"
#include "utils.h"

bool remove_thin_triangles(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB)
{
	Assert(F.cols() == 3);

	constexpr Scalar angle_threshold = M_PI - radians(20.0);
	constexpr Scalar edge_angle_threshold = (M_PI - radians(10.0)) / 2.0;

	enum Op { None = 0, Collapse, Flip };

	bool mesh_changed = false;
	for (int fi = 0; fi < (int)F.rows(); ++fi) {

		if (F(fi, 0) == INVALID_INDEX)
			continue;

		Vector3 e[3];

		for (int j = 0; j < 3; ++j)
			e[j] = V.row(F(fi, (j + 1) % 3)) - V.row(F(fi, j));

		// per-vertex angles
		Scalar angle[3] = {
			vector_angle((-e[2]).eval(), e[0]),
			vector_angle((-e[0]).eval(), e[1]),
			vector_angle((-e[1]).eval(), e[2])
		};

		// per edge angles
		std::pair<Scalar, Scalar> edge_angle[3] = {
			std::make_pair(angle[0], angle[1]),
			std::make_pair(angle[1], angle[2]),
			std::make_pair(angle[2], angle[0])
		};

		Edge edge[3] = {
			Edge(F(fi, 0), F(fi, 1)),
			Edge(F(fi, 1), F(fi, 2)),
			Edge(F(fi, 2), F(fi, 0))
		};

		Op op = None;
		Edge op_edge(-1, -1);

		// test for two possible configurations: 1 very large angle or 2 angles that ~ sum to M_PI
		int large_angle_index = std::distance(angle,
			std::find_if(angle, angle + 3, [](const Scalar angle_i) { return angle_i > angle_threshold; }));

		if (large_angle_index < 3) {
			Edge opposite_edge = edge[(large_angle_index + 1) % 3];
			Flap opposite_flap = compute_flap(opposite_edge, F, VF);
			if (opposite_flap.size() == 1) {
				// if opposite edge is border, collapse one of the other two
				Edge next_edge = edge[large_angle_index];
				Flap next_flap = compute_flap(next_edge, F, VF);

				Edge prev_edge = edge[(large_angle_index + 2) % 3];
				Flap prev_flap = compute_flap(prev_edge, F, VF);
				if (collapse_preserves_topology(next_flap, F, VF, VB)) {
					op = Collapse;
					op_edge = next_edge;
				}
				else if (collapse_preserves_topology(prev_flap, F, VF, VB)) {
					op = Collapse;
					op_edge = prev_edge;
				}
			}
			else if (opposite_flap.size() == 2) {
				// else flip opposite edge
				if (flip_preserves_topology(opposite_flap, F, VF)) {
					op = Flip;
					op_edge = opposite_edge;
				}
			}
		}
		else {
			int large_edge_angle_index = std::distance(edge_angle,
				std::find_if(edge_angle, edge_angle + 3, [](const std::pair<Scalar, Scalar>& p) { return std::min(p.first, p.second) > edge_angle_threshold; }));
			if (large_edge_angle_index < 3) {
				// collapse large edge angle index
				Flap angle_flap = compute_flap(edge[large_edge_angle_index], F, VF);
				if (collapse_preserves_topology(angle_flap, F, VF, VB)) {
					op = Collapse;
					op_edge = edge[large_edge_angle_index];
				}
			}
		}

		if (op == Collapse) {
			mesh_changed = true;
			Flap flap = compute_flap(op_edge, F, VF);
			if (flap.size() < 3) // collapse only if manifold
				collapse_edge(flap, V, F, VF, VB, V.row(op_edge.first));
		}
		else if (op == Flip) {
			mesh_changed = true;
			FlipInfo flip = flip_edge(V, F, op_edge, VF);
			Assert(flip.ok);
		}
	}

	return mesh_changed;
}

int squash_low_valence_vertices(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB)
{
	int n = 0;

	for (int vi = 0; vi < V.rows(); ++vi) {
		if (!VB(vi) && VF[vi].size() > 0 && VF[vi].size() <= 3) {
			VFEntry vfe = *(VF[vi].begin());
			int v_squashed = F(vfe.first, vfe.second);
			int v_other = F(vfe.first, (vfe.second + 2) % 3);
			Edge e(v_squashed, v_other);
			Flap flap = compute_flap(e, F, VF);
			if (collapse_preserves_geometry(flap, V, F, VF, V.row(v_other), -0.3)) {
				collapse_edge(flap, V, F, VF, VB, V.row(v_other));
				n++;
			}
		}
	}

	return n;
}

int collapse_onto_creases(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar min_crease_angle)
{
	int n = 0;

	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX) {
			for (int j = 0; j < 3; ++j) {
				Edge e(F(fi, j), F(fi, (j + 1) % 3));
				Flap flap = compute_flap(e, F, VF);
				if (flap.size() == 2 && flap_normals_angle(flap, V, F) > min_crease_angle) {
					Edge collapse_e(F(fi, j), F(fi, (j + 2) % 3));
					Flap collapse_flap = compute_flap(collapse_e, F, VF);
					Vector3 collapse_pos = V.row(F(fi, j));
					if (collapse_preserves_topology(collapse_flap, F, VF, VB) && collapse_preserves_geometry(flap, V, F, VF, collapse_pos, 0.9)) {
						collapse_edge(collapse_flap, V, F, VF, VB, V.row(F(fi, j)));
						n++;
						break;
					}
				}
			}
		}
	}

	return n;
}

void unify_vertices(const MatrixX& V, MatrixXi& F, VectorXu8& VB, MatrixXu8& FEB)
{
	// only unify on border edges (uv cuts)

	// store the border vertices
	std::vector<int> border_vertices;
	for (int i = 0; i < V.rows(); ++i) {
		if (VB(i))
			border_vertices.push_back(i);
	}

	// sort the vertices
	std::sort(border_vertices.begin(), border_vertices.end(), [&](int i, int j) { return V.row(i) < V.row(j); });

	std::vector<Vector3> p;
	for (int i : border_vertices)
		p.push_back(V.row(i));
	
	// remap the vertices (with a map<int, int>)
	std::map<int, int> remap;
	for (int i = 0; i < (int)border_vertices.size(); ) {
		int j;
		for (j = i; j < (int)border_vertices.size() && (V.row(border_vertices[i]) == V.row(border_vertices[j])); ++j)
			remap[border_vertices[j]] = border_vertices[i];
		i = j;
	}

	// update mesh indices
	for (int fi = 0; fi < (int)F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX) {
			for (int i = 0; i < 3; ++i) {
				auto it = remap.find(F(fi, i));
				if (it != remap.end())
					F(fi, i) = it->second;
			}
		}
	}

	// count again border edges;
	std::map<Edge, int> em;
	for (int fi = 0; fi < (int)F.rows(); ++fi) {
		for (int j = 0; j < 3; ++j) {
			if (FEB(fi, j)) {
				Edge e(F(fi, j), F(fi, (j + 1) % 3));
				em[e]++;
			}
		}
	}

	// clear border edge flags for stitched edges
	for (int fi = 0; fi < (int)F.rows(); ++fi) {
		for (int j = 0; j < 3; ++j) {
			if (FEB(fi, j) && em[Edge(F(fi, j), F(fi, (j + 1) % 3))] > 1) {
				FEB(fi, j) = 0;
			}
		}
	}

	// update border flags
	VB = per_vertex_border_flag(V, F, FEB);
}

int remove_zero_area_faces(const MatrixX& V, MatrixXi& F)
{
	int n = 0;
	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX && face_area(V, F.row(fi)) == 0) {
			n++;
			for (int j = 0; j < 3; ++j) {
				F(fi, j) = INVALID_INDEX;
			}
		}
	}
	return n;
}

int split_pass_(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar target_len)
{
	int n = 0;
	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX) {
			for (int i = 0; i < 3; ++i) {
				Edge e(F(fi, i), F(fi, (i + 1) % 3));
				if ((V.row(e.first) - V.row(e.second)).norm() > 2 * target_len) {
					Flap flap = compute_flap(e, F, VF);
					SplitInfo si = split_edge(flap, V, F, VF, VB);
					if (si.ok)
						n++;
				}
			}
		}
	}
	return n;
}

int split_pass_(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar target_len, std::unordered_set<int>& remesh_faces)
{
	int n = 0;
	std::set<Edge> edges; // edges to split

	Timer t;

	// first pass, detect edges to split
	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX && remesh_faces.find(fi) != remesh_faces.end()) {
			for (int i = 0; i < 3; ++i) {
				Edge e(F(fi, i), F(fi, (i + 1) % 3));
				if ((V.row(e.first) - V.row(e.second)).norm() > 2 * target_len)
					edges.insert(e);
			}
		}
	}

	std::cout << "Analysis time: " << t.time_elapsed() << std::endl;

	// second pass, actually split edges
	for (const Edge& e : edges) {
		Flap flap = compute_flap(e, F, VF);
		if (flap.size() == 1 || (flap.size() == 2 && (remesh_faces.count(flap.f[0]) > 0 && remesh_faces.count(flap.f[1]) > 0))) {
			SplitInfo si = split_edge(flap, V, F, VF, VB);
			if (si.ok) {
				n++;
				remesh_faces.insert(si.f12);
				if (flap.size() == 2)
					remesh_faces.insert(si.f22);
			}
		}
	}
	std::cout << "Split time: " << t.time_since_last_check() << std::endl;

	return n;
}

int split_pass(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar target_len, std::unordered_set<int>& remesh_faces)
{
	int n = 0;
	std::set<Edge> edges; // edges to split

	Timer t;

	// first pass, detect edges to split
	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX && remesh_faces.find(fi) != remesh_faces.end()) {
			for (int i = 0; i < 3; ++i) {
				Edge e(F(fi, i), F(fi, (i + 1) % 3));
				if ((V.row(e.first) - V.row(e.second)).norm() > 2 * target_len)
					edges.insert(e);
			}
		}
	}

	std::cout << "Analysis time: " << t.time_elapsed() << std::endl;

	// second pass, actually split edges

	// first pre-allocate data (assume all splits are for flaps of size 2)
	int f_new = F.rows();
	int v_new = V.rows();
	int edges_to_split = edges.size();
	F.conservativeResize(F.rows() + 2 * edges_to_split, 3);
	V.conservativeResize(V.rows() + edges_to_split, 3);
	VF.resize(VF.size() + edges_to_split);
	VB.resize(VB.rows() + edges_to_split);

	for (const Edge& e : edges) {
		Flap flap = compute_flap(e, F, VF);
		if (flap.size() == 1 || (flap.size() == 2 && (remesh_faces.count(flap.f[0]) > 0 && remesh_faces.count(flap.f[1]) > 0))) {
			SplitInfo si = detail::split_edge(flap, V, F, VF, VB, v_new, f_new, f_new + 1);
			Assert(si.ok);
			if (si.ok) {
				n++;
				remesh_faces.insert(si.f12);
				v_new++;
				f_new++;
				if (flap.size() == 2) {
					remesh_faces.insert(si.f22);
					f_new++;
				}
			}
		}
	}
	std::cout << "Split time: " << t.time_since_last_check() << std::endl;

	return n;
}



int flip_pass(const MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, const std::unordered_set<int>& remesh_faces)
{
	Timer t;
	Assert(F.cols() == 3);

	int nflip = 0;
	for (int fi = 0; fi < (int)F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX && remesh_faces.find(fi) != remesh_faces.end()) {
			for (int j = 0; j < 3; ++j) {
				Edge e(F(fi, j), F(fi, (j + 1) % 3));
				Flap flap = compute_flap(e, F, VF);
				bool flip = false;
				if (flap.size() == 2) {
					if (flap_normals_angle(flap, V, F) < radians(30)) {
						if (flip_improves_diagonal_split(flap, V, F, radians(120)))
							flip = true;
					}
					else {
					}

					if (flip && flip_preserves_geometry(flap, V, F) && flip_preserves_topology(flap, F, VF)) {
						FlipInfo info = flip_edge(flap, V, F, VF);
						Assert(info.ok);
						nflip++;
					}
				}
			}
		}
	}

	return nflip;
}

int collapse_pass(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar target_len, std::unordered_set<int>& remesh_faces)
{
	int n = 0;

	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX && remesh_faces.find(fi) != remesh_faces.end()) {
			for (int j = 0; j < 3; ++j) {
				Edge e(F(fi, j), F(fi, (j + 1) % 3));
				Vector3 p0 = V.row(e.first);
				Vector3 p1 = V.row(e.second);
				if ((p1 - p0).norm() < 0.5 * target_len) {
					Flap flap = compute_flap(e, F, VF);
					if (flap.size() == 1 || (flap.size() == 2 && (remesh_faces.count(flap.f[0]) > 0 && remesh_faces.count(flap.f[1]) > 0))) {
						if (collapse_preserves_topology(flap, F, VF, VB)) {
							collapse_edge(flap, V, F, VF, VB, 0.5 * (p1 + p0));
							n++;
							break;
						}
					}
				}
			}
		}
	}

	return n;
}


