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

#include <vector>
#include <set>

// adjacency.h
// 
// This file implements the vertex-face adjacency (VF) and some related utility functions
// 
// The adjacency is stored as a vector of std::set<VFEntry>, one for each vertex `vi`
// 
// Each VFEntry is a <face index, face-vertex index> pair where the face index is the
// index of a face adjacent to the vertex `vi`, and the face-vertex index is the index
// within that face that references vertex `vi`. i.e., if F is a face index matrix and 
// `vfe` is any VFEntry in `VF[vi]`, then
// 
//   F(vfe.first, vfe.second) == vi
// 
// Faces that share a common edge are sometimes collected in a Flap object. 
// The Vertex ring ordering over a 2-manifold flap storing a pair of faces
// (i.e., a quad) is defined as
//
//          0 /\
//           /  \
//          /    \
//         /  f0  \
//        /        \
//       1 -------- 3
//        \        /
//         \  f1  /
//          \    /
//           \  /
//            \/ 2
//

// A Vertex-Face adjacency entry.
// VFEntry.first is the index of the face referencig a particular vertex index
// VFEntry.second is the index within that face (i.e. 0, 1 or 2) that references
// the vertex
struct VFEntry : public std::pair<int, int8_t> {

	VFEntry(int f, int8_t s) : std::pair<int, int8_t>(f, s) {}

	// returns the face index
	int fi() const { return first; }

	// returns the vertex index within the face
	int8_t vi() const { return second; }
};

// The VFAdjacency is a vector of #V sets of VFEntry objects
typedef std::vector<std::set<VFEntry>> VFAdjacency;

// An Edge-Face adjacency entry.
// EFEntry.first is the index of the adjacent face
// EFEntry.second is the corresponding edge-index within the face
struct EFEntry : public std::pair<int, int8_t> {

	EFEntry(int f, int8_t s) : std::pair<int, int8_t>(f, s) {}

	// returns the face index
	int fi() const { return first; }

	// returns the edge index within the face
	int8_t ei(short i = 0) const { return (second + i) % 3; }
};

// the EFAdjaceny is a map Edge -> list of face indices (EFEntry)
typedef std::map<Edge, std::vector<EFEntry>> EFAdjacency;

// A flap stores a collection of faces sharing a common edge, in no particular order
// The flap is stored as a list of faces and a list of edge indices. For the j-th entry,
// the edge can be retrieved as
//   Edge( F(f[j], e[j]), F(f[j], e[(j + 1) % 3]) )
// where F is a face index matrix

struct Flap {
	// List of faces in the flap
	std::vector<int> f;

	// List of edge indices in the flap
	std::vector<int> e;

	int size() const { return f.size(); }
};

// Computes the VFAdjacency from the face index matrix F
VFAdjacency compute_adjacency_vertex_face(const MatrixX& V, const MatrixXi& F);

// Computes the EFAdjacency from the face index matrix F
EFAdjacency compute_adjacency_edge_face(const MatrixXi& F);

// Returns the set of faces shared by the two vertices v1, v2
std::set<int> shared_faces(const VFAdjacency& VF, int v1, int v2);

// Returns the set of vertices adjacent to vi
std::set<int> vertex_star(int vi, const MatrixXi& F, const VFAdjacency& VF);

// Computes the vertex border flags from the given mesh topology F
void per_vertex_border_flag(const MatrixX& V, const MatrixXi& F, VectorXu8& VB);
VectorXu8 per_vertex_border_flag(const MatrixX& V, const MatrixXi& F, const MatrixXu8& FEB);
void per_vertex_border_flag(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, VectorXu8& VB);

// Computes the face edge border flags from the given mesh topology F
MatrixXu8 per_face_edge_border_flag(const MatrixX& V, const MatrixXi& F);

// Computes the flap of edge e (list of faces sharing e)
Flap compute_flap(Edge e, const MatrixXi& F, const VFAdjacency& VF);

// Computes the vertex ring of the 2-manifold size 2 flap and returns it
// as a 4-dimensional vector of indices.
// Vertex ordering follows the convention described in adjacency.h
Vector4i vertex_ring(const Flap& flap, const MatrixXi& F);

// Computes the planarity of the two-manifold size 2 flap
// The planarity of the flap is defined as 6 times the volume of the spanned tetra
Scalar flap_planarity(const Flap& flap, const MatrixX& V, const MatrixXi& F);

// Returns the angle in radians between the face normals of a 2-manifold size 2 flap
Scalar flap_normals_angle(const Flap& flap, const MatrixX& V, const MatrixXi& F);

// Returns the interior angles of a 2-manifold size 2 flap
Vector4 flap_angles(const Flap& flap, const MatrixX& V, const MatrixXi& F);

// Returns a circle of EFEntry objects around a pivoting vertex
std::vector<EFEntry> compute_face_circle(EFEntry start, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF);

