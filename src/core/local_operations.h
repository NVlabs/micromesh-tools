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

#include <utility>
#include <set>

struct Quadric;

enum LocalOperationStatus {
	Ok = 0,
	FailTopology = -1,
	FailGeometry = -2,
	FailAspectRatio = -3
};

struct CollapseInfo {
	bool ok;
	int vertex;
	int collapsed_vertex; // the vertex that is discarded
	std::set<Edge> expired;
};

struct FlipInfo {
	bool ok;
};

// if ok == false the rest of the data is meaningless
struct SplitInfo {
	bool ok;
	Edge e;
	int u;
	int f1;
	int f2;
	int f12;
	int f22;
	Vector3 split_position;
};

struct VertexSplitInfo {
	bool ok;
	int f1;
	int f2;
	int f1_old;
	int f2_old;
	int old_vertex;
	int new_vertex;
};

// -- Local operations --

namespace detail {
	// Low-level implementation of split_edge that takes explicitl indices for the new vertex and the new face(s) added by the split
	// to avoid explicit reallocation of the mesh buffers
	// u is the new vertex index
	// f12 is the first face index
	// f22 is the second face index (unused if flap.size() == 1)
	// It is responsibility of the caller to ensure the supplied indices are 'free' (unreferenced vertex and deleted or unititialized faces)
	SplitInfo split_edge(const Flap& flap, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, int u, int f12, int f22);
}
	
CollapseInfo collapse_edge(const Flap& flap, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, const Vector3& vertex_position);

FlipInfo flip_edge(const Flap& flap, const MatrixX& V, MatrixXi& F, VFAdjacency& VF);

SplitInfo split_edge(const Flap& flap, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB);

VertexSplitInfo split_vertex(int vi, Edge e1, Edge e2, MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB);

// -- Full mesh pass --

FlipInfo flip_edge(const MatrixX& V, MatrixXi& F, Edge e, VFAdjacency& VF, const Quadric& edge_quadric, Scalar err_threshold);
FlipInfo flip_edge(const MatrixX& V, MatrixXi& F, Edge e, VFAdjacency& VF);

// -- Qualitative tests (hard constraints) --

bool collapse_preserves_topology(const Flap& flap, const MatrixXi& F, const VFAdjacency& VF, VectorXu8& VB);
bool collapse_preserves_geometry(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	double min_cos_theta);

// Ensure the collapse does not change the face normals too much
// We specify 2 different deviation threshold for internal and border faces, so border faces can
// optionally use a tighter deviation bound. This can help preserve the orientation of the mesh border
// and mitigate the distortion of the displaced micro-geometry
bool collapse_preserves_geometry(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	Scalar orientation_threshold, Scalar border_orientation_threshold, const VectorXu8& VB, const MatrixX& FN);

// If collapsing between a border vertex and an internal vertex, make sure all the faces incident on the internal vertex
// are already border faces.
// the idea is that we are only allowing border faces to propagate into the surface, so that when we test the orientation
// of border faces we are sure that they are indeed original border faces.
bool collapse_preserves_orientation_topology(const Flap& flap, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB);

// Aspect ratio test with adaptive thresholding
bool collapse_preserves_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	double min_ratio, const VectorX& FR, Scalar tolerance);

// Cone of face normal test (should be replaced by the visibility estimation)
bool collapse_preserves_vertex_ring_normals(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos,
	double max_vertex_ring_normals_angle);

bool flip_preserves_topology(const Flap& flap, const MatrixXi& F, const VFAdjacency& VF);
bool flip_preserves_geometry(const Flap& flap, const MatrixX& V, const MatrixXi& F);
bool flip_preserves_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF);
bool flip_improves_diagonal_split(const Flap& flap, const MatrixX& V, const MatrixXi& F, Scalar angle_threshold);

// -- Quantitative tests (soft constraints) ----------------

// Returns the face normal correlation after the collapse
// The correlation is computed as the area-weighted sum of the normal correlation of the resulting faces
// The second version uses cached original face normals
Scalar compute_collapse_normal_correlation(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos);
Scalar compute_collapse_normal_correlation(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos, const MatrixX& FN);

// Returns the **minimum** aspect ratio that is produced by the collapse
Scalar compute_collapse_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos);

// Returns the aspect ratio integrated over the area surrounding the collapse
Scalar compute_collapse_area_aspect_ratio(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos);

// Returns the vertex visibility after the collapse
std::pair<Vector3, Scalar> compute_collapse_visibility(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos);

// Returns the minimum visibility induced by any of the updated faces after the collapse
// Visibility is computed with respect to either the new vertex or the old immediate neighbors.
// For neighboring vertices, it **does  not** update the visibility directions so the test is conservative
// (i.e. some face may induce negative visibility with the old direction, but positive if the direction were to be
// updated with the new face normals)
// Vertices that had negative visibility to begin with are ignored
Scalar compute_collapse_visibility_approx(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos, const MatrixX& VD, const VectorX& VIS);
Scalar compute_collapse_visibility_approx2(const Flap& flap, const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const Vector3& vpos, const MatrixX& VD, const VectorX& VIS);
