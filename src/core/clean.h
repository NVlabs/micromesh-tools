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

#include <vector>
#include <set>
#include <unordered_set>

// clean.h
// 
// Utility functions to manage buffers and perform some simple mesh
// cleaning to get rid of degenerate or near-degenerate faces.
// 
// Low level buffer management
//   remove_degenerate_faces()
//   compact_vertex_data()
//   detail::compact_data()
// 
// Mesh cleaning (require data compaction after)
//   remove_thin_triangles() Collapses near-degenerate faces
//   squash_low_valence_vertices() Clears vertices with valence <= 3
//   unify_vertices() Performs vertex unification (welding) to remove topological cuts

// Recursive templates to perform data compaction using a remapped index
namespace detail {
	inline void compact_data(const std::vector<int>& remap, int n_remapped)
	{
		// do nothing, just close the recursion
	}

	template<typename Derived, typename... Args>
	void compact_data(const std::vector<int>& remap, int n_remapped, Eigen::PlainObjectBase<Derived>& mdata, Args&... data);

	template<typename DataType, typename... Args>
	void compact_data(const std::vector<int>& remap, int n_remapped, std::vector<DataType>& vdata, Args&... data);

	template<typename Derived, typename... Args>
	void compact_data(const std::vector<int>& remap, int n_remapped, Eigen::PlainObjectBase<Derived>& mdata, Args&... data)
	{
		if (!mdata.data())
			return;

		Assert((int)remap.size() <= (int)mdata.size());

		for (int i = 0; i < (int)remap.size(); ++i) {
			if (remap[i] != -1) {
				Assert(remap[i] <= i); // ensure that the data is 'pulled back' (linear compaction)
				Assert(remap[i] < (int)mdata.rows());
				mdata.row(remap[i]) = mdata.row(i);
			}
		}
		mdata.conservativeResize(n_remapped, mdata.cols());
		compact_data(remap, n_remapped, data...);
	}

	template<typename DataType, typename... Args>
	void compact_data(const std::vector<int>& remap, int n_remapped, std::vector<DataType>& vdata, Args&... data)
	{
		static_assert(!std::is_same<std::vector<DataType>, VFAdjacency>::value, "Can't compact VFAdjacency if indices change, it must be recomputed");
		Assert(remap.size() <= vdata.size());

		for (int i = 0; i < (int)remap.size(); ++i) {
			if (remap[i] != -1) {
				Assert(remap[i] <= i); // ensure that the data is 'pulled back' (linear compaction)
				vdata[remap[i]] = vdata[i];
			}
		}
		vdata.erase(vdata.begin() + n_remapped, vdata.end());
		compact_data(remap, n_remapped, data...);
	}
} // namespace detail

// Removes degenerate faces (faces with two or more equal indices)
inline void remove_degenerate_faces(MatrixXi& F)
{
	std::vector<uint8_t> degenerate(F.rows(), false);
	int fn = 0;

	for (int i = 0; i < F.rows(); ++i) {
		std::set<int> refs;
		for (int j = 0; j < F.cols(); ++j) {
			refs.insert(F(i, j));
		}
		if (refs.size() < 3)
			degenerate[i] = true;
		else
			fn++;
	}

	if (fn < F.rows()) {
		MatrixXi FF(fn, F.cols());
		int current_row = 0;
		for (int i = 0; i < F.rows(); ++i)
			if (!degenerate[i])
				FF.row(current_row++) = F.row(i);
		F = FF;
	}	
}

// Removes degenerate faces (faces with two or more equal indices)
// Arguments
//   F INOUT The mesh index buffer, updated to filter degenerate faces
//   vertex_data... INOUT List of face buffers to compact
template <typename... Args>
inline std::vector<int> remove_degenerate_faces_inplace(MatrixXi& F, Args&... face_data)
{
	int deg = (int)F.cols();

	auto is_degenerate = [deg](const VectorXi& face) -> bool {
		for (int i = 0; i < deg; ++i)
			for (int j = i + 1; j < deg; ++j)
				if (face(i) == face(j))
					return true;
		return false;
	};

	int fn = 0;
	std::vector<int> remap(F.rows(), -1);
	for (int i = 0; i < (int)remap.size(); ++i) {
		if (!is_degenerate(F.row(i))) {
			remap[i] = fn++;
		}
	}

	detail::compact_data(remap, fn, F, face_data...);

	return remap;
}

// Removes unreferenced vertex data, updating the mesh indexing
// Arguments
//   F INOUT The mesh index buffer, updated after vertex data compaction
//   vertex_data... INOUT List of vertex buffers to compact
template <typename... Args>
std::vector<int> compact_vertex_data(MatrixXi& F, Args&... vertex_data)
{
	int deg = F.cols();
	std::vector<uint8_t> vref(F.maxCoeff() + 1, false);
	for (int i = 0; i < F.rows(); ++i)
		if (F(i, 0) != INVALID_INDEX)
			for (int j = 0; j < deg; ++j)
				vref[F(i, j)] = true;

	int vn = 0;
	std::vector<int> remap(vref.size(), -1);
	for (int i = 0; i < (int)remap.size(); i++)
		if (vref[i])
			remap[i] = vn++;

	// update F
	for (int i = 0; i < F.rows(); ++i)
		if (F(i, 0) != INVALID_INDEX)
			for (int j = 0; j < deg; ++j)
				F(i, j) = remap[F(i, j)];

	detail::compact_data(remap, vn, vertex_data...);

	return remap;
}

// Removes near-degenerate triangles by performing a series of edge flips and edge collapses
// Single pass
// Returns true if the mesh changed (at least one collapse/flip was performed)
bool remove_thin_triangles(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB);

// Removes low-valence (<= 3) internal vertices by collapsing any of their adjacent edges
// Returns the number of removed vertices (i.e. the number of edge collapses)
int squash_low_valence_vertices(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB);

// Collapse edges onto creases
int collapse_onto_creases(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar min_crease_angle);

// Unifies duplicate vertices that are not topologically connected (welding)
void unify_vertices(const MatrixX& V, MatrixXi& F, VectorXu8& VB, MatrixXu8& FEB);

// Removes faces that have zero-area
// This includes both faces that are collapsed to a line/point, as well as
// faces that are topologically degenerate (refrence the same vertex more than once)
// Returns the number of faces removed
int remove_zero_area_faces(const MatrixX& V, MatrixXi& F);

// Performs a split pass over the mesh, splitting edges that are longer
// than 2 * target_len
// The remesh_faces parameter specifies on which faces the pass operates
// Returns the number of splits performed
int split_pass(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar target_len, std::unordered_set<int>& remesh_faces);

// Performs a flip pass over the mesh, flipping edges that improve the
// diagonal split of the resulting quad
// The remesh_faces parameter specifies on which faces the pass operates
// Returns the number of flips performed
int flip_pass(const MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, const std::unordered_set<int>& remesh_faces);

// Performs a collapse pass over the mesh, collapsing edges that are
// shorter than target_len
// The remesh_faces parameter specifies on which faces the pass operates
// Returns the number of collapses performed
int collapse_pass(MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, Scalar target_len, std::unordered_set<int>& remesh_faces);
