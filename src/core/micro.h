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
#include "bvh.h"
#include "utils.h"

#include <vector>
#include <functional>

enum class RoundMode {
	Down,
	Up
};

enum SubdivisionDecimationFlag {
	Edge0 = 1 << 0,
	Edge1 = 1 << 1,
	Edge2 = 1 << 2
};

struct DisplaceInfo {
	long n_rays = 0;
	long n_miss = 0;
	long n_fallback = 0;
	long n_backward = 0;
};

struct RunningStats {
	int n = 0;
	Scalar minval = std::numeric_limits<Scalar>::max();
	Scalar maxval = std::numeric_limits<Scalar>::lowest();
	Scalar partial_sum = 0;

	void add(Scalar s) {
		minval = std::min(minval, s);
		maxval = std::max(maxval, s);
		partial_sum += s;
		n++;
	}

	Scalar min() const { return minval; }

	Scalar max() const { return maxval; }

	Scalar avg() const { return partial_sum / (Scalar)n; }
};

// Utility class to compute common quantities on a barycentric grid
// such as linearization of indices, barycentric coordinates of grid points
// etc...
// Assumes the grid is used to compute a regular subdivision of a triangle
//
// LEVEL     NV     NE              V0
//     0      2      1              /\       
//     1      3      2             /\/\      
//     2      5      4            /\/\/\     
//     3      9      8        V1 /\/\/\/\ V2
//
class BarycentricGrid {
	// grid dimension (side, the grid is always a square)
	int n;
	// number of edges per side
	int ne;

public:

	BarycentricGrid(uint8_t subdivision_level)
		: n((1 << subdivision_level) + 1), ne(1 << subdivision_level)
	{
	}

	// 'linearized' index of a lower-triangular matrix: <number of elements up to the previous row> + <current column index>
	// note: the number of elems in a K-sized lower-triangular matrix is K * (K+1) / 2
	// note: the number of lower-triangular elements of a KxK matrix is K * (K-1) / 2
	int index(int i, int j) const
	{
		Assert(valid_index(i, j));
		return (i * (i + 1)) / 2 + j;
	};

	// this formula can be derived from the formulas for triangular numbers and triangular roots
	void inverted_index(int linear_index, int* i, int* j) const
	{
		Assert(linear_index < num_samples());
		*i = int(std::floor(((-1 + std::sqrt(1 + 8 * linear_index)) / 2))); // floor
		*j = linear_index - ((*i * (*i + 1)) / 2);
	}

	// returns true if the coordinates are within the grid
	bool valid_index(int i, int j) const
	{
		return i >= 0 && j >= 0 && i < n && j <= i;
	}

	// Given a grid position, returns the barycentric coordinates relative to the
	// vertices
	// These can be found by moving on the grid using direction vectors
	//   vr = (v1 - v0) / ne
	//   vc = (v2 - v1) / ne
	// Then, the vertex at position (i, j) is given by v0 + i * vr + j * vc
	Vector3 barycentric_coord(int i, int j) const
	{
		return Vector3(1 -  i / Scalar(ne), i / Scalar(ne) - j / Scalar(ne), j / Scalar(ne));
	}

	Vector3 barycentric_coord(int vi) const
	{
		int i, j;
		inverted_index(vi, &i, &j);
		return barycentric_coord(i, j);
	}

	// Grid dimension (side)
	int samples_per_side() const
	{
		return n;
	}

	int edges_per_side() const
	{
		return ne;
	}

	// Grid samples
	int num_samples() const
	{
		return (n * (n - 1)) / 2 + n;
	}
};

// micro-mesh face
// Displacement vectors are stored explicitly for now (no offsets)
struct SubdivisionTri {

	int base_fi;
	uint8_t subdivision_bits;

	Matrix3 base_V;
	Matrix3 base_VD;

	MatrixX V;
	MatrixXi F;
	MatrixX VN;

	// Displacement vectors (per micro-vertex)
	MatrixX VD;

	// Quality
	VectorX FQ;
	VectorX VQ;

	// Border flags
	Vector3u8 border_e;
	Vector3u8 border_v;

	// base vertex ray thresholds
	Vector3 ray_threshold;

	// Mask bits. If VM(micro_vi) == 1 then the displacement was clipped
	VectorXu8 VM;

	// Referenced vertices. If ref(micro_vi) == 0 then the vertex is not
	// referenced by any micro-face (because of edge decimation flags)
	VectorXu8 ref;

	SubdivisionTri(int fi)
		: base_fi(fi), subdivision_bits(0), base_V(Matrix3::Zero()), base_VD(Matrix3::Zero()),
		border_e(Vector3u8::Zero()), border_v(Vector3u8::Zero())
	{
	}

	bool is_border_edge(int ei) const
	{
		return border_e(ei);
	}

	bool is_border_vertex(int vi) const
	{
		return border_e(vi);
	}

	uint8_t subdivision_level() const
	{
		return subdivision_bits >> 3;
	}

	Vector3 interpolate_direction(const Vector3& bary) const
	{
		return base_VD.row(0) * bary(0) + base_VD.row(1) * bary(1) + base_VD.row(2) * bary(2);
	}

	Scalar scalar_displacement(int vi) const
	{
		Vector3 bary = compute_bary_coords(V.row(vi), base_V.row(0), base_V.row(1), base_V.row(2));
		Vector3 vd = interpolate_direction(bary);
		Scalar d = vd.dot(VD.row(vi)) / vd.squaredNorm();
		if (std::isnan(d) || !std::isfinite(d)) {
			d = 0;
		}
		return d;
	}

	void subdivide(uint8_t bits, const Vector3 v0, const Vector3 v1, const Vector3 v2);
	DisplaceInfo displace(const BVHTree& bvh, const BVHTree::IntersectionFilter& bvh_test,
		const BVHTree& bvh_border, bool interpolate);

	void displace(const std::vector<Scalar>& micro_displacements);

	void _interpolate_culled_displacements();

	// returns true if subdiv. level before upsampling is < 8
	bool _upsample(const BVHTree& bvh, Scalar ray_clipping_threshold, const std::function<bool(IntersectionInfo)>& bvh_test);

	void _ray_cast(const std::vector<int>& indices, const BVHTree& bvh, Scalar ray_clipping_threshold, const std::function<bool(IntersectionInfo)>& bvh_test);

	void coarsen_to_level(uint8_t bits);

	RunningStats downsampling_error() const;
	Scalar max_downsampling_error() const;
	Scalar avg_downsampling_error() const;

	// extracts the boundary edges as a polyline
	void extract_boundary_polyline(MatrixX& VP, std::vector<int>& FP) const;
};

struct SubdivisionMesh {

	// low-level control of culling and interpolation of micro-displacements
	// to remove outliers and wrong ray intersections

	// controls the culling of micro-displacements
	bool _interpolate = true;

	constexpr static uint8_t Subdivided = 1 << 0;
	constexpr static uint8_t Displaced = 1 << 1;
	constexpr static uint8_t FaceQuality = 1 << 2;
	constexpr static uint8_t VertexQuality = 1 << 3;

	int base_fn = 0;
	int64_t micro_fn = 0;

	uint8_t _status_bits = 0;

	std::vector<SubdivisionTri> faces;

	// Compute the subdivision mesh structure (vertices and faces)
	// according to the prescribed base mesh subdivision
	void compute_mesh_structure(const MatrixX& base_V, const MatrixXi& base_F, const VectorXu8& subdivision_bits);
	void compute_mesh_structure(const MatrixX& base_V, const MatrixXi& base_F, uint8_t subdivision_level);

	// Compute the displacements from base displacement vectors
	// and high-resolution mesh data
	void compute_micro_displacements(
		const MatrixX& base_V,
		const MatrixX& base_VD,
		const MatrixXi& base_F,
		const BVHTree& bvh,
		const std::function<bool(IntersectionInfo)>& bvh_test,
		const MatrixX& border_V,
		const MatrixXi& border_F,
		const MatrixX& border_VN);

	// Selectively compute displacements for base_f, skipping faces
	// for which update(base_fi) evaluates to false
	void compute_micro_displacements(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN, const VectorXi& update,
		const MatrixX& border_V, const MatrixXi& border_F, const MatrixX& border_VN);

	// compute displacements from displacement directions and scalar displacements (no ray-casting)
	void compute_micro_displacements(const MatrixX& base_VD, const MatrixXi& base_F, const std::vector<std::vector<Scalar>>& micro_displacements);

	// compute displacements using a pre-computed BVHTree of the high-resolution mesh
	void compute_micro_displacements(const BVHTree& bvh, const std::function<bool(IntersectionInfo)>& bvh_test,
		const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, const VectorXi& update,
		const MatrixX& border_V, const MatrixXi& border_F, const MatrixX& border_VN);

	// Zeroes the per-microvertex displacement vectors and the corresponding bit flag
	void clear_micro_displacements();

	// Update the micromesh while preserving the current displacement state
	void update_mesh_structure(const VectorXu8& subdivision_bits, const MatrixX& base_V, const MatrixXi& base_F,
		const BVHTree& bvh, const std::function<bool(IntersectionInfo)>& bvh_test, const MatrixX& border_V, const MatrixXi& border_F, const MatrixX& border_VN);

	// Ensures water-tightness by averaging microdisplacements at base-mesh primitives
	void stitch_primitives(const MatrixX& base_V, const MatrixXi& base_F);

	// Recompute vertex positions and displacements so that all
	// displacements vectors are positive. This changes the vertex positions
	// of the 'flat' (non-displaced) subdivision mesh
	// returns a new base mesh with these attributes
	void normalize_displacements(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, MatrixX& new_base_V, MatrixX& new_base_VD);

	// Minimizes the norm of base displacement directions using an aggregation-based heuristic
	void minimize_base_directions_length(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, MatrixX& new_base_V, MatrixX& new_base_VD, MatrixX& top_VD, MatrixX& bottom_VD);
	void minimize_base_directions_length_with_shrinking(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, MatrixX& new_base_V, MatrixX& new_base_VD, MatrixX& top_VD, MatrixX& bottom_VD);

	// Computes vertex quality from displacement distances (absolute value of the scalar displacements)
	void compute_vertex_quality_displacement_distance();

	// Comptes face quality from aspect ratio of micro-triangles
	void compute_face_quality_aspect_ratio();

	// Computes face quality from displacement stretch deformation (min/max singular value of the micro-triangle deformations)
	void compute_face_quality_stretch();

	// Returns true if the subdivision structure has been computed
	bool is_subdivided() const { return _status_bits & Subdivided; }

	// Returns true if the displacements have been computed
	bool is_displaced() const { return _status_bits & Displaced; }

	// Returns true if the face quality has been computed
	bool has_face_quality() const { return _status_bits & FaceQuality; }

	// Returns true if the vertex quality has been computed
	bool has_vertex_quality() const { return _status_bits & VertexQuality; }

	void extract_mesh(MatrixX& V, MatrixXi& F) const;
	void extract_mesh_with_uvs(const MatrixX& base_UV, const MatrixX& base_VN, const MatrixX& base_VT, const MatrixXi& base_F,
		MatrixX& V, MatrixXi& F, MatrixX& UV, MatrixX& VN, MatrixX& VT) const;

	void _debug_displacements_minmax();
	void _debug_reproject_displacements(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F);

	void _quantize_displacements(int nbits);

};

// Compute per-face subdivision levels according to the prescribed triangle area
//VectorXu8 compute_face_subdivision_bits(const MatrixX& V, const MatrixXi& F, Scalar average_area_inv);
//VectorXu8 compute_face_subdivision_bits(const MatrixX& V, const MatrixXi& F, Scalar average_area_inv, Scalar global_subdivision_level, VectorXi& area_multiplier, RoundMode rm);

VectorXu8 compute_subdivision_levels_constant(const MatrixXi& F, uint8_t level);

VectorXu8 compute_subdivision_levels_uniform_area(const MatrixX& V, const MatrixXi& F, Scalar average_area_inv, Scalar global_subdivision_level);

VectorXu8 compute_mesh_subdivision_adaptive(
	const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F,
	const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN, int target_ufn, Scalar target_max_error);

// adjust level correction vector to ensure distance is at most 1 across edges
// returns a vector of edge-decimation flags
VectorXu8 adjust_subdivision_levels(const MatrixXi& F, const VectorXu8& subdivisions, VectorXi8& corrections, RoundMode rm);

// Compute UV texture coordinates to visualize subdivisions
void compute_subdivision_texture(const MatrixXi& F, const VectorXi& subdivision, MatrixX& VT, MatrixXi& FT);

// Compute triangle subdivision according to the given subdivision bits
// the 5 most significant bits encode the subdivision level, the remaining
// 3 bits encode the per-edge decimation flags
// Subdivision is computed on a barycentric grid, using the utility class defined in this file
// The grid indices (and consequently the linearized indices) of the base-level vertices are defined
// in the following way:
//    v0 -> (0, 0)
//    v1 -> (0, n-1)
//    v2 -> (n-1, n-1)
// where n is the number of samples per edge (2 ad level 0, 3 at level 1, 5 at level 2 and so on...)
// regardless of the decimation bits
// TODO this function should return the indices of v0, v1, and v2 explicitly...
// TODO this function can probably be simplified a lot by first computing a regular subdivision and then
//      dealing with the decimation flags...
void subdivide_tri(const Vector3& v0, const Vector3& v1, const Vector3& v2, MatrixX& V, MatrixXi& F, uint8_t subdivision_bits);

// Compute per vertex projection vectors from the first mesh to the
// corresponding closest point on the second mesh
// Projection vectors are returned in matrix form (one row per vertex projection)
MatrixX per_vertex_nearest_projection_vectors(const MatrixX& from_V, const MatrixXi& from_F, const MatrixX& to_V, const MatrixXi& to_F);

// Flip displacement directions to agree with the base mesh vertex normals
void correct_directions_with_normals(MatrixX& VD, const MatrixX& base_VN);

// Compute displacement directions by averaging sampled normals in the voronoi
// regions around base mesh vertices
//MatrixX per_vertex_displacements_from_sampled_normals(const SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F);
MatrixX per_vertex_displacements_from_micro_normals(const SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F);

// TODO FIXME this function should be the combination of the two below?
MatrixX per_vertex_averaged_micro_nearest_projection_vectors(const SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN);

MatrixX per_vertex_base_displacements_from_micro_displacements(SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN);

void optimize_base_mesh_positions_and_displacements(SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F, MatrixX& new_V, MatrixX& new_VD);

MatrixX optimize_base_mesh_positions_ls(const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN);
MatrixX optimize_base_mesh_displacements_alternating(const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN, int iterations);
MatrixX optimize_base_mesh_displacements_averaging(const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN);

