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

#include "micro.h"
#include "mesh_utils.h"
#include "bvh.h"
#include "utils.h"
#include "tangent.h"
#include "geodesic.h"
#include "utils.h"
#include "quality.h"

#include "mesh_io.h"

#include <set>
#include <map>
#include <functional>
#include <iostream>
#include <random>
#include <execution>


// -- SubdivisionTri ------------

void SubdivisionTri::subdivide(uint8_t bits, const Vector3 v0, const Vector3 v1, const Vector3 v2)
{
	subdivision_bits = bits;

	base_V.resize(3, 3);
	base_V.row(0) = v0;
	base_V.row(1) = v1;
	base_V.row(2) = v2;

	subdivide_tri(base_V.row(0), base_V.row(1), base_V.row(2), V, F, subdivision_bits);
	VD.resizeLike(V);
	VD.setZero();
	VN.resizeLike(V);
	VN.rowwise() = compute_area_vector(base_V.row(0), base_V.row(1), base_V.row(2)).normalized().transpose();
	
	VM.resize(V.rows());
	VM.setZero();

	ref = VectorXu8::Constant(V.rows(), 0);
	for (int ufi = 0; ufi < F.rows(); ++ufi) {
		for (int i = 0; i < 3; ++i)
			ref(F(ufi, i)) = 1;
	}
}

DisplaceInfo SubdivisionTri::displace(const BVHTree& bvh, const BVHTree::IntersectionFilter& bvh_test,
	const BVHTree& bvh_border, bool interpolate)
{
	DisplaceInfo di;

	BarycentricGrid grid(subdivision_level());

	// mark border micro-vertices, these are intersected against the border ``wall`` mesh
	Vector3 bVB = border_v.cast<Scalar>();

	VM.setZero();
	VD.setZero();

	Vector3 base_face_normal = (base_V.row(1) - base_V.row(0)).cross(base_V.row(2) - base_V.row(0)).normalized();

	// wrong, all should have positive dot product with the face normal
	//bool coherent_displacement_orientation = (base_VD.row(0).dot(base_VD.row(1)) > 0) && (base_VD.row(0).dot(base_VD.row(2)) > 0) && (base_VD.row(1).dot(base_VD.row(2)) > 0);
	//bool coherent_displacement_orientation = (base_VD.row(0).dot(base_face_normal) > 0) && (base_VD.row(1).dot(base_face_normal) > 0) && (base_VD.row(2).dot(base_face_normal) > 0);
	bool coherent_displacement_orientation = true;

	if (coherent_displacement_orientation) {
		for (int uvi = 0; uvi < grid.num_samples(); ++uvi) {
			if (ref(uvi)) {
				Vector3 w = grid.barycentric_coord(uvi);

				VD.row(uvi) = w.transpose() * base_VD;

				VD.row(uvi).normalize();

				Scalar border_value = w.transpose() * bVB;
				Scalar ray_clipping_threshold = w.transpose() * ray_threshold;

				// Intersect the BVH with the interpolated ray
				// Find the nearest hit ***in absolute value*** (i.e. the ray can go backward)
				// If the ray misses, set displacement to 0
				IntersectionInfo isect;

				bool border_hit = false;
				bool hit = false;
				if (border_value > 0.9) {
					// if we are very close to the border, we first test against the border mesh
					hit = bvh_border.ray_intersection(V.row(uvi), VD.row(uvi), &isect) && std::abs(isect.t) < ray_clipping_threshold;
					border_hit = hit;
				}

				if (!hit) { // no hit or 'reasonably' far from border
					hit = bvh.ray_intersection(V.row(uvi), VD.row(uvi), &isect, bvh_test) && std::abs(isect.t) < ray_clipping_threshold;
					if (!hit) {
						// the ray either missed, or the hit was too far away
						// if the micro-vertex is still close to the border, try again against the border mesh
						if (border_value > 0.5) {
							hit = bvh_border.ray_intersection(V.row(uvi), VD.row(uvi), &isect) && std::abs(isect.t) < ray_clipping_threshold;
							border_hit = hit;
						}
						else {
							hit = false; // else the ray missed (possibly intersecting the surface, but too far)
						}
					}
				}

				if (hit) {
					// scale displacement vector
					VD.row(uvi) *= isect.t;

					di.n_rays++;

					if (isect.t < 0)
						di.n_backward++;

					const MatrixX* VNptr = border_hit ? bvh_border.VN : bvh.VN;
					const MatrixXi* Fptr = border_hit ? bvh_border.F : bvh.F;
					// sample normal from hi-res mesh
					Vector3 n0 = VNptr->row((*Fptr)(isect.fi, 0));
					Vector3 n1 = VNptr->row((*Fptr)(isect.fi, 1));
					Vector3 n2 = VNptr->row((*Fptr)(isect.fi, 2));
					VN.row(uvi) = (isect.b[0] * n0 + isect.b[1] * n1 + isect.b[2] * n2).normalized();
				}
				else {
					VM(uvi) = 1; // mark the vertex if the ray missed
				}
			}
			else { // microvertex is unreferenced, treat it as a missed ray
				VM(uvi) = 1;
			}

			if (VM(uvi)) {
				VM(uvi) = ref(uvi); // preserve mark only if referenced

				// set the displacement vector to 0
				VD.row(uvi).setZero();
				VN.row(uvi).setZero();

				if (ref(uvi))
					di.n_miss++;
			}
		}
	}

	if (interpolate) {
		while (VM.sum() > 0) {
			VectorXu8 VM_pre = VM;
			_interpolate_culled_displacements();
			if (VM == VM_pre) // bail out if interpolation left VM unchanged ***TODO FIXME this should not happen***
				break;
		}
	}

	return di;
}

void SubdivisionTri::displace(const std::vector<Scalar>& micro_displacements)
{
	BarycentricGrid grid(subdivision_level());

	if (micro_displacements.size() == 0 || micro_displacements.size() == VD.rows()) {
		for (int uvi = 0; uvi < VD.rows(); ++uvi) {
			Vector3 w = grid.barycentric_coord(uvi);
			VD.row(uvi) = micro_displacements[uvi] * (w(0) * base_VD.row(0) + w(1) * base_VD.row(1) + w(2) * base_VD.row(2));
		}
	}
}

// Warning! Can create cracks!
void SubdivisionTri::_interpolate_culled_displacements()
{
	BarycentricGrid grid(subdivision_level());

	// handle the case in which all displacements have been culled (no interpolation, just clear everything)
	if (VM.all()) {
		VM.setZero();
		VD.setZero();
		VN.rowwise() = compute_area_vector(base_V.row(0), base_V.row(1), base_V.row(2)).normalized().transpose();
		return;
	}

	// grid offsets for the 6 grid neighbors of a vertex (note that some may fall outside the grid)
	std::vector<Vector2i> offsets = {
		Vector2i(-1, -1),
		Vector2i(-1,  0),
		Vector2i( 0, -1),
		Vector2i( 0, +1),
		Vector2i(+1,  0),
		Vector2i(+1, +1)
	};

	std::vector<int> interpolated; // track interpolated micro-vertices

	// [!] Note on the scalar displacement interpolation
	// We can't average the unnormalized scalar displacements directly, as their magnitude depends on the length of the interpolated directions
	// i.e. if two neighbor vertices have interpolated directions where one's norm is 10x the other, its microdisplacement magnitude
	// will be 1/10th of the other and averaging them does not make sense.
	// We cannot treat the unnormalized microdisplacements as a scalar field, we have to interpolate in object space
	// to ensure the actual displacement distances remain consistent
	for (int uvi = 0; uvi < grid.num_samples(); ++uvi) {
		if (ref(uvi) && VM(uvi) > 0) {
			Vector2i u;
			grid.inverted_index(uvi, &u.y(), &u.x());
			Scalar d_sum = 0;
			Vector3 n_sum = Vector3::Zero();
			int count = 0;
			for (const Vector2i& o : offsets) {
				Vector2i uj = u + o;
				if (grid.valid_index(uj.y(), uj.x())) {
					int uvj = grid.index(uj.y(), uj.x());
					if (ref(uvj) && VM(uvj) == 0) {
						d_sum += VD.row(uvj).norm(); // take the norm here (see comment above)
						n_sum += VN.row(uvj);
						count++;
					}
				}
			}
			if (count > 0) {
				Vector3 w = grid.barycentric_coord(uvi);
				Vector3 dir = interpolate_direction(w).normalized(); // normalize here (see comment above)
				VD.row(uvi) = (d_sum / Scalar(count)) * dir;
				VN.row(uvi) = (n_sum / Scalar(count)).normalized();
				interpolated.push_back(uvi);
			}
		}
	}

	// clear mask for interpolated micro-vertices
	for (int uvi : interpolated) {
		VM(uvi) = 0;
	}
}

// Warning! Can create T-junctions!
bool SubdivisionTri::_upsample(const BVHTree& bvh, Scalar ray_clipping_threshold, const std::function<bool(IntersectionInfo)>& bvh_test)
{
	if (subdivision_level() > 8)
		return false;

	Assert(V.rows() > 0);
	Assert(F.rows() > 0);
	
	bool has_vq = VQ.size() > 0;

	int new_level = subdivision_level() + 1;

	BarycentricGrid lo_grid(subdivision_level());
	BarycentricGrid hi_grid(new_level);

	subdivision_bits = new_level << 3;

	// base subdivision
	subdivide_tri(base_V.row(0), base_V.row(1), base_V.row(2), V, F, subdivision_bits);

	MatrixX VNup = MatrixX::Constant(V.rows(), V.cols(), 0);
	MatrixX VDup = MatrixX::Constant(V.rows(), V.cols(), 0);
	MatrixX VQup = VectorX::Constant(V.rows(), 0);

	std::vector<int> indices;
	indices.reserve(V.rows());
	// propagate VD and VN
	for (int i = 0; i < hi_grid.samples_per_side(); ++i) {
		for (int j = 0; j <= i; ++j) {
			int hi_vi = hi_grid.index(i, j);
			if (i % 2 == 0 && j % 2 == 0) {
				int lo_vi = lo_grid.index(i / 2, j / 2);
				VNup.row(hi_vi) = VN.row(lo_vi);
				VDup.row(hi_vi) = VD.row(lo_vi);
				if (has_vq)
					VQup(hi_vi) = VQ(lo_vi);
			}
			else {
				indices.push_back(hi_vi);
			}
		}
	}

	VN = VNup;
	VD = VDup;
	if (has_vq)
		VQ = VQup;

	// ray-cast missing samples
	_ray_cast(indices, bvh, ray_clipping_threshold, bvh_test);

	return true;
}

void SubdivisionTri::_ray_cast(const std::vector<int>& indices, const BVHTree& bvh, Scalar ray_clipping_threshold, const std::function<bool(IntersectionInfo)>& bvh_test)
{
	BarycentricGrid bary_grid(subdivision_level());

	Vector3 nf = (base_V.row(1) - base_V.row(0)).cross(base_V.row(2) - base_V.row(0)).normalized();

	auto displace_vertex = [&](int vi) -> void {
		int i, j;
		bary_grid.inverted_index(vi, &i, &j);
		Vector3 w = bary_grid.barycentric_coord(i, j);
		VD.row(vi) = w[0] * base_VD.row(0) + w[1] * base_VD.row(1) + w[2] * base_VD.row(2);
		VD.row(vi).normalize();

		// Intersect the BVH with the interpolated ray
		// Find the nearest hit ***in absolute value*** (i.e. the ray can go backward)
		// If the ray misses, set displacement to 0
		IntersectionInfo isect;
		bool miss = true;
		if (bvh.ray_intersection(V.row(vi), VD.row(vi), &isect, bvh_test)) {
			// scale displacement vector
			VD.row(vi) *= isect.t;

			// if intersection is close enough displacement won't be truncated
			if (VD.row(vi).norm() <= ray_clipping_threshold) {
				miss = false;
				// sample normal from hi-res mesh
				Vector3 n0 = (*bvh.VN).row((*bvh.F)(isect.fi, 0));
				Vector3 n1 = (*bvh.VN).row((*bvh.F)(isect.fi, 1));
				Vector3 n2 = (*bvh.VN).row((*bvh.F)(isect.fi, 2));
				VN.row(vi) = (isect.b[0] * n0 + isect.b[1] * n1 + isect.b[2] * n2).normalized();
			}
		}

		if (miss) {
			// set the displacement vector to 0
			VD.row(vi).setZero();
			VN.row(vi) = nf;
		}
	};

	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), displace_vertex);
}

void SubdivisionTri::coarsen_to_level(uint8_t bits)
{
	int new_level = bits >> 3;
	Assert(new_level < subdivision_level());

	bool has_vn = VN.rows() > 0;
	bool has_vd = VD.rows() > 0;
	bool has_vq = VQ.rows() > 0;

	BarycentricGrid old_grid(subdivision_level());
	BarycentricGrid bary_grid(new_level);

	int nskip = subdivision_level() - new_level;
	subdivision_bits = bits;

	subdivide_tri(base_V.row(0), base_V.row(1), base_V.row(2), V, F, subdivision_bits);

	int lod = 1 << nskip;

	for (int i = 0; i < bary_grid.samples_per_side(); ++i) {
		for (int j = 0; j <= i; ++j) {
			int vi = bary_grid.index(i, j);
			int vi_old = old_grid.index(i * lod, j * lod);
			if (has_vn)
				VN.row(vi) = VN.row(vi_old);
			if (has_vd)
				VD.row(vi) = VD.row(vi_old);
			if (has_vq)
				VQ.row(vi) = VQ.row(vi_old);
		}
	}

	VN.conservativeResizeLike(V);
	VD.conservativeResizeLike(V);

	if (has_vq)
		VQ.conservativeResize(V.rows());

	if (!has_vd)
		VD.setZero();

	if (!has_vn)
		VN.rowwise() = compute_area_vector(base_V.row(0), base_V.row(1), base_V.row(2)).normalized().transpose();
}

RunningStats SubdivisionTri::downsampling_error() const
{
	RunningStats rs;

	if (subdivision_level() < 1) {
		rs.add(std::numeric_limits<Scalar>::max());
		return rs;
	}

	BarycentricGrid grid(subdivision_level());
	for (int i = 0; i < grid.samples_per_side(); ++i) {
		for (int j = 0; j <= i; ++j) {
			int v_curr = grid.index(i, j);
			int v_prev = -1;
			int v_next = -1;
			if ((i % 2) == 0) {
				if (j % 2 == 1) {
					v_prev = grid.index(i, j - 1);
					v_next = grid.index(i, j + 1);
				}
			}
			else {
				if ((j % 2) == 0) {
					v_prev = grid.index(i - 1, j);
					v_next = grid.index(i + 1, j);
				}
				else {
					v_prev = grid.index(i - 1, j - 1);
					v_next = grid.index(i + 1, j + 1);
				}
			}

			if (v_prev != -1 && v_next != -1) {
				Vector3 v = V.row(v_curr) + VD.row(v_curr);
				Vector3 u = 0.5 * (V.row(v_prev) + VD.row(v_prev) + V.row(v_next) + VD.row(v_next));
				Scalar err = (v - u).norm();
				rs.add(err);
			}
		}
	}

	return rs;
}

Scalar SubdivisionTri::max_downsampling_error() const
{
	return downsampling_error().max();
}

Scalar SubdivisionTri::avg_downsampling_error() const
{
	return downsampling_error().avg();
}

void SubdivisionTri::extract_boundary_polyline(MatrixX& VP, std::vector<int>& L) const
{
	int d[3] = {};

	d[0] = bool(subdivision_bits & SubdivisionDecimationFlag::Edge0);
	d[1] = bool(subdivision_bits & SubdivisionDecimationFlag::Edge1);
	d[2] = bool(subdivision_bits & SubdivisionDecimationFlag::Edge2);

	BarycentricGrid grid(subdivision_level());

	int ne_side = grid.edges_per_side();

	// total number of vertices of the polyline
	int nv = (ne_side >> d[0]) + (ne_side >> d[1]) + (ne_side >> d[2]);

	VP = MatrixX(nv, 3);
	int vi = 0;

	for (int e = 0; e < 3; ++e) {
		for (int k = 0; k < ne_side; k += (1 + d[e])) {
			int grid_i = e == 1 ? ne_side
				: ((e == 2) ? ne_side - k : k);
			int grid_j = e == 0 ? 0
				: ((e == 2) ? ne_side - k : k);
			int grid_vi = grid.index(grid_i, grid_j);
			VP.row(vi) = V.row(grid_vi) + VD.row(grid_vi);
			L.push_back(vi++);
		}
	}
}

// -- SubdivisionMesh -----------


void SubdivisionMesh::compute_mesh_structure(const MatrixX& base_V, const MatrixXi& base_F, const VectorXu8& subdivision_bits)
{
	_status_bits = 0;

	micro_fn = 0;
	base_fn = 0;

	MatrixXu8 base_FEB = per_face_edge_border_flag(base_V, base_F);
	VectorXu8 base_VB;
	per_vertex_border_flag(base_V, base_F, base_VB);

	faces.clear();
	faces.reserve(base_F.rows());

	// compute the adaptive ray thresholds
	// for each base vertex, it is the longest incident base edge
	VectorX base_ray_threshold = VectorX::Constant(base_V.rows(), 0);
	for (int fi = 0; fi < base_F.rows(); ++fi) {
		for (int i = 0; i < 3; ++i) {
			int vi0 = base_F(fi, i);
			int vi1 = base_F(fi, (i + 1) % 3);
			Scalar l = (base_V.row(vi0) - base_V.row(vi1)).norm();
			base_ray_threshold(vi0) = std::max(base_ray_threshold(vi0), l);
			base_ray_threshold(vi1) = std::max(base_ray_threshold(vi1), l);
		}
	}

	// ... but never go beyond the average edge
	Scalar avg_edge = average_edge(base_V, base_F);
	for (Scalar& threshold : base_ray_threshold)
		threshold = std::min(threshold, avg_edge);

	for (int fi = 0; fi < base_F.rows(); ++fi)
		if (base_F(fi, 0) != INVALID_INDEX)
			faces.push_back(SubdivisionTri(fi));

	int deg = (int)base_V.cols();

	auto subdivide = [&](SubdivisionTri& st) -> void {
		int fi = st.base_fi;
		st.subdivide(subdivision_bits(fi), base_V.row(base_F(fi, 0)), base_V.row(base_F(fi, 1)), base_V.row(base_F(fi, 2)));

		st.border_e = base_FEB.row(fi);
		for (int j = 0; j < 3; ++j)
			st.border_v(j) = base_VB(base_F(fi, j));
		
		for (int j = 0; j < 3; ++j)
			st.ray_threshold(j) = base_ray_threshold(base_F(fi, j));
	};

	std::for_each(std::execution::par_unseq, faces.begin(), faces.end(), subdivide);

	for (const SubdivisionTri& st : faces) {
		micro_fn += st.F.rows();
		base_fn++;
	}

	_status_bits |= Subdivided;
}

void SubdivisionMesh::update_mesh_structure(
	const VectorXu8& subdivision_bits,
	const MatrixX& base_V,
	const MatrixXi& base_F,
	const BVHTree& bvh,
	const std::function<bool(IntersectionInfo)>& bvh_test,
	const MatrixX& border_V,
	const MatrixXi& border_F,
	const MatrixX& border_VN)
{
	BVHTree bvh_border;
	bvh_border.build_tree(&border_V, &border_F, &border_VN);

	for (int fi = 0; fi < base_fn; ++fi) {
		SubdivisionTri& face = faces[fi];
		int ufn_old = face.F.rows();
		if (subdivision_bits(face.base_fi) != face.subdivision_bits) {
			// refine
			face.subdivide(subdivision_bits(face.base_fi), face.base_V.row(0), face.base_V.row(1), face.base_V.row(2));
			if (is_displaced()) {
				face.displace(bvh, bvh_test, bvh_border, _interpolate);
			}
		}

		int ufn_new = face.F.rows();
		micro_fn += (ufn_new - ufn_old);
	}

	stitch_primitives(base_V, base_F);
}

void SubdivisionMesh::stitch_primitives(const MatrixX& base_V, const MatrixXi& base_F)
{
	// stitch vertices
	MatrixX VD = MatrixX::Constant(base_V.rows(), 3, 0);
	VectorX W = VectorX::Constant(base_V.rows(), 0);

	// accumulate
	for (SubdivisionTri& st : faces) {
		BarycentricGrid grid(st.subdivision_level());
		int n = grid.samples_per_side();

		int v0 = base_F(st.base_fi, 0);
		int v1 = base_F(st.base_fi, 1);
		int v2 = base_F(st.base_fi, 2);

		VD.row(v0) += st.VD.row(grid.index(0, 0));
		W(v0) += 1;

		VD.row(v1) += st.VD.row(grid.index(n - 1, 0));
		W(v1) += 1;

		VD.row(v2) += st.VD.row(grid.index(n - 1, n - 1));
		W(v2) += 1;
	}

	// average
	for (SubdivisionTri& st : faces) {
		BarycentricGrid grid(st.subdivision_level());
		int n = grid.samples_per_side();

		int v0 = base_F(st.base_fi, 0);
		int v1 = base_F(st.base_fi, 1);
		int v2 = base_F(st.base_fi, 2);

		st.VD.row(grid.index(0, 0)) = VD.row(v0) / W(v0);
		st.VD.row(grid.index(n - 1, 0)) = VD.row(v1) / W(v1);
		st.VD.row(grid.index(n - 1, n - 1)) = VD.row(v2) / W(v2);
	}

	// stitch edges
	EFAdjacency EF = compute_adjacency_edge_face(base_F);

	for (auto& entry : EF) {
		uint8_t sl = 0xFF;
		for (EFEntry ef : entry.second)
			sl = std::min(sl, faces[ef.fi()].subdivision_level());

		std::vector<Vector3> EVD((1 << sl) + 1, Vector3::Zero());
		std::vector<Vector3> EVN((1 << sl) + 1, Vector3::Zero());

		// pass 1 - accumulate
		int efi = 0; // track parity (if 2 edges are present, scanning micro-displacements along the second edge must use the reverse order wrt the first edge)
		// ***note*** this assumes manifoldness and coherent orientation of the base faces
		for (EFEntry ef : entry.second) {
			const SubdivisionTri& st = faces[ef.fi()];
			BarycentricGrid grid(st.subdivision_level());
			int n = grid.samples_per_side();
			int step = (st.subdivision_level() == sl) ? 1 : 2;
			for (int k = 1 * step; k < n - 1; k += step) {
				int i = k / step;
				switch (ef.ei()) {
				case 0:
					if (efi % 2) {
						EVD[i] += st.VD.row(grid.index(k, 0));
						EVN[i] += st.VN.row(grid.index(k, 0));
					}
					else {
						EVD[i] += st.VD.row(grid.index(n - 1 - k, 0));
						EVN[i] += st.VN.row(grid.index(n - 1 - k, 0));
					}
					break;
				case 1:
					if (efi % 2) {
						EVD[i] += st.VD.row(grid.index(n - 1, k));
						EVN[i] += st.VN.row(grid.index(n - 1, k));
					}
					else {
						EVD[i] += st.VD.row(grid.index(n - 1, n - 1 - k));
						EVN[i] += st.VN.row(grid.index(n - 1, n - 1 - k));
					}
					break;
				case 2:
					if (efi % 2) {
						EVD[i] += st.VD.row(grid.index(n - 1 - k, n - 1 - k));
						EVN[i] += st.VN.row(grid.index(n - 1 - k, n - 1 - k));
					}
					else {
						EVD[i] += st.VD.row(grid.index(k, k));
						EVN[i] += st.VN.row(grid.index(k, k));
					}
					break;
				default:
					Assert(0);
				}
			}
			efi++;
		}

		// pass 2 - average
		efi = 0;
		for (EFEntry ef : entry.second) {
			SubdivisionTri& st = faces[ef.fi()];
			BarycentricGrid grid(st.subdivision_level());
			int n = grid.samples_per_side();
			int step = (st.subdivision_level() == sl) ? 1 : 2;

			Scalar s = Scalar(1) / entry.second.size();

			for (int k = 1 * step; k < n - 1; k += step) {
				int i = k / step;
				switch (ef.ei()) {
				case 0:
					if (efi % 2) {
						st.VD.row(grid.index(k, 0)) = s * EVD[i];
						st.VN.row(grid.index(k, 0)) = s * EVN[i];
					}
					else {
						st.VD.row(grid.index(n - 1 - k, 0)) = s * EVD[i];
						st.VN.row(grid.index(n - 1 - k, 0)) = s * EVN[i];
					}
					break;
				case 1:
					if (efi % 2) {
						st.VD.row(grid.index(n - 1, k)) = s * EVD[i];
						st.VN.row(grid.index(n - 1, k)) = s * EVN[i];
					}
					else {
						st.VD.row(grid.index(n - 1, n - 1 - k)) = s * EVD[i];
						st.VN.row(grid.index(n - 1, n - 1 - k)) = s * EVN[i];
					}
					break;
				case 2:
					if (efi % 2) {
						st.VD.row(grid.index(n - 1 - k, n - 1 - k)) = s * EVD[i];
						st.VN.row(grid.index(n - 1 - k, n - 1 - k)) = s * EVN[i];
					}
					else {
						st.VD.row(grid.index(k, k)) = s * EVD[i];
						st.VN.row(grid.index(k, k)) = s * EVN[i];
					}
					break;
				default:
					Assert(0);
				}
			}
			efi++;
		}
	}
}

void SubdivisionMesh::clear_micro_displacements()
{
	for (SubdivisionTri& st : faces) {
		st.VD.setZero();
	}

	_status_bits &= ~Displaced;
}

void SubdivisionMesh::compute_mesh_structure(const MatrixX& base_V, const MatrixXi& base_F, uint8_t subdivision_level)
{
	VectorXu8 subdivision_bits = VectorXu8::Constant(base_F.rows(), subdivision_level);
	for (int i = 0; i < subdivision_bits.rows(); ++i)
		subdivision_bits(i) = subdivision_bits(i) << 3;
	compute_mesh_structure(base_V, base_F, subdivision_bits);
}

void SubdivisionMesh::compute_micro_displacements(
	const MatrixX& base_V,
	const MatrixX& base_VD,
	const MatrixXi& base_F,
	const BVHTree& bvh,
	const std::function<bool(IntersectionInfo)>& bvh_test,
	const MatrixX& border_V,
	const MatrixXi& border_F,
	const MatrixX& border_VN)
{
	compute_micro_displacements(bvh, bvh_test, base_V, base_VD, base_F, VectorXi::Constant(base_F.rows(), 1), border_V, border_F, border_VN);
}

void SubdivisionMesh::compute_micro_displacements(const MatrixX& base_VD, const MatrixXi& base_F, const std::vector<std::vector<Scalar>>& micro_displacements)
{
	Assert(is_subdivided());

	auto displace = [&](SubdivisionTri& st) -> void {
		int fi = st.base_fi;
		for (int j = 0; j < 3; ++j) {
			st.base_VD.row(j) = base_VD.row(base_F(fi, j));
		}
		Assert(fi < micro_displacements.size());
		st.displace(micro_displacements[fi]);
	};
	
	std::for_each(std::execution::par_unseq, faces.begin(), faces.end(), displace);

	_status_bits |= Displaced;
}

void SubdivisionMesh::compute_micro_displacements(const BVHTree& bvh, const std::function<bool(IntersectionInfo)>& bvh_test,
	const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, const VectorXi& update,
	const MatrixX& border_V, const MatrixXi& border_F, const MatrixX& border_VN)
{
	Assert(is_subdivided());
	Timer t;

	int deg = base_V.cols();

	BVHTree bvh_border;
	bvh_border.build_tree(&border_V, &border_F, &border_VN, 16);

	std::vector<DisplaceInfo> displace_infos(base_F.rows());

	auto displace = [&](SubdivisionTri& st) -> void {
		int fi = st.base_fi;
		if (update(fi)) {
			for (int j = 0; j < deg; ++j)
				st.base_VD.row(j) = base_VD.row(base_F(fi, j));

			displace_infos[fi] = st.displace(bvh, bvh_test, bvh_border, _interpolate);
		}
	};

	std::for_each(std::execution::par_unseq, faces.begin(), faces.end(), displace);

	long n_rays = 0;
	long n_miss = 0;
	long n_backward = 0;
	long n_fallback = 0;

	for (const DisplaceInfo& di : displace_infos) {
		n_rays += di.n_rays;
		n_miss += di.n_miss;
		n_backward += di.n_backward;
		n_fallback += di.n_fallback;
	}

	_status_bits |= Displaced;

	stitch_primitives(base_V, base_F);

	std::cout << "Micro displacement computation took " << t.time_elapsed() << " seconds" << std::endl;
	std::cout << "Average time per base face: " << t.time_elapsed() / micro_fn << " seconds" << std::endl;
	std::cout << "Number of ray-queries: " << n_rays << std::endl;
	std::cout << "Backward hits: " << n_backward << std::endl;
	std::cout << "Misses: " << n_miss << std::endl;
	std::cout << "Fallback queries: " << n_fallback << std::endl;
}

void SubdivisionMesh::compute_micro_displacements(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN, const VectorXi& update,
	const MatrixX& border_V, const MatrixXi& border_F, const MatrixX& border_VN)
{
	Assert(is_subdivided());
	Timer t;

	int deg = base_V.cols();

	BVHTree bvh;
	bvh.build_tree(&hi_V, &hi_F, &hi_VN);

	MatrixX hi_FN = compute_face_normals(hi_V, hi_F);
	auto bvh_test = [&](const IntersectionInfo& ii) -> bool {
		return ii.d.dot(hi_FN.row(ii.fi)) >= 0;
	};

	compute_micro_displacements(bvh, bvh_test, base_V, base_VD, base_F, update, border_V, border_F, border_VN);
}

void SubdivisionMesh::normalize_displacements(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, MatrixX& new_base_V, MatrixX& new_base_VD)
{
	Assert(is_displaced());

	// start from the global max and min displacement to find the initial scale factors K_top and K_bottom
	Scalar global_d_max = std::numeric_limits<Scalar>::lowest();
	Scalar global_d_min = std::numeric_limits<Scalar>::max();

	for (const SubdivisionTri& st : faces) {
		for (int vi = 0; vi < st.V.rows(); ++vi) {
			Scalar d = st.scalar_displacement(vi);
			global_d_max = std::max(global_d_max, d);
			global_d_min = std::min(global_d_min, d);
		}
	}

	new_base_V = base_V + global_d_min * base_VD;
	new_base_VD = (global_d_max - global_d_min) * base_VD;
}

void SubdivisionMesh::minimize_base_directions_length(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, MatrixX& new_base_V, MatrixX& new_base_VD, MatrixX& top_VD, MatrixX& bottom_VD)
{
	VectorX max_displacement = VectorX::Constant(base_VD.rows(), 0);
	VectorX min_displacement = VectorX::Constant(base_VD.rows(), 0);

	for (int i = 0; i < (int)base_F.rows(); ++i) {
		if (base_F(i, 0) != INVALID_INDEX) {
			for (int j = 0; j < 3; ++j) {
				max_displacement(base_F(i, j)) = std::numeric_limits<Scalar>::lowest();
				min_displacement(base_F(i, j)) = std::numeric_limits<Scalar>::max();
			}
		}
	}

	for (SubdivisionTri& st : faces) {
		BarycentricGrid bary_grid(st.subdivision_level());
		Assert(bary_grid.num_samples() == (int)st.V.rows());
		for (int uvi = 0; uvi < bary_grid.num_samples(); ++uvi) {
			if (st.ref(uvi)) {
				Vector3 w = bary_grid.barycentric_coord(uvi);
				Scalar displacement = st.scalar_displacement(uvi);
				Assert(std::isfinite(displacement));
				Assert(!std::isnan(displacement));
				for (int k = 0; k < 3; ++k) {
					if (w[k] > 0) {
						int base_vk = base_F(st.base_fi, k);
						max_displacement(base_vk) = std::max(max_displacement(base_vk), displacement);
						min_displacement(base_vk) = std::min(min_displacement(base_vk), displacement);
					}
				}
			}
		}
	}

	//Box3 box;
	//for (int i = 0; i < base_V.rows(); ++i)
	//	box.add(base_V.row(i));
	//Scalar delta = box.diagonal().norm() * 1e-8;

	//for (int i = 0; i < base_VD.rows(); ++i) {
	//	Assert(max_displacement(i) >= min_displacement(i));
	//	Assert(max_displacement(i) != std::numeric_limits<Scalar>::lowest());
	//	Assert(min_displacement(i) != std::numeric_limits<Scalar>::max());
	//	if (max_displacement(i) - min_displacement(i) < delta)
	//		max_displacement(i) += delta - min_displacement(i);
	//}

	MatrixX Vp = base_V + min_displacement.asDiagonal() * base_VD;
	MatrixX VDp = (max_displacement - min_displacement).asDiagonal() * base_VD;

	//write_obj("min_base_1.obj", Vp, base_F);
	//write_obj("min_top_1.obj", Vp + VDp, base_F);

	new_base_V = Vp;
	new_base_VD = VDp;

	top_VD = Vp + VDp;
	bottom_VD = Vp;
}

void SubdivisionMesh::minimize_base_directions_length_with_shrinking(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F, MatrixX& new_base_V, MatrixX& new_base_VD, MatrixX& top_VD, MatrixX& bottom_VD)
{
	enum Membrane {Top = 0, Bottom = 1};
	
	struct ViolatedConstraintInfo {
		int base_fi;
		int vi;
		Scalar H;
		Membrane m;
	};

	// start from the global max and min displacement to find the initial scale factors K_top and K_bottom
	Scalar global_d_max = std::numeric_limits<Scalar>::lowest();
	Scalar global_d_min = std::numeric_limits<Scalar>::max();

	for (const SubdivisionTri& st : faces) {
		for (int vi = 0; vi < st.V.rows(); ++vi) {
			Scalar d = st.scalar_displacement(vi);
			global_d_max = std::max(global_d_max, d);
			global_d_min = std::min(global_d_min, d);
		}
	}

	{
		MatrixX Vp = base_V + global_d_min * base_VD;
		MatrixX VDp = (global_d_max - global_d_min) * base_VD;

		//write_obj("base.obj", Vp, base_F);
		//write_obj("top.obj", Vp + VDp, base_F);
	}

	// compute top and bottom displacement volume offsets (relative to the base mesh positions)
	//VectorX K_top = global_d_max * (base_VD.rowwise().norm());
	//VectorX K_bottom = global_d_min * (base_VD.rowwise().norm());

	VectorX K_top = VectorX::Constant(base_V.rows(), global_d_max);
	VectorX K_bottom = VectorX::Constant(base_V.rows(), global_d_min);

	// 0. compute height field
	std::vector<VectorX> H_in;
	H_in.resize(base_F.rows());

	for (const SubdivisionTri& st : faces) {
		H_in[st.base_fi].resize(st.V.rows());
		for (int vi = 0; vi < st.V.rows(); ++vi)
			H_in[st.base_fi](vi) = st.scalar_displacement(vi);
	}

	// TODO FIXME what if the normal ends up pointing inwards (i.e. the force pushes outwards and inflates instead of shrinking)?

	// 1.1 compute the force (pull) vectors (direction = -n, magnitude proportional to area, accumulated)
	VectorX pull;
	{
		MatrixX P = MatrixX::Constant(base_V.rows(), base_V.cols(), 0); // top forces
		for (const SubdivisionTri& st : faces) {
			Vector3 p = -(1 / Scalar(3)) * compute_area_vector(base_V.row(base_F(st.base_fi, 0)), base_V.row(base_F(st.base_fi, 1)), base_V.row(base_F(st.base_fi, 2)));
			for (int i = 0; i < 3; ++i) {
				Vector3 di = base_VD.row(base_F(st.base_fi, i));
				Vector3 pi = clamp(p.dot(di), Scalar(-1), Scalar(0)) * di.normalized();
				P.row(base_F(st.base_fi, i)) += pi;
			}
		}
		
		pull = P.rowwise().norm();
	}

	// scale the pull so that it is proportional to the displacements
	Scalar pull_scale = std::max(std::abs(global_d_max), std::abs(global_d_min)) / pull.maxCoeff();
	Assert(pull_scale > 0);
	pull *= pull_scale;

	std::cout << "MAX PULL " << pull.maxCoeff() << " " << pull.minCoeff() << " " << pull(35) << std::endl;

	// 1.3. detect unconstrained vertices
	std::vector<ViolatedConstraintInfo> violated;
	
	for (const SubdivisionTri& st : faces) {
		BarycentricGrid grid(st.subdivision_level());
		int base_v0 = base_F(st.base_fi, 0);
		int base_v1 = base_F(st.base_fi, 1);
		int base_v2 = base_F(st.base_fi, 2);

		for (int vi = 0; vi < st.V.rows(); ++vi) if (st.ref(vi)) {
			Vector3 w = grid.barycentric_coord(vi);

			Scalar h_in = H_in[st.base_fi](vi);
			Scalar h_top = w(0) * K_top(base_v0) + w(1) * K_top(base_v1) + w(2) * K_top(base_v2);
			Scalar h_bottom = w(0) * K_bottom(base_v0) + w(1) * K_bottom(base_v1) + w(2) * K_bottom(base_v2);

			if (h_in > h_top && std::abs(h_in - h_top) > 1e-6) {
				violated.push_back(ViolatedConstraintInfo{ .base_fi = st.base_fi, .vi = vi, .H = h_in - h_top, .m = Top });
			}
	 		else if (h_in < h_bottom && std::abs(h_bottom - h_in) > 1e-6) {
				violated.push_back(ViolatedConstraintInfo{ .base_fi = st.base_fi, .vi = vi, .H = h_in - h_bottom, .m = Bottom });
			}
		}
	}

	Assert(violated.size() == 0);

	// 1.4. enforce constraints
	//std::sort(violated.begin(), violated.end(), [](const ViolatedConstraintInfo& vci1, const ViolatedConstraintInfo& vci2) -> bool { return vci1.H > vci2.H; });
	//while (!violated.empty()) {
	//	ViolatedConstraintInfo vci = violated.back();
	//	violated.pop_back();

	//	const SubdivisionTri& st = faces[vci.base_fi];
	//	BarycentricGrid grid(st.subdivision_level());

	//	int base_v0 = base_F(st.base_fi, 0);
	//	int base_v1 = base_F(st.base_fi, 1);
	//	int base_v2 = base_F(st.base_fi, 2);

	//	Vector3 w = grid.barycentric_coord(vci.vi);
	//	Scalar H_new = -1;

	//	// first check if the constraint is still violated before adjusting the scalar multipliers
	//	{
	//		Scalar h_in = H_in[st.base_fi](vci.vi);
	//		Scalar h_top = w(0) * K_top(base_v0) + w(1) * K_top(base_v1) + w(2) * K_top(base_v2);
	//		Scalar h_bottom = w(0) * K_bottom(base_v0) + w(1) * K_bottom(base_v1) + w(2) * K_bottom(base_v2);

	//		if ((h_in <= h_top && vci.m == Top) || (h_in >= h_bottom && vci.m == Bottom)) {
	//			// no longer violated, skip
	//			continue;
	//		}
	//		else {
	//			if (h_in > h_top && vci.m == Top)
	//				H_new = h_in - h_top;
	//			else if (h_in < h_bottom && vci.m == Bottom)
	//				H_new = h_in - h_bottom;
	//		}

	//		if (H_new == -1)
	//			continue;
	//	}

	//	vci.H = H_new;

	//	Vector3 d_min = w * (vci.H / w.squaredNorm());

	//	for (int i = 0; i < 3; ++i) {
	//		if (vci.m == Top) {
	//			K_top(base_F(st.base_fi, i)) += d_min(i);
	//		}
	//		else {
	//			K_bottom(base_F(st.base_fi, i)) += d_min(i);
	//		}
	//	}
	//}

	// set the 'step' size
	Scalar step = 0.001;

	Scalar H_max = std::numeric_limits<Scalar>::min();

	const int N_ITER = 500;

	// while (not converged)
	for (int i = 0; i < N_ITER; ++i) {

		// ensure no constraints are violated
		{
			std::vector<ViolatedConstraintInfo> violated;
			
			for (const SubdivisionTri& st : faces) {
				BarycentricGrid grid(st.subdivision_level());
				int base_v0 = base_F(st.base_fi, 0);
				int base_v1 = base_F(st.base_fi, 1);
				int base_v2 = base_F(st.base_fi, 2);

				for (int vi = 0; vi < st.V.rows(); ++vi) if (st.ref(vi)) {
					Vector3 w = grid.barycentric_coord(vi);

					Scalar h_in = H_in[st.base_fi](vi);
					Scalar h_top = w(0) * K_top(base_v0) + w(1) * K_top(base_v1) + w(2) * K_top(base_v2);
					Scalar h_bottom = w(0) * K_bottom(base_v0) + w(1) * K_bottom(base_v1) + w(2) * K_bottom(base_v2);

					if (h_in > h_top && std::abs(h_in - h_top) > 1e-6) {
						violated.push_back(ViolatedConstraintInfo{ .base_fi = st.base_fi, .vi = vi, .H = h_in - h_top, .m = Top });
					}
					else if (h_in < h_bottom && std::abs(h_bottom - h_in) > 1e-6) {
						violated.push_back(ViolatedConstraintInfo{ .base_fi = st.base_fi, .vi = vi, .H = h_in - h_bottom, .m = Bottom });
					}
				}
			}

			if (violated.size() > 0) {
				Scalar max_elem = std::max_element(violated.begin(), violated.end(), [](const ViolatedConstraintInfo& vci1, const ViolatedConstraintInfo& vci2) -> bool { return vci1.H > vci2.H; })->H;
				std::cout << "Violated size is " << violated.size() << " viol = " << max_elem << std::endl;
			}
			Assert(violated.size() == 0);
		}

		// pull membranes
		K_top = K_top - step * pull;
		K_bottom = K_bottom + step * pull;

		// detect violated constraints
		std::vector<ViolatedConstraintInfo> violated;
		
		for (const SubdivisionTri& st : faces) {
			BarycentricGrid grid(st.subdivision_level());
			int base_v0 = base_F(st.base_fi, 0);
			int base_v1 = base_F(st.base_fi, 1);
			int base_v2 = base_F(st.base_fi, 2);

			for (int vi = 0; vi < st.V.rows(); ++vi) if (st.ref(vi)) {
				Vector3 w = grid.barycentric_coord(vi);

				Scalar h_in = H_in[st.base_fi](vi);
				Scalar h_top = w(0) * K_top(base_v0) + w(1) * K_top(base_v1) + w(2) * K_top(base_v2);
				Scalar h_bottom = w(0) * K_bottom(base_v0) + w(1) * K_bottom(base_v1) + w(2) * K_bottom(base_v2);

				if (h_in > h_top) {
					violated.push_back(ViolatedConstraintInfo{ .base_fi = st.base_fi, .vi = vi, .H = h_in - h_top, .m = Top });
				}
				else if (h_in < h_bottom) {
					violated.push_back(ViolatedConstraintInfo{ .base_fi = st.base_fi, .vi = vi, .H = h_in - h_bottom, .m = Bottom });
				}
			}
		}

		// enforce constraints
		std::sort(violated.begin(), violated.end(), [](const ViolatedConstraintInfo& vci1, const ViolatedConstraintInfo& vci2) -> bool { return vci1.H > vci2.H; });
		while (!violated.empty()) {
			ViolatedConstraintInfo vci = violated.back();
			violated.pop_back();

			const SubdivisionTri& st = faces[vci.base_fi];
			BarycentricGrid grid(st.subdivision_level());

			int base_v0 = base_F(st.base_fi, 0);
			int base_v1 = base_F(st.base_fi, 1);
			int base_v2 = base_F(st.base_fi, 2);

			Vector3 w = grid.barycentric_coord(vci.vi);

			Scalar H_new = -1;

			// first check if the constraint is still violated before adjusting the scalar multipliers
			{
				Scalar h_in = H_in[st.base_fi](vci.vi);
				Scalar h_top = w(0) * K_top(base_v0) + w(1) * K_top(base_v1) + w(2) * K_top(base_v2);
				Scalar h_bottom = w(0) * K_bottom(base_v0) + w(1) * K_bottom(base_v1) + w(2) * K_bottom(base_v2);

				if ((h_in <= h_top && vci.m == Top) || (h_in >= h_bottom && vci.m == Bottom)) {
					// no longer violated, skip
					continue;
				}
				else {
					if (h_in > h_top && vci.m == Top)
						H_new = h_in - h_top;
					else if (h_in < h_bottom && vci.m == Bottom)
						H_new = h_in - h_bottom;
				}

				if (H_new == -1)
					continue;
			}

			vci.H = H_new;

			H_max = std::max(H_max, std::abs(H_new));

			Vector3 d_min = w * (vci.H / w.squaredNorm());

			for (int i = 0; i < 3; ++i) {
				if (vci.m == Top) {
					K_top(base_F(st.base_fi, i)) += d_min(i);
				}
				else {
					K_bottom(base_F(st.base_fi, i)) += d_min(i);
				}
			}
		}

	}

	std::cout << " H_MAX = " << H_max << std::endl;

	{
		MatrixX Vp = base_V + K_bottom.asDiagonal() * base_VD;
		MatrixX VDp = (K_top - K_bottom).asDiagonal() * base_VD;

		Assert(Vp.rows() == VDp.rows());

		//write_obj("min_base_2.obj", Vp, base_F);
		//write_obj("min_top_2.obj", Vp + VDp, base_F);
		//write_obj("micro.obj", *this, 1.0);

		new_base_V = Vp;
		new_base_VD = VDp;

		top_VD = Vp + VDp;
		bottom_VD = Vp;
	}

}

void SubdivisionMesh::compute_vertex_quality_displacement_distance()
{
	auto f = [&](SubdivisionTri& st) -> void {
		st.VQ = -st.VD.rowwise().norm();
	};

	std::for_each(std::execution::par_unseq, faces.begin(), faces.end(), f);

	_status_bits |= SubdivisionMesh::VertexQuality;
}

void SubdivisionMesh::compute_face_quality_aspect_ratio()
{
	auto f = [&](SubdivisionTri& st) -> void {
		::compute_face_quality_aspect_ratio(st.V + st.VD, st.F, st.FQ);
	};

	std::for_each(std::execution::par_unseq, faces.begin(), faces.end(), f);

	_status_bits |= SubdivisionMesh::FaceQuality;
}

void SubdivisionMesh::compute_face_quality_stretch()
{
	auto f = [&](SubdivisionTri& st) -> void {
		::compute_face_quality_stretch(st.V, st.V + st.VD, st.F, st.FQ);
	};

	std::for_each(std::execution::par_unseq, faces.begin(), faces.end(), f);

	_status_bits |= SubdivisionMesh::FaceQuality;
}

void SubdivisionMesh::extract_mesh(MatrixX& V, MatrixXi& F) const
{
	int vn = 0;
	int fn = 0;
	for (const SubdivisionTri& st : faces) {
		vn += st.V.rows();
		fn += st.F.rows();
	}

	V = MatrixX::Constant(vn, 3, 0);
	F.resize(fn, 3);

	int vi = 0;
	int fi = 0;
	for (const SubdivisionTri& st : faces) {
		int vn_st = st.V.rows();
		int fn_st = st.F.rows();
		V.block(vi, 0, vn_st, 3) = st.V + st.VD;
		F.block(fi, 0, fn_st, 3) = st.F + MatrixXi::Constant(st.F.rows(), st.F.cols(), vi);
		vi += vn_st;
		fi += fn_st;
	}
}

void SubdivisionMesh::extract_mesh_with_uvs(const MatrixX& base_UV, const MatrixX& base_VN, const MatrixX& base_VT, const MatrixXi& base_F, MatrixX& V, MatrixXi& F, MatrixX& UV, MatrixX& VN, MatrixX& VT) const
{
	extract_mesh(V, F);

	UV.resize(V.rows(), 2);
	VN.resize(V.rows(), 3);
	VT.resize(V.rows(), 4);

	auto uv_iterator = UV.rowwise().begin();
	auto vn_iterator = VN.rowwise().begin();
	auto tg_iterator = VT.rowwise().begin();

	for (const SubdivisionTri& st : faces) {
		BarycentricGrid grid(st.subdivision_level());

		Vector2 u0 = base_UV.row(base_F(st.base_fi, 0));
		Vector2 u1 = base_UV.row(base_F(st.base_fi, 1));
		Vector2 u2 = base_UV.row(base_F(st.base_fi, 2));

		Vector3 n0 = base_VN.row(base_F(st.base_fi, 0));
		Vector3 n1 = base_VN.row(base_F(st.base_fi, 1));
		Vector3 n2 = base_VN.row(base_F(st.base_fi, 2));

		Vector4 tg0 = base_VT.row(base_F(st.base_fi, 0));
		Vector4 tg1 = base_VT.row(base_F(st.base_fi, 1));
		Vector4 tg2 = base_VT.row(base_F(st.base_fi, 2));

		for (int vi = 0; vi < grid.num_samples(); ++vi) {
			Vector3 w = grid.barycentric_coord(vi);
			*uv_iterator++ = w(0) * u0 + w(1) * u1 + w(2) * u2;
			*vn_iterator++ = (w(0) * n0 + w(1) * n1 + w(2) * n2).normalized();

			Vector4 tg = w(0) * tg0 + w(1) * tg1 + w(2) * tg2;
			Vector3 tgn = Vector3(tg(0), tg(1), tg(2)).normalized();
			*tg_iterator++ = Vector4(tgn(0), tgn(1), tgn(2), std::round(tg(3)));
		}
	}
}

void SubdivisionMesh::_debug_displacements_minmax()
{
	Scalar min_displacement = std::numeric_limits<Scalar>::max();
	Scalar max_displacement = std::numeric_limits<Scalar>::lowest();

	for (SubdivisionTri& st : faces) {
		BarycentricGrid bary_grid(st.subdivision_level());
		for (int i = 0; i < bary_grid.samples_per_side(); ++i) {
			for (int j = 0; j <= i; ++j) {
				int vi = bary_grid.index(i, j);
				//Vector3 w = bary_grid.barycentric_coord(i, j);
				//Vector3 vd = w[0] * st.base_VD.row(0) + w[1] * st.base_VD.row(1) + w[2] * st.base_VD.row(2);
				//Scalar displacement = vd.dot(st.VD.row(vi)) / vd.squaredNorm();
				Scalar displacement = st.scalar_displacement(vi);
				min_displacement = std::min(min_displacement, displacement);
				max_displacement = std::max(max_displacement, displacement);
			}
		}
	}

	std::cout << "Min displacement is " << min_displacement << std::endl;
	std::cout << "Max displacement is " << max_displacement << std::endl;
}

void SubdivisionMesh::_debug_reproject_displacements(const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F)
{
	for (SubdivisionTri& st : faces) {
		Vector3i bv = base_F.row(st.base_fi);

		//for (int i = 0; i < 3; ++i) {
		//	Assert(base_VD.row(bv(i)) == st.base_VD.row(i));
		//	Assert(base_V.row(bv(i)) == st.base_V.row(i));
		//}

		for (int vi = 0; vi < (int)st.V.rows(); ++vi) {
			Scalar displacement = st.scalar_displacement(vi);

			Assert(!std::isnan(displacement));
			Assert(std::isfinite(displacement));
	
			Vector3 w = compute_bary_coords(st.V.row(vi), st.base_V.row(0), st.base_V.row(1), st.base_V.row(2));
			st.VD.row(vi) = displacement * st.interpolate_direction(w);
		}
	}
}

void SubdivisionMesh::_quantize_displacements(int nbits)
{
	if (nbits <= 0 || nbits > 16)
		return;

	int max_q_val = 0;

	Assert(nbits <= 16);
	for (int i = 0; i < nbits; ++i)
		max_q_val |= (1 << i);

	for (SubdivisionTri& st : faces) {
		BarycentricGrid grid(st.subdivision_level());
		for (int vi = 0; vi < (int)st.V.rows(); ++vi) {
			Scalar displacement = st.scalar_displacement(vi);

			int quantized_val = int(std::min(displacement, Scalar(1)) * max_q_val);
			Scalar dequantized_val = quantized_val / Scalar(max_q_val);

			st.VD.row(vi) = dequantized_val * st.interpolate_direction(grid.barycentric_coord(vi));
		}
	}
}

// Other functions

VectorXu8 compute_subdivision_levels_constant(const MatrixXi& F, uint8_t level)
{
	return VectorXu8::Constant(F.rows(), level);
}

VectorXu8 compute_subdivision_levels_uniform_area(const MatrixX& V, const MatrixXi& F, Scalar average_area_inv, Scalar global_subdivision_level)
{
	VectorXu8 subdivision = VectorXu8::Constant(F.rows(), 0);

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			Scalar f_area = face_area(V, F.row(i));
			int sublevel = std::round(global_subdivision_level + std::log2(f_area * average_area_inv) / 2.0);
			if (sublevel < 0)
				sublevel = 0;

			subdivision(i) = (uint8_t) sublevel;
		}
	}

	return subdivision;
}

// Utility class used to track error levels in the adaptive subdivision computation

struct MicrosamplingError {
	SubdivisionTri st;
	std::vector<Scalar> errors;
	uint8_t current_level;

	MicrosamplingError(int base_fi)
		: st(base_fi), errors(), current_level(0)
	{
	}

	Scalar current_error() const
	{
		Assert(current_level >= 0);
		Assert(current_level < (int)errors.size());
		return errors[current_level];
	}

	bool operator<(const MicrosamplingError& other) const {
		return current_error() < other.current_error();
	}
};

// Adaptively increases subdivision levels until further subdivision either surpasses the target_ufn budget or
// displacement of the next subdivision level produces a coarsening error (distance of the new samples to the
// surface at the previous level) lower than the target_max_error threshold
// Initialization precomputes coarsening errors up to a predefined level, then iterations increase the base face subdivisions
// prioritizing the largest next coarsening error (upsampling if going above the cached levels).
VectorXu8 compute_mesh_subdivision_adaptive(
	const MatrixX& base_V, const MatrixX& base_VD, const MatrixXi& base_F,
	const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN, int target_ufn, Scalar target_max_error)
{
	Timer t;
	int deg = (int)base_F.cols();

	std::cout << "TODO FIXME compute_mesh_subdivision_adaptive(): Border intersection" << std::endl;

	// build BVH

	BVHTree bvh;
	bvh.build_tree(&hi_V, &hi_F, &hi_VN);

	Scalar ray_clipping_threshold = 2 * average_edge(base_V, base_F);

	MatrixX hi_FN = compute_face_normals(hi_V, hi_F);
	auto bvh_test = [&](const IntersectionInfo& ii) -> bool {
		return ii.d.dot(hi_FN.row(ii.fi)) >= 0;
	};
	
	// Initialize microsampling error levels

	std::vector<MicrosamplingError> samplers;
	for (int fi = 0; fi < base_F.rows(); ++fi)
		if (base_F(fi, 0) != INVALID_INDEX)
			samplers.push_back(MicrosamplingError(fi));

	// Initialize the MicrosamplingError objects by precomputing errors up to max_init_level

	auto init_mse = [&](MicrosamplingError& mse, const uint8_t max_init_level) -> void {
		mse.current_level = 0;
		mse.errors.clear();

		for (int j = 0; j < deg; ++j) {
			mse.st.base_VD.row(j) = base_VD.row(base_F(mse.st.base_fi, j));
			mse.st.base_V.row(j) = base_V.row(base_F(mse.st.base_fi, j));
		}

		Timer tt;
		mse.st.subdivide(uint8_t(max_init_level << 3), mse.st.base_V.row(0), mse.st.base_V.row(1), mse.st.base_V.row(2));
		mse.st.ray_threshold = Vector3(ray_clipping_threshold, ray_clipping_threshold, ray_clipping_threshold);
		// TODO FIXME BVH_BORDER
		mse.st.displace(bvh, bvh_test, bvh, true);

		for (uint8_t level = max_init_level; level > 0; --level) {
			mse.errors.push_back(mse.st.avg_downsampling_error());
			mse.st.coarsen_to_level((level - 1) << 3);
		}

		std::reverse(mse.errors.begin(), mse.errors.end());
	};

	// Cache up to 4 levels ahead to speed up things a bit...

	uint8_t max_level_1 = 4;
	auto init_mse_level_1 = std::bind(init_mse, std::placeholders::_1, max_level_1);

	std::for_each(std::execution::par_unseq, samplers.begin(), samplers.end(), init_mse_level_1);
	
	// Setup heap

	std::vector<uint32_t> heap;
	heap.reserve(samplers.size());
	for (uint32_t i = 0; i < samplers.size(); ++i)
		heap.push_back(i);

	auto heap_cmp = [&](uint32_t i, uint32_t j) { return samplers[i] < samplers[j]; };

	std::make_heap(heap.begin(), heap.end(), heap_cmp);

	long ufn = 0;
	for (const MicrosamplingError& mse : samplers)
		ufn += 1 << (2 * mse.current_level);

	// Determine subdivision levels by looking at the error incurred by down-sampling from the successive
	// subdivision level (note: there are obvious conter-examples where this is *NOT* a good approximation
	// of the geometric error of the micro-mesh relative to the input surface)

	while (ufn < target_ufn && heap.size() > 0) {
		// extract entry from heap
		std::pop_heap(heap.begin(), heap.end(), heap_cmp);
		uint32_t i = heap.back();
		heap.pop_back();

		if (samplers[i].current_error() < target_max_error) {
			break;
		}
		else {
			MicrosamplingError& mse = samplers[i];
			
			int face_ufn = 1 << (2 * mse.current_level);

			// if necessary, upsample and compute next error
			if (mse.current_level == mse.errors.size() - 1) {
				bool upsampled = false;

				if (mse.current_level == max_level_1 - 1) {
					mse.st.subdivide(uint8_t((max_level_1 + 1) << 3), mse.st.base_V.row(0), mse.st.base_V.row(1), mse.st.base_V.row(2));
					// TODO FIXME BVH_BORDER
					mse.st.displace(bvh, bvh_test, bvh, true);
					upsampled = true;
				}
				else {
					// FIXME this seems broken although it should do exactly what the 'if' branch does...
					upsampled = mse.st._upsample(bvh, ray_clipping_threshold, bvh_test);
				}

				if (upsampled)
					mse.errors.push_back(mse.st.avg_downsampling_error());
			}

			// increase level of entry if possible
			if (mse.current_level < mse.errors.size() - 1) {
				// update micro_fn
				mse.current_level++;
				ufn += (1 << (2 * mse.current_level)) - face_ufn;

				// reinsert updated entry
				heap.push_back(i);
				std::push_heap(heap.begin(), heap.end(), heap_cmp);
			}
		}
	}

	VectorXu8 subdivision_levels = VectorXu8::Constant(base_F.rows(), 0);
	for (const MicrosamplingError& mse : samplers)
		subdivision_levels(mse.st.base_fi) = mse.current_level;

	std::cout << "Adaptive subdivision computation took " << t.time_elapsed() << " seconds" << std::endl;

	return subdivision_levels;
}

VectorXu8 adjust_subdivision_levels(const MatrixXi& F, const VectorXu8& subdivisions, VectorXi8& corrections, RoundMode rm)
{
	int deg = F.cols();

	typedef std::pair<int, int> EFElem; // face index, face-edge index
	std::map<Edge, std::set<EFElem>> EF;
	for (int i = 0; i < F.rows(); ++i) {
		for (int j = 0; j < deg; ++j) {
			Edge e(F(i, j), F(i, (j + 1) % deg));
			EF[e].insert(std::make_pair(i, j));
		}
	}

	auto pool = [&rm](uint8_t s1, uint8_t s2) -> uint8_t {
		return (rm == RoundMode::Up) ? std::max(s1, s2) : std::min(s1, s2);
	};

	// tests true if diff with cmp is greater than 1 (according to the rounding mode)
	auto test_subdivision = [&rm](uint8_t s, uint8_t cmp) -> bool {
		return (rm == RoundMode::Up) ? (s < cmp - 1) : (s > cmp + 1);
	};

	auto adjust = [&rm](uint8_t s) -> uint8_t {
		return (rm == RoundMode::Up) ? s - 1 : s + 1;
	};

	bool changed = true;
	int npass = 0;
	while (changed) {
		changed = false;
		npass++;
		int k = 0;
		for (const auto& entry : EF) {
			uint8_t pooled_subdivision = (rm == RoundMode::Up) ? 0 : std::numeric_limits<uint8_t>::max();
			for (EFElem ef : entry.second) {
				uint8_t face_subdivision = subdivisions(ef.first) + corrections(ef.first);
				pooled_subdivision = pool(pooled_subdivision, face_subdivision);
			}
			for (EFElem ef : entry.second) {
				uint8_t face_subdivision = subdivisions(ef.first) + corrections(ef.first);
				if (test_subdivision(face_subdivision, pooled_subdivision)) {
					uint8_t adjusted_val = adjust(pooled_subdivision);
					corrections(ef.first) += int8_t(adjusted_val) - int8_t(face_subdivision);
					changed = true;
					k++;
				}
			}
		}
		std::cout << "Subdivision level adjustment - Pass " << npass << ", " << k << " corrections" << std::endl;
	}

	VectorXu8 decimation = VectorXu8::Constant(F.rows(), 0);

	for (const auto& entry : EF) {
		uint8_t max_subdivision = 0;
		uint8_t min_subdivision = std::numeric_limits<uint8_t>::max();
		for (EFElem ef : entry.second) {
			uint8_t face_subdivision = subdivisions(ef.first) + corrections(ef.first);
			max_subdivision = std::max(max_subdivision, face_subdivision);
			min_subdivision = std::min(min_subdivision, face_subdivision);
		}
		if (max_subdivision != min_subdivision) {
			Assert(min_subdivision == max_subdivision - 1);
			for (EFElem ef : entry.second) {
				uint8_t face_subdivision = subdivisions(ef.first) + corrections(ef.first);
				if (face_subdivision == max_subdivision) {
					decimation(ef.first) |= (1 << ef.second);
				}
			}
		}
	}

	std::cout << "Subdivision level adjustment - Done." << std::endl;

	return decimation;
}

void compute_subdivision_texture(const MatrixXi& F, const VectorXi& subdivision, MatrixX& VT, MatrixXi& FT)
{
	Assert(F.cols() == 3 && "TODO Add quad support");

	VT = MatrixX(3 * F.rows(), 2);
	FT = MatrixXi(F.rows(), F.cols());

	Vector2 vt0(0, 0);
	Vector2 vt10(1, 0);
	Vector2 vt20 = Vector2(0.5, 1);

	for (int i = 0; i < F.rows(); ++i) {
		FT(i, 0) = 3 * i;
		FT(i, 1) = 3 * i + 1;
		FT(i, 2) = 3 * i + 2;
		Scalar tris_per_side = std::sqrt(std::pow(2, 2 * subdivision(i))) / 2;
		VT.row(FT(i, 0)) = vt0;
		VT.row(FT(i, 1)) = vt10 * std::min(tris_per_side / 32.0, 1.0);
		VT.row(FT(i, 2)) = vt20 * std::min(tris_per_side / 32.0, 1.0);
	}
}

// TODO FIXME this should be rewritten
void subdivide_tri(const Vector3& v0, const Vector3& v1, const Vector3& v2, MatrixX& V, MatrixXi& F, uint8_t subdivision_bits)
{
	int level = subdivision_bits >> 3;
	bool d0 = subdivision_bits & SubdivisionDecimationFlag::Edge0;
	bool d1 = subdivision_bits & SubdivisionDecimationFlag::Edge1;
	bool d2 = subdivision_bits & SubdivisionDecimationFlag::Edge2;

	/*
	* LEVEL  NV   NE
	* 0      2    1
	* 1      3    2
	* 2      5    4
	* 3      9    8
	*/
	int edges_per_side = 1 << level;
	int n = edges_per_side + 1;

	int vn = (n * (n - 1)) / 2 + n;

	BarycentricGrid grid(level);

	Vector3 vr = (v1 - v0) / edges_per_side;
	Vector3 vc = (v2 - v1) / edges_per_side;

	V.resize(vn, 3);
	V.setZero();
	V.row(grid.index(0, 0)) = v0;
	V.row(grid.index(n - 1, 0)) = v1;
	V.row(grid.index(n - 1, n - 1)) = v2;

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j <= i; ++j) {
			V.row(grid.index(i, j)) = v0 + i * vr + j * vc;
		}
	}

	MatrixXi Fk(edges_per_side * edges_per_side, 3);
	Fk.setZero();

	int k = 0;
	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j <= i; ++j) {

			bool skip = d0 && (i % 2 == 0) && (j == 0);
			skip = skip || (d1 && (i == n - 2) && (j % 2 == 1));
			skip = skip || (d2 && (i % 2 == 1) && (j == i));

			if (!skip) {
				Fk(k, 0) = grid.index((d0 && j == 0) ? (i - 1) : i, j);
				Fk(k, 1) = grid.index(i + 1, j);
				Fk(k, 2) = grid.index((d2 && (j == i)) ? i + 2 : i + 1, ((d1 && i == n - 2) || (d2 && (j == i))) ? (j + 2) : (j + 1));

				if (d0 && d1 && i == n - 2 && j == 0) {
					Fk(k, 2) = grid.index(i, j + 1);
				}
				k++;
			}
			if (j < i && !(d1 && d2 && j == n - 3)) {
				Fk(k, 0) = grid.index((d0 && (i % 2) && j == 0) ? (i - 1) : i, j);
				Fk(k, 1) = grid.index(i + 1, (d1 && (j % 2 == 0) &&  i == n - 2) ? (j + 2) : (j + 1));
				Fk(k, 2) = grid.index((d2 && (i % 2) && (j == i - 1)) ? i + 1 : i, (d2 && (i % 2) && (j == i - 1)) ? j + 2 : j + 1);

				if (d0 && d1 && i == n - 2 && j == 0) {
					Fk(k, 0) = grid.index(i + 1, j);
				}
	
				k++;
			}
		}
	}

	if (k == 1) {
		Fk(0, 0) = grid.index(0, 0);
		Fk(0, 1) = grid.index(n - 1, 0);
		Fk(0, 2) = grid.index(n - 1, n - 1);
	}

	F = Fk.block(0, 0, k, 3);
}

MatrixX per_vertex_nearest_projection_vectors(const MatrixX& from_V, const MatrixXi& from_F, const MatrixX& to_V, const MatrixXi& to_F)
{
	std::cout << "Computing per vertex nearest projection vectors..." << std::endl;
	Timer t;

	int deg = from_F.cols();

	MatrixX P = MatrixX::Constant(from_V.rows(), from_V.cols(), 0);

	BVHTree bvh;
	bvh.build_tree(&to_V, &to_F, nullptr);

	MatrixX from_VN = compute_vertex_normals(from_V, from_F);
	MatrixX to_FN = compute_face_normals(to_V, to_F);

	auto bvh_nearest_test = [&](int from_vi, int to_fi) -> bool {
		return from_VN.row(from_vi).dot(to_FN.row(to_fi)) > 0;
	};

	std::vector<int8_t> referenced(from_V.rows(), 0);
	std::vector<int> verts;
	verts.reserve(from_V.rows());

	for (int fi = 0; fi < from_F.rows(); ++fi) {
		if (from_F(fi, 0) != INVALID_INDEX) {
			for (int i = 0; i < deg; ++i) {
				if (!referenced[from_F(fi, i)]) {
					referenced[from_F(fi, i)] = 1;
					verts.push_back(from_F(fi, i));
					Assert(verts.back() < from_VN.rows());
				}
			}
		}
	}

	std::cout << "Querying BVH..." << std::endl;

	auto vproject = [&](const int vi) -> void {
		Vector3 p_from = from_V.row(vi);
		Assert(vi < from_VN.rows());

		NearestInfo ni;
		if (bvh.nearest_point(p_from, &ni, std::bind(bvh_nearest_test, vi, std::placeholders::_1)))
			P.row(vi) = ni.p - p_from;
		else
			P.row(vi) = Vector3::Zero();
	};

	std::for_each(std::execution::par_unseq, verts.begin(), verts.end(), vproject);

	std::cout << "Closest direction computation took " << t.time_elapsed() << " seconds" << std::endl;
	std::cout << "Average time per base vertex: " << t.time_elapsed() / (Scalar)verts.size() << " seconds" << std::endl;

	return P;
}

void correct_directions_with_normals(MatrixX& VD, const MatrixX& base_VN)
{
	for (int i = 0; i < VD.rows(); ++i) {
		if (VD.row(i).dot(base_VN.row(i)) < 0) {
			VD.row(i) = -VD.row(i);
		}
	}
}

/*
MatrixX per_vertex_displacements_from_sampled_normals(const SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F)
{
	MatrixX VD = Eigen::MatrixXd::Zero(base_V.rows(), base_V.cols());

	for (const SubdivisionTri& uface : umesh.faces) {
		BarycentricGrid grid(uface.subdivision_level());
		int v0 = grid.index(0, 0);
		int v1 = grid.index(grid.samples_per_side() - 1, 0);
		int v2 = grid.index(grid.samples_per_side() - 1, grid.samples_per_side() - 1);

		VFAdjacency VF = compute_adjacency_vertex_face(uface.V, uface.F);
		VectorX dist;
		VectorXi pred;
		VectorXi cluster_V;
		VectorXi cluster_F;
		std::set<int> sources = { v0, v1, v2 };

		// call dijkstra on the ***displaced*** uface;
		dijkstra(uface.V + uface.VD, uface.F, VF, sources, dist, pred, cluster_V, cluster_F);
		
		for (int vi = 0; vi < uface.V.rows(); ++vi) {
			if (cluster_V(vi) == v0)
				VD.row(base_F(uface.base_fi, 0)) += uface.VN.row(vi);
			else if (cluster_V(vi) == v1)
				VD.row(base_F(uface.base_fi, 1)) += uface.VN.row(vi);
			else if (cluster_V(vi) == v2)
				VD.row(base_F(uface.base_fi, 2)) += uface.VN.row(vi);
		}
	}

	for (int i = 0; i < VD.rows(); ++i)
		VD.row(i).normalize();

	return VD;
}
*/

MatrixX per_vertex_displacements_from_micro_normals(const SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F)
{
	MatrixX VD = MatrixX::Zero(base_V.rows(), base_V.cols());
	MatrixX W = VectorX::Zero(base_V.rows());

	for (const SubdivisionTri& uface : umesh.faces) {
		BarycentricGrid grid(uface.subdivision_level());
		int v0 = grid.index(0, 0);
		int v1 = grid.index(grid.samples_per_side() - 1, 0);
		int v2 = grid.index(grid.samples_per_side() - 1, grid.samples_per_side() - 1);

		MatrixX displacedV = uface.V + uface.VD;
		for (int ufi = 0; ufi < uface.F.rows(); ++ufi) {
			Vector3 n = compute_face_normal(ufi, displacedV, uface.F);
			Scalar area = n.norm();
			Vector3 centroid = (uface.V.row(uface.F(ufi, 0)) + uface.V.row(uface.F(ufi, 1)) + uface.V.row(uface.F(ufi, 2))) / 3.0;
			Vector3 bary_coords = compute_bary_coords(centroid, uface.V.row(v0), uface.V.row(v1), uface.V.row(v2));

			VD.row(base_F(uface.base_fi, 0)) += n * bary_coords(0);
			VD.row(base_F(uface.base_fi, 1)) += n * bary_coords(1);
			VD.row(base_F(uface.base_fi, 2)) += n * bary_coords(2);
			
			W(base_F(uface.base_fi, 0)) += area * bary_coords(0);
			W(base_F(uface.base_fi, 1)) += area * bary_coords(1);
			W(base_F(uface.base_fi, 2)) += area * bary_coords(2);
		}
	}

	for (int i = 0; i < VD.rows(); ++i)
		VD.row(i).normalize();

	return VD;
}

MatrixX per_vertex_averaged_micro_nearest_projection_vectors(const SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN)
{
	std::cout << "Averaging nearest projections" << std::endl;
	BVHTree bvh;
	bvh.build_tree(&hi_V, &hi_F, &hi_VN);

	MatrixX VD = MatrixX::Zero(base_V.rows(), base_V.cols());
	MatrixX W = VectorX::Zero(base_V.rows());

	Timer t;
	for (const SubdivisionTri& uface : umesh.faces) {
		BarycentricGrid grid(uface.subdivision_level());
		int v0 = grid.index(0, 0);
		int v1 = grid.index(grid.samples_per_side() - 1, 0);
		int v2 = grid.index(grid.samples_per_side() - 1, grid.samples_per_side() - 1);

		MatrixX Vnearest = MatrixX::Constant(uface.V.rows(), uface.V.cols(), 0);
		for (int uvi = 0; uvi < uface.V.rows(); ++uvi) {
			NearestInfo ni;
			if (bvh.nearest_point(uface.V.row(uvi), &ni))
				Vnearest.row(uvi) = ni.p;
			else
				Vnearest.row(uvi) = uface.V.row(uvi);
			//Vnearest.row(uvi) = bvh.nearest_point(uface.V.row(uvi));
		}

		MatrixX uVD_new = Vnearest - uface.V;

		for (int uvi = 0; uvi < uface.V.rows(); ++uvi) {
			if (uface.ref(uvi) && uVD_new.row(uvi).dot(uface.VN.row(uvi)) < 0) {
				uVD_new.row(uvi) = -uVD_new.row(uvi);
			}
		}

		for (int uvi = 0; uvi < uface.V.rows(); ++uvi) {
			if (uface.ref(uvi)) {
				Vector3 vbary = compute_bary_coords(uface.V.row(uvi), uface.base_V.row(0), uface.base_V.row(1), uface.base_V.row(2));
				Scalar vd_length = uVD_new.row(uvi).norm();
				VD.row(base_F(uface.base_fi, 0)) += uVD_new.row(uvi) * vbary(0);
				VD.row(base_F(uface.base_fi, 1)) += uVD_new.row(uvi) * vbary(1);
				VD.row(base_F(uface.base_fi, 2)) += uVD_new.row(uvi) * vbary(2);

				W(base_F(uface.base_fi, 0)) += vd_length * vbary(0);
				W(base_F(uface.base_fi, 1)) += vd_length * vbary(1);
				W(base_F(uface.base_fi, 2)) += vd_length * vbary(2);
			}
		}
	}

	std::cout << "Averaging nearest projections took " << t.time_elapsed() << " seconds" << std::endl;
	std::cout << "    Time per base-face: " << t.time_elapsed() / umesh.base_fn << " seconds" << std::endl;

	for (int i = 0; i < VD.rows(); ++i)
		VD.row(i).normalize();

	return VD;
}

MatrixX per_vertex_base_displacements_from_micro_displacements(SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN)
{
	MatrixX VD = MatrixX::Zero(base_V.rows(), base_V.cols());
	MatrixX W = VectorX::Zero(base_V.rows());

	Timer t;
	for (SubdivisionTri& uface : umesh.faces) {

		for (int uvi = 0; uvi < uface.V.rows(); ++uvi) {
			if (uface.ref(uvi) && uface.VD.row(uvi).dot(uface.VN.row(uvi)) < 0) {
				uface.VD.row(uvi) = -uface.VD.row(uvi);
			}
		}

		for (int uvi = 0; uvi < uface.V.rows(); ++uvi) {
			if (uface.ref(uvi)) {
				Vector3 vbary = compute_bary_coords(uface.V.row(uvi), uface.base_V.row(0), uface.base_V.row(1), uface.base_V.row(2));
				Scalar vd_length = uface.VD.row(uvi).norm();
				VD.row(base_F(uface.base_fi, 0)) += uface.VD.row(uvi).normalized() * vbary(0) * (1.0 / uface.V.rows());
				VD.row(base_F(uface.base_fi, 1)) += uface.VD.row(uvi).normalized() * vbary(1) * (1.0 / uface.V.rows());
				VD.row(base_F(uface.base_fi, 2)) += uface.VD.row(uvi).normalized() * vbary(2) * (1.0 / uface.V.rows());
 
				W(base_F(uface.base_fi, 0)) += vd_length * vbary(0);
				W(base_F(uface.base_fi, 1)) += vd_length * vbary(1);
				W(base_F(uface.base_fi, 2)) += vd_length * vbary(2);
			}
		}
	}

	for (int i = 0; i < VD.rows(); ++i) {
		if (VD.row(i).norm() < 1e-8)
			VD.row(i) = base_VN.row(i);
		else
			VD.row(i).normalize();
	}

	return VD;
}

void optimize_base_mesh_positions_and_displacements(SubdivisionMesh& umesh, const MatrixX& base_V, const MatrixXi& base_F, MatrixX& new_V, MatrixX& new_VD)
{
	new_V = MatrixX::Zero(base_V.rows(), base_V.cols());
	new_VD = MatrixX::Zero(base_V.rows(), base_V.cols());

	VectorX WV = VectorX::Zero(base_V.rows());

	int deg = (int)base_F.cols();

	for (SubdivisionTri& uface : umesh.faces) {
		VectorX S(uface.V.rows());
		for (int uvi = 0; uvi < uface.V.rows(); ++uvi)
			S(uvi) = (uface.VD.row(uvi).dot(uface.VN.row(uvi)) < 0) ? -1 : +1;

		for (int uvi = 0; uvi < uface.V.rows(); ++uvi) {
			Vector3 vbary = compute_bary_coords(uface.V.row(uvi), uface.base_V.row(0), uface.base_V.row(1), uface.base_V.row(2));
			Scalar vd_length = uface.VD.row(uvi).norm();
			for (int i = 0; i < deg; ++i) {
				Scalar w = vbary(i) * (1.0 / uface.V.rows());
				new_VD.row(base_F(uface.base_fi, i)) += S(uvi) * uface.VD.row(uvi).normalized() * w;
				//new_V.row(base_F(uface.base_fi, i)) += (uface.V.row(uvi) + uface.VD.row(uvi)) * w;
				new_V.row(base_F(uface.base_fi, i)) += (uface.V.row(uvi) + uface.VD.row(uvi)) * (1.0 / uface.V.rows());
				//WV(base_F(uface.base_fi, i)) += w;
				WV(base_F(uface.base_fi, i)) += (1.0 / uface.V.rows());
			}
		}
	}

	for (int i = 0; i < (int)new_VD.rows(); ++i) {
		new_VD.row(i).normalize();
		new_V.row(i) /= WV(i);
	}
}

MatrixX optimize_base_mesh_positions_ls(const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN)
{
	typedef Eigen::Triplet<Scalar> Triplet;

	Timer t;

	// build bvh  on base
	BVHTree bvh;
	bvh.build_tree(&base_V, &base_F, &base_VN);

	MatrixX FN = compute_face_normals(base_V, base_F);

	// for each input vertex V:
	//    find nearest base-mesh projection P
	//    compute barycentric coefficients of P w.r.t base vertices
	//    add equation b1 v1 + b2 v2 + b3 v3 = V
	//    solve At A [xyz] = At b (or use QR or whatever)
	
	VectorX W = compute_voronoi_vertex_areas(hi_V, hi_F);

	std::vector<Triplet> coefficients;
	coefficients.reserve(hi_V.rows() * 3);

	MatrixX hi_V_proj(hi_V.rows(), hi_V.cols());
	hi_V_proj.setZero();
	
	for (int i = 0; i < (int)hi_V.rows(); ++i) {
		NearestInfo ni;
		bool nearest_found = bvh.nearest_point(hi_V.row(i), &ni, [&](int base_fi) -> bool {
			return FN.row(base_fi).dot(hi_VN.row(i)) > 0;
		});

		//Assert(nearest_found);

		if (nearest_found) {
			Vector3 bary = compute_bary_coords(ni.p, base_V.row(base_F(ni.fi, 0)), base_V.row(base_F(ni.fi, 1)), base_V.row(base_F(ni.fi, 2)));
			
			Assert(bary[0] >= 0);
			Assert(bary[1] >= 0);
			Assert(bary[2] >= 0);
			Assert(bary.sum() <= 1 + 1e-6);
			
			coefficients.push_back(Triplet(i, base_F(ni.fi, 0), W(i) * bary(0)));
			coefficients.push_back(Triplet(i, base_F(ni.fi, 1), W(i) * bary(1)));
			coefficients.push_back(Triplet(i, base_F(ni.fi, 2), W(i) * bary(2)));

			hi_V_proj.row(i) = ni.p;
		}
	}

	Scalar base_w = std::max(Scalar(W.minCoeff() * 0.2), Scalar(0.01));
	for (int vi = 0; vi < (int)base_V.rows(); ++vi) {
		coefficients.push_back(Triplet(hi_V.rows() + vi, vi, base_w));
	}

	SparseMatrix A(hi_V.rows() + base_V.rows(), base_V.rows());
	A.setFromTriplets(coefficients.begin(), coefficients.end());

	SparseMatrix AtA = A.transpose() * A;
	MatrixX AtB;
	{
		MatrixX B(hi_V.rows() + base_V.rows(), 3);
		B.block(0, 0, hi_V.rows(), 3) = W.asDiagonal() * hi_V;
		B.block(hi_V.rows(), 0, base_V.rows(), 3) = base_w * base_V;

		AtB = A.transpose() * B;
	}

	Eigen::SimplicialLDLT<SparseMatrix> ldlt;
	ldlt.compute(AtA);
	MatrixX new_base_V = ldlt.solve(AtB);

	//std::cout << "    Error: " << (AtA * new_base_V - AtB).norm() / AtB.norm() << std::endl;
	//{
	//	MatrixX r2 = A * new_base_V - W.asDiagonal() * hi_V;
	//	std::cout << "   residual = " << (r2.cwiseProduct(r2).sum()) << std::endl;
	//}
	//std::cout << "ldlt total time " << t.time_elapsed() << " seconds" << std::endl;;

	Assert(ldlt.info() == Eigen::Success);
	std::cout << "Solving " << A.rows() << "x" << A.cols() << " least squares system took " << t.time_elapsed() << " seconds." << std::endl;

	return new_base_V;
}

// computes the VN_from x VN_to matrix of barycentric coordinates of the projcections of 'from' onto 'to'
static SparseMatrix compute_projection_barycentrics(const MatrixX& to_V, const MatrixXi& to_F, const MatrixX& to_VN, const MatrixX& from_V, const MatrixXi& from_F, const MatrixX& from_VN)
{
	typedef Eigen::Triplet<Scalar> Triplet;

	// build bvh
	BVHTree bvh;
	bvh.build_tree(&to_V, &to_F, &to_VN);

	MatrixX FN = compute_face_normals(to_V, to_F);

	std::vector<Triplet> coefficients;
	coefficients.reserve(from_V.rows() * 3);

	for (int i = 0; i < (int)from_V.rows(); ++i) {
		NearestInfo ni;
		bool nearest_found = bvh.nearest_point(from_V.row(i), &ni, [&](int to_fi) -> bool {
			return FN.row(to_fi).dot(from_VN.row(i)) > 0;
		});

		Assert(nearest_found);

		if (nearest_found) {
			Vector3 bary = compute_bary_coords(ni.p, to_V.row(to_F(ni.fi, 0)), to_V.row(to_F(ni.fi, 1)), to_V.row(to_F(ni.fi, 2)));
			
			Assert(bary[0] >= 0);
			Assert(bary[1] >= 0);
			Assert(bary[2] >= 0);
			Assert(bary.sum() <= 1 + 1e-6);
			
			coefficients.push_back(Triplet(i, to_F(ni.fi, 0), bary(0)));
			coefficients.push_back(Triplet(i, to_F(ni.fi, 1), bary(1)));
			coefficients.push_back(Triplet(i, to_F(ni.fi, 2), bary(2)));
		}
	}

	SparseMatrix A(from_V.rows(), to_V.rows());
	A.setFromTriplets(coefficients.begin(), coefficients.end());

	return A;
	
}

MatrixX optimize_base_mesh_displacements_alternating(const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN, int iterations)
{
	// compute a list of vertex projections from hi to base
	// filter missed projections
	// initialization: accumulate projection vectors at base vertices, all with the same weight
	// At this point I have:
	//    a set of base displacement directions d_i
	//    a set of projection samples p_i, with the corresponding base face index and barycentric coordinates
	//    a set of input vertex positions q_i, corresponding to each sample
	// Alternating optimization
	//    STEP A: Adjust multipliers K
	//      for each projection sample
	//        compute the interpolated displacement direction
	//        find the ray parameter that minimizes the distance to the corresponding sample (ray-point distance, ray origin and directions are well defined)
	//    STEP B: Adjust base displacements
	//      This is a simple least squares fit where the displacement vectors become variables, the equation for each sample is
	//      p_i + b[0] * d0 + b[1] * d1 + b[2] * d[2] = q_i
	//         => K[i] * b[0] * d0 + K[i] * b[1] * d1 +K[i] * b[2] * d[2] = q_i - p_i

	Assert(iterations >= 0);

	VectorX W = compute_voronoi_vertex_areas(hi_V, hi_F);
	SparseMatrix M = compute_projection_barycentrics(base_V, base_F, base_VN, hi_V, hi_F, hi_VN);

	// find the vectors from the projected vertices to the original positions
	MatrixX hi_to_base_V = M * base_V;
	MatrixX diff_V = hi_V - hi_to_base_V;

	write_obj("base_proj.obj", base_V, base_F);
	write_obj("hi_proj.obj", hi_to_base_V, hi_F);

	MatrixX base_VD(base_V.rows(), 3);
	base_VD.setZero();

	// accumulate the difference vectors at base vertices
	for (int k = 0; k < M.outerSize(); ++k) {
		for (SparseMatrix::InnerIterator it(M, k); it; ++it) {
			int hi_vi = it.row();
			int base_vi = it.col();
			base_VD.row(base_vi) += W(hi_vi) * diff_V.row(hi_vi);
		}
	}

	base_VD.rowwise().normalize();

	for (int n = 0; n < iterations; ++n) {
		std::cout << "Alternating optimization - Iteration " << n + 1 << std::endl;
		// step A - find multipliers K
		MatrixX micro_VD = M * base_VD;

		VectorX K(micro_VD.rows());
		K.setZero();
		for (int i = 0; i < (int)micro_VD.rows(); ++i) {
			K(i) = project_onto_ray(hi_V.row(i), hi_to_base_V.row(i), micro_VD.row(i));
			//Assert(std::abs(micro_VD.row(i).dot((hi_to_base_V.row(i) + K(i) * micro_VD.row(i)) - hi_V.row(i))) < 1e-12);
		}
		write_obj("base_proj_disp_before.obj", hi_to_base_V + (K.asDiagonal() * (M * base_VD)), hi_F);

		// step B - adjust displacement vectors
		SparseMatrix A = (W.cwiseProduct(K)).asDiagonal() * M;
		SparseMatrix AtA = A.transpose() * A;
		MatrixX AtB = A.transpose() * (W.asDiagonal() * (diff_V));

		Eigen::SimplicialLDLT<SparseMatrix> ldlt;
		ldlt.compute(AtA);
		base_VD = ldlt.solve(AtB);

		Assert(ldlt.info() == Eigen::Success);
		
		write_obj("base_proj_disp.obj", hi_to_base_V + (K.asDiagonal() * (M * base_VD)), hi_F);
	}

	return base_VD;
}

MatrixX optimize_base_mesh_displacements_averaging(const MatrixX& base_V, const MatrixXi& base_F, const MatrixX& base_VN, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN)
{
	VectorX W = compute_voronoi_vertex_areas(hi_V, hi_F);
	SparseMatrix M = compute_projection_barycentrics(base_V, base_F, base_VN, hi_V, hi_F, hi_VN);

	// find the vectors from the projected vertices to the original positions
	MatrixX hi_to_base_V = M * base_V;
	MatrixX diff_V = hi_V - hi_to_base_V;

	write_obj("base_proj.obj", base_V, base_F);
	write_obj("hi_proj.obj", hi_to_base_V, hi_F);

	MatrixX base_VD(base_V.rows(), 3);
	base_VD.setZero();

	// accumulate the difference vectors at base vertices
	for (int k = 0; k < M.outerSize(); ++k) {
		for (SparseMatrix::InnerIterator it(M, k); it; ++it) {
			int hi_vi = it.row();
			int base_vi = it.col();
			base_VD.row(base_vi) += it.value() * W(hi_vi) * diff_V.row(hi_vi);
		}
	}

	base_VD.rowwise().normalize();

	return base_VD;
}

