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

#include "flip.h"
#include "utils.h"
#include "local_operations.h"
#include "tangent.h"
#include "direction_field.h"
#include "bvh.h"
#include "quality.h"

#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <execution>
#include <random>

// Sums the directions around the flap
// Returns true if flipping the edge aligns with the summed directions better (smaller angle)
static bool flip_follows_direction_field(const Flap& flap, const MatrixX& V, const MatrixXi& F, const MatrixX& D, const VectorX& DW, Scalar min_angle)
{
	Vector4i ring = vertex_ring(flap, F);
	Vector3 direction = D.row(ring(0));
	for (int i = 1; i < 4; ++i)
		direction += oriented_direction(direction, D.row(ring(i)));

	Vector3 edge_dir = oriented_direction(direction, V.row(ring(1)) - V.row(ring(3)));
	Vector3 flip_dir = oriented_direction(direction, V.row(ring(0)) - V.row(ring(2)));

	Scalar edge_angle = vector_angle(direction, edge_dir);
	Scalar flip_angle = vector_angle(direction, flip_dir);

	return flip_angle < edge_angle && flip_angle <= min_angle;
}

static int flip_valence(const Flap& flap, const MatrixXi& F, const VFAdjacency& VF)
{
	Vector4i ring = vertex_ring(flap, F);
	Vector4i valence;
	valence(0) = VF[ring(0)].size() + 1;
	valence(1) = VF[ring(1)].size() - 1;
	valence(2) = VF[ring(2)].size() + 1;
	valence(3) = VF[ring(3)].size() - 1;
	return valence.maxCoeff();
}

static bool flip_follows_direction_field_2(const Flap& flap, const MatrixX& V, const MatrixXi& F, const MatrixX& D, const VectorX& DW, Scalar min_angle)
{
	Vector4i ring = vertex_ring(flap, F);
	
	Vector3 edge_direction = D.row(ring(1));
	Vector3 flip_direction = D.row(ring(0));

	edge_direction += oriented_direction(edge_direction, D.row(ring(3)));
	flip_direction += oriented_direction(flip_direction, D.row(ring(2)));


	Vector3 edge_dir = oriented_direction(edge_direction, V.row(ring(1)) - V.row(ring(3)));
	Vector3 flip_dir = oriented_direction(flip_direction, V.row(ring(0)) - V.row(ring(2)));

	Scalar edge_angle = vector_angle(edge_direction, edge_dir);
	Scalar flip_angle = vector_angle(flip_direction, flip_dir);

	//Assert(edge_angle <= radians(90));
	//Assert(flip_angle <= radians(90));

	return edge_angle > (radians(90) - min_angle) && flip_angle < min_angle;
}

int align_to_direction_field(const MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, const MatrixX& D, const VectorX& DW)
{
	Timer t;
	Assert(F.cols() == 3);

	int nflip = 0;
	for (int fi = 0; fi < (int)F.rows(); ++fi) {
		if (F(fi, 0) == INVALID_INDEX)
			continue;

		for (int j = 0; j < 3; ++j) {
			Edge e(F(fi, j), F(fi, (j + 1) % 3));
			Flap flap = compute_flap(e, F, VF);
			bool flip = false;
			if (flap.size() == 2) {
				if (flap_normals_angle(flap, V, F) > radians(45)) {
					if (flip_follows_direction_field(flap, V, F, D, DW, radians(15)) && flip_preserves_topology(flap, F, VF)) {
						FlipInfo info = flip_edge(flap, V, F, VF);
						Assert(info.ok);
						flip = true;
						nflip++;
					}
				}
			}
		}
	}

	return nflip;
}

int flip_pass(const MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, const MatrixX& D, const VectorX& DW)
{
	Timer t;
	Assert(F.cols() == 3);

	int nflip = 0;
	for (int fi = 0; fi < (int)F.rows(); ++fi) {
		if (F(fi, 0) == INVALID_INDEX)
			continue;

		for (int j = 0; j < 3; ++j) {
			Edge e(F(fi, j), F(fi, (j + 1) % 3));
			Flap flap = compute_flap(e, F, VF);
			bool flip = false;
			if (flap.size() == 2) {
				if (flap_normals_angle(flap, V, F) < radians(30)) {
					//if (flip_improves_diagonal_split(flap, V, F, radians(120)) && flip_valence(flap, F, VF) <= 6)
					if (flip_improves_diagonal_split(flap, V, F, radians(120)))
						flip = true;
					//else
					//	if (flip_preserves_aspect_ratio(flap, V, F, VF))
					//		flip = true;
				}
				else {
					//if (flip_follows_direction_field_2(flap, V, F, D, DW, radians(20)) && flip_valence(flap, F, VF) <= 6)
					//	flip = true;
				}

				if (flip && flip_preserves_geometry(flap, V, F) && flip_preserves_topology(flap, F, VF)) {
					FlipInfo info = flip_edge(flap, V, F, VF);
					Assert(info.ok);
					nflip++;
				}
			}
		}
	}

	return nflip;
}

static Vector3 sample_bary()
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<Scalar> u(0, 1);

	Scalar a = u(generator);
	Scalar b = u(generator);

	return Vector3(std::min(a, b), std::max(a, b) - std::min(a, b), 1 - std::max(a, b));
};

int flip_to_reduce_error(const MatrixX& V, MatrixXi& F, VFAdjacency& VF, VectorXu8& VB, const MatrixX& hi_V, const MatrixXi& hi_F)
{
	// build bvh of hi
	BVHTree bvh;
	bvh.build_tree(&hi_V, &hi_F, nullptr);

	MatrixX hi_FN = compute_face_normals(hi_V, hi_F);

	typedef std::pair<Flap, Scalar> FlipOp;

	std::set<Edge> visited_edges;
	std::vector<FlipOp> flips;

	// iterate over flaps, computing the geometric error of  non-flat flaps before and after flips
	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX) {
			for (int j = 0; j < 3; ++j) {
				Edge e(F(fi, j), F(fi, (j + 1) % 3));
				if (visited_edges.find(e) == visited_edges.end()) {
					visited_edges.insert(e);
					Flap flap = compute_flap(e, F, VF);
					if (flap.size() == 2 && flap_normals_angle(flap, V, F) > radians(30))
						flips.push_back(std::make_pair(flap, 0));
				}
			}
		}
	}

	auto sample_face_error = [&](const Vector3& p0, const Vector3& p1, const Vector3& p2, int n_samples) -> std::pair<Scalar, Scalar> {
		Vector3 n = compute_area_vector(p0, p1, p2);

		auto sample_test = [&](int hi_fi) -> bool {
			return n.dot(hi_FN.row(hi_fi)) > 0;
		};

		Scalar cumulative_error = 0;

		for (int i = 0; i < n_samples; ++i) {
			Vector3 s = sample_bary();
			Vector3 p = s(0) * p0 + s(1) * p1 + s(2) * p2;

			NearestInfo ni;
			if (bvh.nearest_point(p, &ni, sample_test))
				cumulative_error += (p - ni.p).norm();
			else
				cumulative_error += Infinity;
		}

		return std::make_pair(cumulative_error, n.norm());;
	};

	constexpr int num_samples = 5;
	//constexpr int num_samples = 1;

	auto compute_flip_gain = [&](FlipOp& fo) -> void {
		Vector4i ring = vertex_ring(fo.first, F);
		std::pair<Scalar, Scalar> e0 = sample_face_error(V.row(ring(0)), V.row(ring(1)), V.row(ring(3)), num_samples);
		std::pair<Scalar, Scalar> e1 = sample_face_error(V.row(ring(1)), V.row(ring(2)), V.row(ring(3)), num_samples);
		std::pair<Scalar, Scalar> ef0 = sample_face_error(V.row(ring(0)), V.row(ring(1)), V.row(ring(2)), num_samples);
		std::pair<Scalar, Scalar> ef1 = sample_face_error(V.row(ring(2)), V.row(ring(3)), V.row(ring(0)), num_samples);

		Scalar error_pre = (e0.first * e0.second + e1.first * e1.second) / (e0.second + e1.second);
		Scalar error_post = (ef0.first * ef0.second + ef1.first * ef1.second) / (ef0.second + ef1.second);

		fo.second = error_post - error_pre;
	};

	std::for_each(std::execution::par_unseq, flips.begin(), flips.end(), compute_flip_gain);

	// sort the flaps by largest error-removal

	std::sort(flips.begin(), flips.end(), [](const FlipOp& op1, const FlipOp& op2) { return op1.second < op2.second; });

	// initialize a visit vector
	VectorXu8 visited = VectorXu8::Constant(F.rows(), 0);

	int nflips = 0;

	// for each operation, if both faces are not visited, flip the edge
	for (const FlipOp& fo : flips) {
		const Flap& flap = fo.first;
		if (fo.second < 0 && !visited(flap.f[0]) && !visited(flap.f[1])) {
			if (/* flip_preserves_geometry(V, F, flap.f[0], flap.f[1], flap.e[0], flap.e[1]) && */ flip_preserves_topology(flap, F, VF)) {
				flip_edge(flap, V, F, VF);
				visited(flap.f[0]) = 1;
				visited(flap.f[1]) = 1;
				nflips++;
			}
		}
	}

	return nflips;
}

