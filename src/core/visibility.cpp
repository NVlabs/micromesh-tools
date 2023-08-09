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

#include "visibility.h"
#include "tangent.h"

#include "mesh_io.h"

#include <algorithm>
#include <execution>
#include <random>
#include <fstream>

static Scalar compute_max_angle(const std::vector<Vector3>& half_planes, const Vector3& d_new)
{
	Scalar max_angle = 0;
	for (const Vector3& n : half_planes) {
		max_angle = std::max(max_angle, vector_angle(n, d_new));
	}
	return max_angle;
}

static inline Vector3 get_point_on_sphere(Scalar theta, Scalar phi)
{
	Scalar st = std::sin(theta);
	Scalar ct = std::cos(theta);
	Scalar sp = std::sin(phi);
	Scalar cp = std::cos(phi);

	return Vector3(ct * sp, cp, st * sp);
}

static Vector3 compute_optimal_visibility_direction_with_local_search(const std::vector<Vector3>& half_spaces, int random_seed)
{
	Vector3 d_opt = Vector3(1, 0, 0); // just a random starting vector

	Scalar current_max_angle = std::numeric_limits<Scalar>::max();

	auto evaluate_direction = [&](const Vector3& d_new) -> void {
		Scalar new_max_angle = compute_max_angle(half_spaces, d_new);
		if (new_max_angle < current_max_angle) {
			current_max_angle = new_max_angle;
			d_opt = d_new;
			d_opt.normalize();
		}
	};

	int theta_ticks = 16;
	int phi_ticks = 8;
	for (int i = 0; i < theta_ticks; ++i) {
		Scalar theta = 2 * M_PI * (i / Scalar(theta_ticks));
		for (int j = 0; j < phi_ticks; ++j) {
			Scalar phi = std::acos(1 - 2 *(j / Scalar(phi_ticks)));
			evaluate_direction(get_point_on_sphere(theta, phi));
		}
	}

	std::default_random_engine g(random_seed);
	std::uniform_real_distribution<Scalar> u(0, 1);

	int nsteps = 300;
	for (int n = 0; n < nsteps; n++) {
		// sample random point on the unit sphere
		Scalar theta = 2 * u(g) * M_PI;
		Scalar phi = std::acos(1 - 2 * u(g));

		Vector3 p = get_point_on_sphere(theta, phi);

		// compute axis of rotation from d_opt to p (an arc on the unit sphere)
		Vector3 axis = p.cross(d_opt).normalized();

		// compute angle of rotation (decreases with iterations)
		Scalar angle = radians((1 - (n / Scalar(nsteps))) * 2);

		// compute new position
		Vector3 d_new = rotate(d_opt, axis, angle);

		evaluate_direction(d_new);
	}

	return d_opt;
}

MatrixX compute_displacement_directions_from_convex_cone(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF)
{
	MatrixX FN = compute_face_normals(V, F);

	std::vector<int> verts(V.rows(), -1);
	std::iota(verts.begin(), verts.end(), 0);

	MatrixX VD = MatrixX::Constant(V.rows(), 3, 0);

	auto compute_direction = [&](int vi) -> void {
		std::vector<Vector3> half_spaces;
		for (const VFEntry& vfe : VF[vi])
			half_spaces.push_back(FN.row(vfe.first));

		VD.row(vi) = compute_optimal_visibility_direction_with_local_search(half_spaces, vi);
	};

	Timer t;
	std::for_each(std::execution::par_unseq, verts.begin(), verts.end(), compute_direction);
	std::cout << "compute_displacement_directions_from_convex_cone() took " << t.time_elapsed() << " seconds" << std::endl;

	return VD;
}

// returns optimal (positive) visibility from 3 unit vectors
static inline std::pair<Vector3, Scalar> min_cap3(const Vector3& n0, const Vector3& n1, const Vector3& n2)
{
	Vector3 n = ((n1 - n0).cross(n2 - n0)).normalized();
	Scalar n_corr = n.dot(n0);
	if (n_corr < 0)
		return std::make_pair(Vector3(-n), -n_corr);
	else
		return std::make_pair(n, n_corr);
};

// returns optimal (positive) visibility from 2 unit vectors
static inline std::pair<Vector3, Scalar> min_cap2(const Vector3& n0, const Vector3& n1)
{
	Vector3 n = (0.5 * (n0 + n1)).normalized();
	Scalar n_corr = n.dot(n0);
	return std::make_pair(n, n_corr);
};

// tests a unit vector against a spherical cap (represented by a direction and a radius, i.e. a dot product or correlation threshold)
static inline bool outside(const Vector3& n, const Scalar n_corr, const Vector3& test_direction)
{
	return n.dot(test_direction) - n_corr < -1e-5;
}

std::pair<Vector3, Scalar> compute_positive_visibility_from_directions(const std::vector<Vector3>& directions, Scalar null_tolerance)
{
	std::vector<Vector3> normals;
	for (const Vector3& dir : directions) {
		Scalar length = dir.norm();
		if (length > null_tolerance)
			normals.push_back(dir / length);
	}

	int n = (int)normals.size();

	if (n == 0)
		return std::make_pair(Vector3(0, 0, 0), -1);

	std::vector<bool> frontier(normals.size(), false);

	int support[3] = { -1, -1, -1 };
	int support_size = 0;

	Vector3 x = Vector3(0, 0, 0);
	Scalar x_corr = 1;

#define SET_SUPPORT_1(i0) { support[0] = i0; support_size = 1; x = normals[i0]; x_corr = 1; }
#define SET_SUPPORT_2(i0, i1) { support[0] = i0; support[1] = i1; support_size = 2; std::tie(x, x_corr) = min_cap2(normals[i0], normals[i1]); }
#define SET_SUPPORT_3(i0, i1, i2) { support[0] = i0; support[1] = i1; support[2] = i2; support_size = 3; std::tie(x, x_corr) = min_cap3(normals[i0], normals[i1], normals[i2]); }

	SET_SUPPORT_1(0)

	while (true) {
		// find outlier point
		for (int i = 0; i < support_size; ++i)
			frontier[support[i]] = true;

		int ni = -1;
		for (int i = 0; i < n; ++i) {
			if (!frontier[i] && outside(x, x_corr, normals[i])) {
				ni = i;
				break;
			}
		}

		if (ni == -1)
			break;

		for (int i = 0; i < support_size; ++i)
			frontier[support[i]] = false;

		Vector3 x_prev = x;
		Scalar x_corr_prev = x_corr;

		switch (support_size) {
		case 1: {
			int n0 = support[0];
			SET_SUPPORT_2(n0, ni);
		} break;
		case 2: {
			int n0 = support[0];
			int n1 = support[1];
			std::pair<Vector3, Scalar> x0i, x1i;
			x0i = min_cap2(normals[n0], normals[ni]);
			x1i = min_cap2(normals[n1], normals[ni]);
			if (x0i.second <= x1i.second && !outside(x0i.first, x0i.second, normals[n1])) {
				// x0i is the min cap
				SET_SUPPORT_2(n0, ni)
			}
			else if (x1i.second <= x0i.second && !outside(x1i.first, x1i.second, normals[n0])) {
				// x1i is the min cap
				SET_SUPPORT_2(n1, ni)
			}
			else {
				// 3-b solution
				SET_SUPPORT_3(n0, n1, ni)
			}
		} break;
		case 3: {
			int n0 = support[0];
			int n1 = support[1];
			int n2 = support[2];
			std::pair<Vector3, Scalar> x0i, x1i, x2i;
			x0i = min_cap2(normals[n0], normals[ni]);
			x1i = min_cap2(normals[n1], normals[ni]);
			x2i = min_cap2(normals[n2], normals[ni]);
			if (x0i.second <= x1i.second && x0i.second <= x2i.second && !outside(x0i.first, x0i.second, normals[n1]) && !outside(x0i.first, x0i.second, normals[n2])) {
				SET_SUPPORT_2(n0, ni);
			}
			else if (x1i.second <= x0i.second && x1i.second <= x2i.second && !outside(x1i.first, x1i.second, normals[n0]) && !outside(x1i.first, x1i.second, normals[n2])) {
				SET_SUPPORT_2(n1, ni);
			}
			else if (x2i.second <= x0i.second && x2i.second <= x1i.second && !outside(x2i.first, x2i.second, normals[n0]) && !outside(x2i.first, x2i.second, normals[n1])) {
				SET_SUPPORT_2(n2, ni);
			}
			else {
				// 3-b solution
				// take the smallest valid if it exists, otherwise return false
				x_corr = -2;

				std::pair<Vector3, Scalar> xt = min_cap3(normals[n0], normals[n1], normals[ni]);
				if (xt.second > x_corr && !outside(xt.first, xt.second, normals[n2]))
					SET_SUPPORT_3(n0, n1, ni);

				xt = min_cap3(normals[n0], normals[n2], normals[ni]);
				if (xt.second > x_corr && !outside(xt.first, xt.second, normals[n1]))
					SET_SUPPORT_3(n0, n2, ni);

				xt = min_cap3(normals[n1], normals[n2], normals[ni]);
				if (xt.second > x_corr && !outside(xt.first, xt.second, normals[n0]))
					SET_SUPPORT_3(n1, n2, ni);

			}
		} break;
		default:
			Assert(0 && "Invalid support size");
		}

		if (x_corr == -2 || x_corr >= x_corr_prev) { // no valid positive cap exists, return x_prev as approximate solution
			x = x_prev;
			x_corr = x_corr_prev;
			for (const Vector3& n : normals) {
				x_corr = std::min(x_corr, x.dot(n));
			}
			return std::make_pair(x, x_corr);
		}
	}

#undef SET_SUPPORT_1
#undef SET_SUPPORT_2
#undef SET_SUPPORT_3

	for (int i = 0; i < n; ++i)
		frontier[i] = false;

	for (int i = 0; i < support_size; ++i)
		frontier[support[i]] = true;

	// if feasible, we found the min cap, otherwise we failed (the cap is negative)
	// just to make sure, ensure it's feasible
	for (int i = 0; i < n; ++i) {
		Scalar corr = x.dot(normals[i]);
		if (!frontier[i] && outside(x, x_corr, normals[i])) {
			std::cout << "Very bad!" << std::endl;
		}
	}

	return std::make_pair(x, x_corr);
}

std::pair<MatrixX, VectorX> compute_optimal_visibility_directions(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF)
{
	MatrixX FN = compute_face_area_vectors(V, F);

	std::vector<int> verts(V.rows(), -1);
	std::iota(verts.begin(), verts.end(), 0);

	MatrixX VD = MatrixX::Constant(V.rows(), 3, 0);
	VectorX VIS = VectorX::Constant(V.rows(), -1);

	auto compute_direction = [&](int vi) -> void {
		std::vector<Vector3> area_vectors;
		for (const VFEntry& vfe : VF[vi])
			area_vectors.push_back(FN.row(vfe.first).normalized());

		std::pair<Vector3, Scalar> p = compute_positive_visibility_from_directions(area_vectors);
		VD.row(vi) = p.first;
		VIS(vi) = p.second;
	};

	std::for_each(std::execution::par_unseq, verts.begin(), verts.end(), compute_direction);

	return std::make_pair(VD, VIS);
}

