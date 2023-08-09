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

#include "ambient_occlusion.h"

#include <random>
#include <execution>

#include "bvh.h"
#include "mesh_utils.h"
#include "utils.h"

static inline Vector3 get_point_on_sphere(Scalar theta, Scalar phi)
{
	Scalar st = std::sin(theta);
	Scalar ct = std::cos(theta);
	Scalar sp = std::sin(phi);
	Scalar cp = std::cos(phi);

	return Vector3(ct * sp, cp, st * sp);
}

void compute_ambient_occlusion(const MatrixX& V, MatrixXi& F, const MatrixX& VN, VectorX& AO, int n_samples)
{
	std::default_random_engine g(0);
	std::uniform_real_distribution<Scalar> u(0, 1);

	std::vector<Vector3> samples;
	for (int n = 0; n < n_samples; n++) {
		// sample random point on the unit hemisphere
		Scalar theta = 2 * u(g) * M_PI;
		Scalar phi = std::acos(1 - 2 * u(g)); // cap between 0 and 1

		Vector3 p = get_point_on_sphere(theta, phi);
		samples.push_back(p);
	}

	AO = VectorX::Constant(V.rows(), 0);

	BVHTree bvh;
	bvh.build_tree(&V, &F, &VN);

	std::vector<int> indices = vector_of_indices(V.rows());

	auto f = [&](int vi) {
		int n_hits = 0;
		for (Vector3 d : samples) {
			if (d.dot(VN.row(vi)) < 0)
				d *= -1;
			IntersectionInfo ii;
			if (bvh.ray_intersection(V.row(vi), d, &ii, [] (const IntersectionInfo& ii) { return ii.t > 1e-4; })) {
				n_hits++;
			}
		}
		AO(vi) = n_hits / Scalar(n_samples);
	};

	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), f);
}

