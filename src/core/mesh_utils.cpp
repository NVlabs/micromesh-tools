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

#include "mesh_utils.h"
#include "utils.h"

#include <vector>

Scalar average_edge(const MatrixX& V, const MatrixXi& F)
{
	Scalar avg_edge = 0;
	int n = 0;
	int deg = F.cols();
	for (int i = 0; i < F.rows(); ++i) {
		for (int j = 0; j < deg; ++j) {
			avg_edge += (V.row(F(i, j)) - V.row(F(i, (j + 1) % deg))).norm();
			n++;
		}
	}
	return avg_edge / n;
}

Scalar average_area(const MatrixX& V, const MatrixXi& F)
{
	Scalar area = 0;
	int fn = 0;
	for (int i = 0; i < F.rows(); ++i) {
		Scalar f_area = face_area(V, F.row(i));
		area += f_area;
		if (f_area > 0)
			fn++;
	}

	return area / fn;
}

Scalar face_area(const MatrixX& V, const VectorXi& f)
{
	std::vector<Vector3> fv;
	for (int i = 0; i < f.size(); ++i)
		if (f(i) != INVALID_INDEX)
			fv.push_back(V.row(f(i)));

	if (fv.size() < 3)
		return 0;
	else if (fv.size() == 3)
		return ((fv[1] - fv[0]).cross(fv[2] - fv[0])).norm() / 2.0;
	else
		return ((fv[1] - fv[0]).cross(fv[2] - fv[0])).norm() / 2.0
			+ ((fv[2] - fv[0]).cross(fv[3] - fv[0])).norm() / 2.0;
}


Scalar mesh_displacement_volume(const MatrixX& V, const MatrixX& VD, const MatrixXi& F)
{
	Scalar total_volume = 0;
	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			total_volume += prismoid_volume_approximate(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)), VD.row(F(i, 0)), VD.row(F(i, 1)), VD.row(F(i, 2)));
		}
	}
	return total_volume;
}

VectorX compute_voronoi_vertex_areas(const MatrixX& V, const MatrixXi& F)
{
	int deg = (int)F.cols();
	Assert(deg == 3);
	
	VectorX voronoi_areas = VectorX::Constant(V.rows(), 0);

	for (int i = 0; i < (int)F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			Scalar angle[3] = {};
			for (int j = 0; j < deg; ++j) {
				angle[j] = vector_angle(V.row(F(i, (j + 1) % deg)) - V.row(F(i, j)), V.row(F(i, (j + 2) % deg)) - V.row(F(i, j)));
			}

			bool obtuse = angle[0] > M_PI_2 || angle[1] > M_PI_2 || angle[2] > M_PI_2;

			for (int j = 0; j < deg; ++j) {
				if (obtuse) {
					Scalar area = face_area(V, F.row(i));
					voronoi_areas(F(i, j)) += angle[j] > M_PI_2 ? 0.5 * area : 0.25 * area;
				}
				else {
					int j1 = (j + 1) % deg;
					int j2 = (j + 2) % deg;
					voronoi_areas(F(i, j)) += 0.125 * (
						(V.row(F(i, j1)) - V.row(F(i, j))).squaredNorm() * cot(angle[j2])
						+ (V.row(F(i, j2)) - V.row(F(i, j))).squaredNorm() * cot(angle[j1])
					);
				}
			}
		}
	}

	return voronoi_areas;
}

VectorX compute_face_aspect_ratios(const MatrixX& V, const MatrixXi& F)
{
	VectorX FR = VectorX::Constant(F.rows(), 0);

	for (int fi = 0; fi < F.rows(); ++fi)
		FR(fi) = aspect_ratio(V.row(F(fi, 0)), V.row(F(fi, 1)), V.row(F(fi, 2)));

	return FR;
}

void update_face_aspect_ratios_around_vertex(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, int vi, VectorX& FR)
{
	for (const VFEntry& vfe : VF[vi]) {
		int fi = vfe.first;
		Scalar ar = aspect_ratio(V.row(F(fi, 0)), V.row(F(fi, 1)), V.row(F(fi, 2)));
		FR(fi) = std::max(FR(fi), ar);
	}
}

std::vector<int> vector_of_indices(long n)
{
	std::vector<int> v(n);
	std::iota(v.begin(), v.end(), 0);
	return v;
}

