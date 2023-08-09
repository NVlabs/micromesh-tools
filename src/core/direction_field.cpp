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

#include "direction_field.h"
#include "tangent.h"

void compute_direction_field(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, MatrixX& D, VectorX& DW)
{
	D = MatrixX::Constant(V.rows(), 3, 0);
	DW = VectorX::Constant(V.rows(), 0);

	MatrixX FN = compute_face_normals(V, F);
	std::map<Edge, std::vector<int>> EF;

	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX) {
			for (int j = 0; j < 3; ++j) {
				Edge e(F(fi, j), F(fi, (j + 1) % 3));
				EF[e].push_back(fi);
			}
		}
	}

	for (const auto& entry : EF) {
		const Edge& e = entry.first;
		const std::vector<int>& fv = entry.second;
		if (fv.size() == 2) {
			Scalar dw = vector_angle(FN.row(fv[0]), FN.row(fv[1])); // angle at e
			Vector3 d = dw * (V.row(e.first) - V.row(e.second));
			D.row(e.first) += oriented_direction(D.row(e.first), d);
			D.row(e.second) += oriented_direction(D.row(e.second), d);
			DW(e.first) += dw;
			DW(e.second) += dw;
		}
	}
}

void smooth_direction_field(const MatrixXi& F, const VFAdjacency& VF, MatrixX& D, VectorX& DW)
{
	for (long i = 0; i < D.rows(); ++i) {
		int n = 1;
		for (const VFEntry& vfe : VF[i]) {
			n++;
			int vnext = F(vfe.first, (vfe.second + 1) % 3);
			D.row(i) += oriented_direction(D.row(i), D.row(vnext));
			DW(i) += DW(vnext);
		}
		D.row(i) /= (Scalar)n;
		DW(i) /= (Scalar)n;
	}
}

void update_direction_field_on_collapse(const Edge& e, MatrixX& D, VectorX& DW)
{
	D.row(e.first) += oriented_direction(D.row(e.first), D.row(e.second));
	DW(e.first) += DW(e.second);
	
	D.row(e.second).setZero();
	DW(e.second) = 0;
}

