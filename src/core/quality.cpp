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

#include "quality.h"

#include "bvh.h"
#include "tangent.h"
#include "mesh_utils.h"
#include "utils.h"

#include <vector>
#include <algorithm>
#include <execution>

static Scalar stretch(const Vector2& u10, const Vector2& u20, const Vector2& w10, const Vector2& w20)
{
	Matrix2 M1;
	M1.col(0) = u10;
	M1.col(1) = u20;

	Matrix2 M2;
	M2.col(0) = w10;
	M2.col(1) = w20;

	Matrix2 T = M2 * M1.inverse();

	Matrix2 U, V;
	Vector2 s;
	Eigen::JacobiSVD<Matrix2, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
	svd.compute(T);
	s = svd.singularValues();

	if (std::isnan(s.minCoeff()) || std::isnan(s.maxCoeff()))
		return 0;
	else
		return s.maxCoeff() / s.minCoeff();
}

void compute_face_quality_aspect_ratio(const MatrixX& V, const MatrixXi& F, VectorX& FQ)
{
	FQ.setConstant(F.rows(), 0);

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX)
			FQ(i) = aspect_ratio(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)));
	}
}

void compute_face_quality_stretch(const MatrixX& V1, const MatrixX& V2, const MatrixXi& F, VectorX& FQ)
{
	FQ.setConstant(F.rows(), 0);

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			Matrix3 M1 = local_frame(V1.row(F(i, 0)), V1.row(F(i, 1)), V1.row(F(i, 2)));
			Matrix3 M2 = local_frame(V2.row(F(i, 0)), V2.row(F(i, 1)), V2.row(F(i, 2)));
			Vector2 u10_1 = M1.block(0, 0, 2, 3) * (V1.row(F(i, 1)) - V1.row(F(i, 0))).transpose();
			Vector2 u20_1 = M1.block(0, 0, 2, 3) * (V1.row(F(i, 2)) - V1.row(F(i, 0))).transpose();
			Vector2 u10_2 = M2.block(0, 0, 2, 3) * (V2.row(F(i, 1)) - V2.row(F(i, 0))).transpose();
			Vector2 u20_2 = M2.block(0, 0, 2, 3) * (V2.row(F(i, 2)) - V2.row(F(i, 0))).transpose();

			FQ(i) = Scalar(1) / stretch(u10_1, u20_1, u10_2, u20_2);
		}
	}
}

void compute_vertex_quality_hausdorff_distance(const MatrixX& from_V, const MatrixX& from_VN, const MatrixXi& from_F, const MatrixX& to_V, const MatrixX& to_VN, const MatrixXi& to_F, VectorX& VQ)
{
	Box3 from_box;
	for (int vi = 0; vi < from_V.rows(); ++vi)
		from_box.add(from_V.row(vi));

	VQ = VectorX::Constant(from_V.rows(), std::numeric_limits<Scalar>::max());

	BVHTree bvh;
	bvh.build_tree(&to_V, &to_F, &to_VN, 16);

	VFAdjacency VF = compute_adjacency_vertex_face(from_V, from_F);
	VectorXu8 VB;
	per_vertex_border_flag(from_V, from_F, VB);

	std::vector<int> indices = vector_of_indices(from_V.rows());

	MatrixX to_FN = compute_face_normals(to_V, to_F);;

	auto bvh_filter = [&](int from_vi, int to_fi) -> bool {
		return from_VN.row(from_vi).dot(to_FN.row(to_fi)) > 0;
	};

	auto f = [&](int vi) {
		NearestInfo ni;
		//bool found_nearest = bvh.nearest_point(from_V.row(vi), &ni, std::bind(bvh_filter, vi, std::placeholders::_1));
		bool found_nearest = bvh.nearest_point(from_V.row(vi), &ni);
		//Assert(found_nearest);
		if (found_nearest)
			VQ(vi) = (from_V.row(vi) - ni.p.transpose()).norm() / from_box.diagonal().norm();
		else
			VQ(vi) = 0;
	};

	std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), f);

	//for (int i = 0; i < (int)from_V.rows(); ++i) {
	//	//Vector3 p = bvh.nearest_point(from_V.row(i), [&](int fi) {
	//	//	return compute_face_normal(fi, to_V, to_F).normalized().dot(from_VN.row(i)) > 0;
	//	//});

	//	if (VB[i])
	//		VQ(i) = 0;
	//	else {
	//		NearestInfo ni;
	//		bool found_nearest = bvh.nearest_point(from_V.row(i), &ni, );
	//		Assert(found_nearest);
	//		//Vector3 p = bvh.nearest_point(from_V.row(i));
	//		VQ(i) = -(from_V.row(i) - ni.p.transpose()).norm();
	//	}
	//}
}

