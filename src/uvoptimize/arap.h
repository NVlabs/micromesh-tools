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

struct SubdivisionMesh;

// ARAP parametrization on a micromesh
// at each iteration with fixed rotation matrices (global step) solves
//    min E(U) = E_{ARAP} + \delta |U - \bar{U}|
// where the second term is a regularization to ensure the matrix is invertible
//   \bar{U} is the parametrization at the previous iteration
//   \delta is a small scalar (e.g. 1e-4)
// 
// TODO there is a lot of room for optimizations and caching
struct ARAP {
	typedef Eigen::Matrix<Scalar, 3, 2> Matrix32;

	struct _SVDCache {
		Matrix2 U;
		Matrix2 V;
		Vector2 sigma;
	};

	struct Result {
		bool success;
		Scalar energy_value;
	};

	// list of micro-face cotangents
	// one uF x 3 matrix for each base face, where uF is the number of micro-faces
	std::vector<MatrixX> _cotangents;

	const SubdivisionMesh& _micromesh;
	const MatrixX& _base_V;
	const MatrixXi& _base_F;

	// coefficient for the regularization term sum_i |u_i - \bar{u}_i|^2
	Scalar _delta = 1e-4;

	// convergence tolerance - stop iterations if abs(energy_pre - energy_curr) is less than
	// this value
	Scalar _convergence_tolerance = 1e-5;

	mutable std::vector<std::vector<_SVDCache>> _svd_cache;

	ARAP(const SubdivisionMesh& micromesh, const MatrixX& base_V, const MatrixXi& base_F)
		: _micromesh(micromesh), _base_V(base_V), _base_F(base_F)
	{
	}

	Matrix3 _get_face_vertices(const MatrixX& V, const MatrixX& VD, const MatrixXi& F, int fi) const;

	Scalar _compute_energy(const MatrixX& UV) const;

	void _compute_cotangents();
	SparseMatrix _compute_system_matrix() const;
	std::vector<std::vector<Matrix2>> _compute_rotations(const MatrixX& UV) const;
	MatrixX _compute_rhs(const std::vector<std::vector<Matrix2>>& rotations, const MatrixX& UV) const;

	ARAP::Result solve(MatrixX& UV, int max_iter);

	void _require_svd_cache(const MatrixX& UV) const;
	void _clear_svd_cache() const;
};

