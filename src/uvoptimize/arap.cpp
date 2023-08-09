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

#include "arap.h"
#include "micro.h"
#include "tangent.h"
#include "utils.h"

#include <vector>
#include <algorithm>
#include <execution>
#include <atomic>

static void isometric_2d_projection(const Vector3& p10, const Vector3& p20, Vector2* u10, Vector2* u20)
{
	Scalar p10n = p10.norm();
	Scalar p20n = p20.norm();
	if (p10n == 0 || p20n == 0) {
		if (p10n == 0) p10n = 1e-6;
		if (p20n == 0) p20n = 1e-6;
	}

	Scalar theta = vector_angle(p10, p20);
	if (theta <= 0)
		theta = 1e-3;
	if (theta >= M_PI)
		theta = M_PI - 1e-3;

	(*u10)(0) = p10n;
	(*u10)(1) = 0;
	(*u20)(0) = p20n * std::cos(theta);
	(*u20)(1) = p20n * std::sin(theta);
}

static Matrix2 compute_jacobian(const Vector2 x10, const Vector2& x20, const Vector2& u10, const Vector2& u20)
{
	Matrix2 G;
	Matrix2 F;

	F.col(0) = x10;
	F.col(1) = x20;
	G.col(0) = u10;
	G.col(1) = u20;

	return G * F.inverse();
}

Matrix3 ARAP::_get_face_vertices(const MatrixX& V, const MatrixX& VD, const MatrixXi& F, int fi) const
{
	Matrix3 M;
	for (int i = 0; i < 3; ++i)
		M.row(i) = V.row(F(fi, i)) + VD.row(F(fi, i));
	return M;
}

Scalar ARAP::_compute_energy(const MatrixX& UV) const
{
	_require_svd_cache(UV);

	std::atomic<Scalar> e = 0;
	std::atomic<Scalar> area = 0;

	std::vector<int> tasks(_micromesh.faces.size());
	std::iota(tasks.begin(), tasks.end(), 0);

	// scan base faces
	auto worker = [&](int i) -> void {
		Assert(i < _micromesh.faces.size());
		const SubdivisionTri& st = _micromesh.faces[i];

		// scan microfaces
		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			Matrix3 microV = _get_face_vertices(st.V, st.VD, st.F, ufi);
			Vector2 sigma = _svd_cache[i][ufi].sigma;
			Scalar area_f = compute_area_vector(microV.row(0), microV.row(1), microV.row(2)).norm();
			area += area_f;
			e += area_f * (std::pow(sigma(0) - 1.0, 2.0) + std::pow(sigma(1) - 1.0, 2.0));
		}
	};

	std::for_each(std::execution::par_unseq, tasks.begin(), tasks.end(), worker);

	return e / area;
}

void ARAP::_compute_cotangents()
{
	_cotangents.clear();

	for (const SubdivisionTri& st : _micromesh.faces) {
		MatrixX C(st.F.rows(), 3);
		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			Matrix3 microV = _get_face_vertices(st.V, st.VD, st.F, ufi);
			for (int i = 0; i < 3; ++i) {
				Scalar angle = vector_angle(microV.row((i + 1) % 3) - microV.row(i), microV.row((i + 2) % 3) - microV.row(i));
				C(ufi, i) = cot(clamp(angle, 1e-3, M_PI - 1e-3));
			}
		}
		_cotangents.push_back(C);
	}
}

SparseMatrix ARAP::_compute_system_matrix() const
{
	typedef Eigen::Triplet<Scalar> Triplet;

	int vn = _base_V.rows();
	
	SparseMatrix A;
	A.resize(vn, vn);
	A.setZero();

	std::vector<Triplet> triplets;

	for (const SubdivisionTri& st : _micromesh.faces) {
		Vector3i base_vi(_base_F(st.base_fi, 0), _base_F(st.base_fi, 1), _base_F(st.base_fi, 2));
		const MatrixX& C = _cotangents[st.base_fi];

		BarycentricGrid grid(st.subdivision_level());

		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			// get barycentric weights of microvertices
			Matrix3 W;
			for (int i = 0; i < 3; ++i)
				W.row(i) = grid.barycentric_coord(st.F(ufi, i));

			// accumulate gradient terms over micro half-edges
			for (int i = 0; i < 3; ++i) {
				int j = (i + 1) % 3;
				int k = (i + 2) % 3;

				Scalar cot_k = C(ufi, k);

				Assert(std::isfinite(cot_k));

				// accumulate over base indices
				for (int b = 0; b < 3; ++b) {
					triplets.push_back(Triplet(base_vi(b), base_vi(0), cot_k * (W(i, 0) - W(j, 0)) * (W(i, b) - W(j, b))));
					triplets.push_back(Triplet(base_vi(b), base_vi(1), cot_k * (W(i, 1) - W(j, 1)) * (W(i, b) - W(j, b))));
					triplets.push_back(Triplet(base_vi(b), base_vi(2), cot_k * (W(i, 2) - W(j, 2)) * (W(i, b) - W(j, b))));
				}
			}
		}
	}

	// add regularization term 
	for (int i = 0; i < vn; ++i) {
		triplets.push_back(Triplet(i, i, 2 * _delta));
	}

	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();

	return A;
}

std::vector<std::vector<Matrix2>> ARAP::_compute_rotations(const MatrixX& UV) const
{
	_require_svd_cache(UV);

	std::vector<std::vector<Matrix2>> rotations(_micromesh.faces.size());

	std::vector<int> tasks(_micromesh.faces.size());
	std::iota(tasks.begin(), tasks.end(), 0);

	auto worker = [&](const int i) -> void {
		Assert(i < _micromesh.faces.size());
		const SubdivisionTri& st = _micromesh.faces[i];

		std::vector<Matrix2> micro_rotations;

		// scan microfaces
		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			// compute isometric projection of the displaced microface
			// extract the rotation

			Matrix2 U = _svd_cache[i][ufi].U;
			Matrix2 V = _svd_cache[i][ufi].V;
			Matrix2 R = U * V.transpose();
			if (R.determinant() < 0) {
				U.col(U.cols() - 1) *= -1;
				R = U * V.transpose();
			}

			rotations[i].push_back(R);
		}
	};

	std::for_each(std::execution::par_unseq, tasks.begin(), tasks.end(), worker);

	return rotations;
}

MatrixX ARAP::_compute_rhs(const std::vector<std::vector<Matrix2>>& rotations, const MatrixX& UV) const
{
	MatrixX RHS = MatrixX::Constant(_base_V.rows(), 2, 0);

	std::vector<std::atomic<Scalar>> rhs_x(_base_V.rows());
	std::vector<std::atomic<Scalar>> rhs_y(_base_V.rows());

	std::vector<int> tasks(_micromesh.faces.size());
	std::iota(tasks.begin(), tasks.end(), 0);

	auto worker = [&](int i) -> void {
		Assert(i < _micromesh.faces.size());
		const SubdivisionTri& st = _micromesh.faces[i];

		Vector3i base_vi(_base_F(st.base_fi, 0), _base_F(st.base_fi, 1), _base_F(st.base_fi, 2));
		const MatrixX& C = _cotangents[st.base_fi];

		BarycentricGrid grid(st.subdivision_level());

		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			// get barycentric weights of microvertices
			Matrix3 W;
			for (int i = 0; i < 3; ++i)
				W.row(i) = grid.barycentric_coord(st.F(ufi, i));

			// get isometric 2d coordinates
			Matrix3 microV = _get_face_vertices(st.V, st.VD, st.F, ufi);
			Vector2 x10, x20;
			isometric_2d_projection(microV.row(1) - microV.row(0), microV.row(2) - microV.row(0), &x10, &x20);

			Matrix32 X;
			X.row(0).setZero();
			X.row(1) = X.row(0) + x10.transpose();
			X.row(2) = X.row(0) + x20.transpose();

			// accumulate gradient rhs terms over micro half-edges
			for (int i = 0; i < 3; ++i) {
				int j = (i + 1) % 3;
				int k = (i + 2) % 3;

				Scalar cot_k = C(ufi, k);

				Assert(std::isfinite(cot_k));

				// compute gradient term
				Vector2 x_ij = X.row(i) - X.row(j);
				Vector2 rhs_term = cot_k * (-rotations[st.base_fi][ufi] * x_ij);

				// accumulate weighted gradient term over base indices
				for (int b = 0; b < 3; ++b) {
					rhs_x[base_vi(b)] += (rhs_term.x() * (W(i, b) - W(j, b)));
					rhs_y[base_vi(b)] += (rhs_term.y() * (W(i, b) - W(j, b)));
				}
			}
		}
	};

	std::for_each(std::execution::par_unseq, tasks.begin(), tasks.end(), worker);

	// fill RHS vector and add regularization term
	for (int i = 0; i < RHS.rows(); ++i)
		RHS.row(i) = RowVector2(rhs_x[i], rhs_y[i]) - 2 * _delta * UV.row(i);

	return RHS;
}

ARAP::Result ARAP::solve(MatrixX& UV, int max_iter)
{
	Assert(UV.rows() == _base_V.rows());
	Assert(UV.cols() == 2);

	Timer t;

	_compute_cotangents();
	SparseMatrix A = _compute_system_matrix();

	//Eigen::SparseLU<SparseMatrix> solver;
	Eigen::SimplicialLDLT<SparseMatrix> solver;
	solver.compute(A);

	//solver.analyzePattern(A);
	//solver.factorize(A);

	if (solver.info() != Eigen::Success) {
		std::cerr << "ARAP: Matrix factorization failed " << solver.info() << std::endl;
		return Result{ .success = false, .energy_value = Infinity };
	}

	Assert(max_iter >= 0);

	Scalar energy = _compute_energy(UV);

	std::cout << "ARAP: initial energy is " << energy << std::endl;

	bool converged = false;

	int ring_buffer_size = 5;
	std::vector<Scalar> energy_history(ring_buffer_size, Infinity);
	int buffer_index = 0;

	auto push_energy = [&](Scalar val) -> void {
		energy_history[buffer_index++] = val;
		buffer_index %= ring_buffer_size;
	};

	auto delta_energy = [&]() -> Scalar {
		return std::abs(energy_history[(buffer_index + ring_buffer_size - 1) % ring_buffer_size] - energy_history[buffer_index]);
	};

	push_energy(energy);

	Timer t_iter;

	int i = 0;
	while (i < max_iter && !converged) {
		std::vector<std::vector<Matrix2>> rotations = _compute_rotations(UV);
		MatrixX RHS = _compute_rhs(rotations, UV);

		MatrixX UV_iter = solver.solve(RHS);

		if (solver.info() != Eigen::Success) {
			return Result{ .success = false, .energy_value = Infinity };
		}

		Scalar energy_iter = _compute_energy(UV_iter);

		push_energy(energy_iter);
		if (delta_energy() < _convergence_tolerance)
			converged = true;

		UV = UV_iter;
		energy = energy_iter;

		_clear_svd_cache();

		std::cout << "ARAP: iteration " << i << " energy = " << energy_iter << ", time = " << t_iter.time_since_last_check() << std::endl;

		++i;
	}
	
	std::cout << "ARAP: final energy after " << i << " iterations is " << energy << std::endl;
	std::cout << "ARAP: solution took " << t.time_elapsed() << " seconds" << std::endl;

	return Result{ .success = true, .energy_value = energy };
}

void ARAP::_require_svd_cache(const MatrixX& UV) const
{
	if (!_svd_cache.empty())
		return;

	_svd_cache.resize(_micromesh.faces.size());

	std::vector<int> tasks(_micromesh.faces.size());
	std::iota(tasks.begin(), tasks.end(), 0);

	auto worker = [&](const int i) -> void {
		Assert(i < _micromesh.faces.size());
		const SubdivisionTri& st = _micromesh.faces[i];

		Matrix32 bUV;
		for (int i = 0; i < 3; ++i)
			bUV.row(i) = UV.row(_base_F(st.base_fi, i));

		BarycentricGrid grid(st.subdivision_level());

		std::vector<Matrix2> micro_rotations;

		// scan microfaces
		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			// compute isometric projection of the displaced microface
			Matrix3 microV = _get_face_vertices(st.V, st.VD, st.F, ufi);
			Vector2 x10, x20;
			isometric_2d_projection(microV.row(1) - microV.row(0), microV.row(2) - microV.row(0), &x10, &x20);

			// compute the interpolated UVs of the microvertices
			Vector3 w0 = grid.barycentric_coord(st.F(ufi, 0));
			Vector3 w1 = grid.barycentric_coord(st.F(ufi, 1));
			Vector3 w2 = grid.barycentric_coord(st.F(ufi, 2));

			Vector2 u10 = (w1 - w0).transpose() * bUV;
			Vector2 u20 = (w2 - w0).transpose() * bUV;

			// compute the jacobian of the parametrization
			Matrix2 Jf = compute_jacobian(x10, x20, u10, u20);

			_SVDCache cache;

			// decompose and cache
			Eigen::JacobiSVD<Matrix2, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
			svd.compute(Jf);
			cache.U = svd.matrixU();
			cache.V = svd.matrixV();
			cache.sigma = svd.singularValues();

			_svd_cache[i].push_back(cache);
		}
	};

	std::for_each(std::execution::par_unseq, tasks.begin(), tasks.end(), worker);
}

void ARAP::_clear_svd_cache() const
{
	_svd_cache.clear();
}

