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

#include "curvature.h"
#include "adjacency.h"
#include "mesh_utils.h"
#include "utils.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace detail {
	Scalar compute_smooting_weight(Scalar kh, Scalar k1, Scalar k2, Scalar T)
	{
		Scalar absk1 = std::abs(k1);
		Scalar absk2 = std::abs(k2);
		Scalar abskh = std::abs(kh);

		// the -0.1 threshold is suggested in the paper and I'm sticking with it...
		if (absk1 <= T && absk2 <= T)
			return 1;
		else if (absk1 > T && absk2 > T && k1 * k2 > 0)
			return 0;
		else if (absk1 <= absk2 && absk1 <= abskh)
			return std::max(Scalar(-0.1), k1 / kh);
		else if (absk2 <= absk2 && absk2 <= abskh)
			return std::max(Scalar(-0.1), k2 / kh);
		else if (abskh <= absk1 && abskh <= absk2)
			return 1;
		else
			Assert(0 && "compute_smoothing_weight(): Unreachable");
	}

	MatrixX mean_curvature_flow_smoothing(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, Scalar ldt, bool anisotropic, Scalar T_val)
	{
		Timer t;

		std::cout << "Computing voronoi areas...";
		VectorX voronoi_areas = compute_voronoi_vertex_areas(V, F);
		std::cout << " done." << std::endl;
		std::cout << "Computing Laplacian matrix...";
		SparseMatrix C = compute_cotangent_laplacian(V, F);
		std::cout << " done." << std::endl;

		//Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> D = voronoi_areas.unaryExpr([](const Scalar x) { return 1.0 / (2.0 * x); }).asDiagonal();
		//SparseMatrix L = D * C;
	
		SparseMatrix L =
			SparseMatrix(
				voronoi_areas
				.unaryExpr([](const Scalar x) -> Scalar { return 1.0 / (2.0 * x); })
				.asDiagonal())
			* C;

		SparseMatrix M(V.rows(), V.rows());
		M.setIdentity();
	
		if (!anisotropic) {
			// compute isotropic implicit fairing by integrating curvature flow as is
			M += ldt * 0.5 * L;
		}
		else {
			// compute anisotropic smoothing weights for the flow integration
			VectorX KG;
			VectorX KH;
			VectorX K1;
			VectorX K2;
			MatrixX K;
			std::cout << "Computing gaussian curvature...";
			compute_gaussian_curvature(V, F, VF, voronoi_areas, KG);
			std::cout << " " << KG.minCoeff() << " " << KG.maxCoeff() << " ";
			std::cout << " done." << std::endl;
			std::cout << "Computing mean curvature normal...";
			compute_mean_curvature_normal(V, L, K);
			std::cout << " done." << std::endl;
			std::cout << "Computing mean curvature...";
			compute_mean_curvature(K, KH);
			std::cout << " " << KH.minCoeff() << " " << KH.maxCoeff() << " ";
			std::cout << " done." << std::endl;
			std::cout << "Computing principal curvatures...";
			compute_principal_curvatures(KH, KG, K1, K2);
			std::cout << " done." << std::endl;

			std::cout << "Computing smoothing weights...";
			VectorX W = VectorX::Constant(V.rows(), 0);
			for (int i = 0; i < (int)V.rows(); ++i) {
				W(i) = compute_smooting_weight(KH(i), K1(i), K2(i), T_val);
			}
			std::cout << " done." << std::endl;

			//M += ldt * 0.5 * Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic>(W.asDiagonal()) * L;
			M += ldt * 0.5 * SparseMatrix(W.asDiagonal()) * L;
		}

		std::cout << "M matrix properties: " << M.rows() << "x" << M.cols() << " nnz = " << M.nonZeros() << std::endl;

		M.makeCompressed();
		std::cout << "Time spent setting up linear system: " << t.time_elapsed() << " secs" << std::endl;

		Eigen::SparseLU<SparseMatrix> LU;
		LU.compute(M);

		std::cout << "Time spent factorizing linear system: " << t.time_since_last_check() << " secs" << std::endl;

		MatrixX Vnew = LU.solve(V);
		Assert(LU.info() == Eigen::Success);

		std::cout << "Time spent solving linear system: " << t.time_since_last_check() << " secs" << std::endl;

		std::cout << "Total time: " << t.time_elapsed() << " secs" << std::endl;

		return Vnew;
	}
}

void compute_gaussian_curvature(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorX& voronoi_areas, VectorX& KG)
{
	int deg = (int)F.cols();

	KG = VectorX::Constant(V.rows(), 2.0 * M_PI);

	for (int fi = 0; fi < F.rows(); ++fi) {
		if (F(fi, 0) != INVALID_INDEX) {
			for (int j = 0; j < deg; ++j) {
				int j1 = (j + 1) % deg;
				int j2 = (j + 2) % deg;
				Vector3 v10 = V.row(F(fi, j1)) - V.row(F(fi, j));
				Vector3 v20 = V.row(F(fi, j2)) - V.row(F(fi, j));
				KG(F(fi, j)) -= vector_angle(v10, v20);
			}
		}
	}

	for (int vi = 0; vi < V.rows(); ++vi) {
		if (voronoi_areas(vi) > 0)
			KG(vi) /= voronoi_areas(vi);
		else
			KG(vi) = 0;
	}
}

SparseMatrix compute_cotangent_laplacian(const MatrixX& V, const MatrixXi& F)
{
	typedef Eigen::Triplet<Scalar> Triplet;
	std::vector<Triplet> coefficients;
	coefficients.reserve(4 * 3 * F.rows());
	
	int deg = (int)F.cols();
	for (int i = 0; i < (int)F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			for (int j = 0; j < deg; ++j) {
				int vj0 = F(i, j);
				int vj1 = F(i, (j + 1) % deg);
				int vj2 = F(i, (j + 2) % deg);
				Scalar cot_j = cot(vector_angle(V.row(vj1) - V.row(vj0), V.row(vj2) - V.row(vj0)));
				Assert(std::isfinite(cot_j));
				coefficients.push_back(Triplet(vj1, vj1, cot_j));
				coefficients.push_back(Triplet(vj1, vj2, -cot_j));
				coefficients.push_back(Triplet(vj2, vj2, cot_j));
				coefficients.push_back(Triplet(vj2, vj1, -cot_j));
			}
		}
	}

	SparseMatrix L(V.rows(), V.rows());
	L.setFromTriplets(coefficients.begin(), coefficients.end());
	return L;
}

void compute_mean_curvature_normal(const MatrixX& V, const MatrixXi& F, MatrixX& K)
{
	VectorX voronoi_areas = compute_voronoi_vertex_areas(V, F);
	SparseMatrix L = voronoi_areas
		.unaryExpr([](const Scalar x) -> Scalar { return 1.0 / (2.0 * x); })
		.asDiagonal()
		* compute_cotangent_laplacian(V, F);

	compute_mean_curvature_normal(V, L, K);
}

void compute_mean_curvature_normal(const MatrixX& V, const SparseMatrix& L, MatrixX& K)
{
	K = L * V;
}

// returns the mean curvatre computed from the mean curvature normal
// (half the magnitude of the mean curvature normal)
void compute_mean_curvature(const MatrixX& K, VectorX& KH)
{
	Assert(K.cols() == 3);
	KH = K.rowwise().norm() * 0.5;
}

// returns the principal curvatures values
void compute_principal_curvatures(const VectorX& KH, const VectorX& KG, VectorX& K1, VectorX& K2)
{
	Assert(KH.rows() == KG.rows());
	Assert(KH.cols() == KG.cols());

	K1.resizeLike(KH);
	K2.resizeLike(KH);

	for (int i = 0; i < (int)KH.rows(); ++i) {
		Scalar delta = std::max(KH(i) * KH(i) - KG(i), Scalar(0));
		K1(i) = KH(i) + std::sqrt(delta);
		K2(i) = KH(i) - std::sqrt(delta);
	}
}

MatrixX mean_curvature_flow_smoothing_isotropic(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, Scalar ldt)
{
	return detail::mean_curvature_flow_smoothing(V, F, VF, ldt, false, 0);
}

MatrixX mean_curvature_flow_smoothing_anisotropic(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, Scalar ldt, Scalar T_val)
{
	return detail::mean_curvature_flow_smoothing(V, F, VF, ldt, true, T_val);
}

