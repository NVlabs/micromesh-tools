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
#include "adjacency.h"

// References
// [1] Mark Meyer, Mathieu Desbrun, Peter Schroder and Alan H. Barr
//     Discrete differential-geometry operators for triangulated 2-manifolds
// 
// [2] Mathieu Desbrun, Mark Meyer, Peter Schroder and Alan H. Barr
//     Implicit fairing of irregular meshes using diffusion and curvature flow

// Compute the cotangent-weighted mesh Laplacian as a sparse linear operator
// The matrix rows are NOT area-weighted (i.e. no metric scaling)
SparseMatrix compute_cotangent_laplacian(const MatrixX& V, const MatrixXi& F);

// Compute per-vertex gaussian curvature (angle defect)
void compute_gaussian_curvature(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorX& voronoi_area, VectorX& KG);

// Compute the per-vertex mean curvature normal vectors, as detailed in [1],
// if not provided as argument, this function computes the metric-scaled mesh Laplacian
void compute_mean_curvature_normal(const MatrixX& V, const MatrixXi& F, MatrixX& K);
void compute_mean_curvature_normal(const MatrixX& V, const SparseMatrix& L, MatrixX& K);

// Compute per-vertex mean curvatures from mean curvature normals [1]
void compute_mean_curvature(const MatrixX& K, VectorX& KH);

// Compute per-vertex principal curvature values from mean curvature vectors and Gaussian curvatures
void compute_principal_curvatures(const VectorX& KH, const VectorX& KG, VectorX& K1, VectorX& K2);

// Implicit isotropic fairing by mean curvature flow integration [2]
// ldt is the time-step integration and controls the tradeoff between the
// amount and accuracy of the smoothing
// Returns the new vertex positions computed by solving the following linear system
//    (I - ldt * L) V_new = V_old
// Where L is the metric-scaled cotangent Laplacian which is used to compute the mean curvature flow
MatrixX mean_curvature_flow_smoothing_isotropic(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, Scalar ldt);

// Implicit anisotropic fairing by reweighted mean curvature flow intergration [1]
// ldt is the time-step integration and controls the tradeoff between the
// amount and accuracy of the smoothing
// Returns the new vertex positions computed by solving the following linear system
//    (I - ldt * W * L) V_new = V_old
// Where W is the diagonal matrix of per-vertex anisotropic smoothing weights and
// L is the metric-scaled cotangent Laplacian which is used to compute the mean curvature flow
// T_val is a parameter controlling the sharpness of edges and is used to adjust
// the smoothing weights (it is related to the magnitude of the principal curvatures
// at each vertex)
MatrixX mean_curvature_flow_smoothing_anisotropic(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, Scalar ldt, Scalar T_val);


