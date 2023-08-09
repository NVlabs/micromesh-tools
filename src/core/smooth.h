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

void flip_guarded_laplacian_smooth(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB, const MatrixX& FN);
void flip_guarded_anisotropic_smoothing(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB, const MatrixX& FN,
	int iterations, Scalar border_error_scale, Scalar weight);

// Laplacian smoothing of per-vertex data - Matrix version (e.g. Vertex coordinates)
void laplacian_smooth(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB);

// Laplacian smoothing of per-vertex data - Vector version (e.g. Scalar field)
void laplacian_smooth(VectorX& X, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB);

// Laplacian smoothing of vertex coordinates with tangent-plane reprojection (tangent smoothing)
void tangent_plane_laplacian_smooth(MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const VectorXu8& VB);

