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

// computes the approximate 'most normal' normal by using local search over geodesic paths on the unit sphere
MatrixX compute_displacement_directions_from_convex_cone(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF);

// Returns a pair <optimal_visibility_direction, visibility>
// The optimal visibility direction is the vector that maximizes the minimum dot product
// with any of the supplied directions vectors (cfr. ``On the 'most normal' normal``, Aubrey and Lohner) 
// The visibility is the minmax value (dot product of the worst normal with the
// optimal visibility direction)
// Directions need not to be normalized, and vectors with near-zero length are ignored, so area vectors
// can be passed, and null vectors from zero-area faces are ignored during the computation
// The maximum length for considering a vector null is controlled with the null_tolerance parameter
// 
// This implementation adapts Welzl's algorithm for finding a minimum enclosing disk on spherical points,
// and is guaranteed to find the optimal solution ***only*** if the minmax value is (strictly) positive
std::pair<Vector3, Scalar> compute_positive_visibility_from_directions(const std::vector<Vector3>& directions, Scalar null_tolerance = 1e-12);

// Returns per-vertex optimal visibility directions and visibility values
// For negative visibility vertices, it is not guaranteed to return the optimal solution, but in that
// case things are really bad for micromeshes
std::pair<MatrixX, VectorX> compute_optimal_visibility_directions(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF);

