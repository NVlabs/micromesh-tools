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

inline Vector3 oriented_direction(const Vector3& reference_direction_vector, const Vector3& direction_vector)
{
	return (reference_direction_vector.dot(direction_vector) > 0) ? direction_vector : -direction_vector;
};

void compute_direction_field(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, MatrixX& D, VectorX& DW);
void smooth_direction_field(const MatrixXi& F, const VFAdjacency& VF, MatrixX& D, VectorX& DW);
void update_direction_field_on_collapse(const Edge& e, MatrixX& D, VectorX& DW);
