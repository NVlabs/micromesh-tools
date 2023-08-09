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
#include "aabb.h"
#include "adjacency.h"

#include <set>
#include <vector>

struct PackingInfo {
	Scalar object_scale;
	std::vector<Vector2i> placements;
};

struct Chart {
	std::set<int> faces;
	Box2 box;
};

std::vector<Chart> extract_charts(const MatrixX& UV, const MatrixXi& F, const VFAdjacency& VF);
MatrixX pack_charts(const std::vector<Chart>& charts, int grid_size, const MatrixX& UV, const MatrixXi& F, const VFAdjacency& VF);

