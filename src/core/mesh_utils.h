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

#include "space.h"
#include "adjacency.h"

Scalar average_edge(const MatrixX& V, const MatrixXi& F);
Scalar average_area(const MatrixX& V, const MatrixXi& F);

// if quad subdivides as 0-1-2, 0-2-3
Scalar face_area(const MatrixX& V, const VectorXi& f);

// Measures the approximate displacement volume. The displacement volume is the sum of the
// approximate volume of each prismoid, obtained by decomposing it into 3 tetras.
Scalar mesh_displacement_volume(const MatrixX& V, const MatrixX& VD, const MatrixXi& F);

// Computes a vector of per-vertex voronoi areas, which can be used to
// scale differential operators according to the surface metric, as detailed
// in the paper
// Mark Meyer, Mathieu Desbrun, Peter Schroder and Alan H. Barr
// ``Discrete differential-geometry operators for triangulated 2-Manifolds''
VectorX compute_voronoi_vertex_areas(const MatrixX& V, const MatrixXi& F);

// Returns a vector of face aspect ratios
VectorX compute_face_aspect_ratios(const MatrixX& V, const MatrixXi& F);

// Updates the aspect ratios of the faces around the designated vertex index
void update_face_aspect_ratios_around_vertex(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, int vi, VectorX& FR);

// return a vector of linear indices
std::vector<int> vector_of_indices(long n);

