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

#include <string>


struct SubdivisionMesh;

bool read_obj(const std::string& filename, MatrixX& V, MatrixXi& F, MatrixX& VT, MatrixXi& FT, MatrixX& VN, MatrixXi& FN);
bool read_obj(const std::string& filename, MatrixX& V, MatrixXi& F);
void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F);
void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& VN);
void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& VT, const MatrixXi& FT);

void write_obj(const std::string& filename, const SubdivisionMesh& umesh, Scalar displacement_factor);
void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& VT, const MatrixXi& FT, const MatrixX& VN, const MatrixXi& FN);

void write_obj_lines(const std::string& filename, const std::vector<MatrixX>& Vs, const std::vector<std::vector<int>>& Ls);

bool write_ply(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& C);

bool read_stl(const std::string& filename, MatrixX& V, MatrixXi& F);

