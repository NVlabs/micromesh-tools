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
#include "micro.h"

#include <string>

struct Bary;

struct GLTFWriteInfo {
	const MatrixXi* _F = nullptr;
	const MatrixX* _V  = nullptr;
	const MatrixX* _UV = nullptr;
	const MatrixX* _VD = nullptr;
	const MatrixX* _VN = nullptr;
	const MatrixX* _VT = nullptr;
	const MatrixX* _VC = nullptr;
	const VectorXu8* _subdivision_bits = nullptr;
	const Bary* _bary = nullptr;

	std::string _color_texture;
	std::string _normal_texture;

	bool has_faces() const { return _F; }
	bool has_vertices() const { return _V; }
	bool has_uvs() const { return _UV; }
	bool has_directions() const { return _VD; }
	bool has_normals() const { return _VN; }
	bool has_tangents() const { return _VT; }
	bool has_colors() const { return _VC; }
	bool has_subdivision_bits() const { return _subdivision_bits; }
	bool has_bary() const { return _bary; }
	bool has_color_texture() const { return !_color_texture.empty(); }
	bool has_normal_texture() const { return !_normal_texture.empty(); }

	GLTFWriteInfo& write_faces(const MatrixXi* F) { _F = F; return *this; }
	GLTFWriteInfo& write_vertices(const MatrixX* V)    { _V = V;   return *this; }
	GLTFWriteInfo& write_uvs(const MatrixX* UV)        { _UV = UV; return *this; }
	GLTFWriteInfo& write_directions(const MatrixX* VD) { _VD = VD; return *this; }
	GLTFWriteInfo& write_normals(const MatrixX* VN)    { _VN = VN; return *this; }
	GLTFWriteInfo& write_tangents(const MatrixX* VT)   { _VT = VT; return *this; }
	GLTFWriteInfo& write_colors(const MatrixX* VC)    { _VC = VC; return *this; }
	GLTFWriteInfo& write_subdivision_bits(const VectorXu8* subdivision_bits) { _subdivision_bits = subdivision_bits; return *this; }
	GLTFWriteInfo& write_bary(const Bary* bary) { _bary = bary; return *this; }
	GLTFWriteInfo& write_color_texture(const std::string& color_texture) { _color_texture = color_texture; return *this; }
	GLTFWriteInfo& write_normal_texture(const std::string& normal_texture) { _normal_texture = normal_texture; return *this; }

	const MatrixXi& F() const { return *_F; }
	const MatrixX& V()  const { return *_V; }
	const MatrixX& UV() const { return *_UV; }
	const MatrixX& VD() const { return *_VD; }
	const MatrixX& VN() const { return *_VN; }
	const MatrixX& VT() const { return *_VT; }
	const MatrixX& VC() const { return *_VC; }
	const VectorXu8& subdivision_bits() const { return *_subdivision_bits; }
	const Bary& bary() const { return *_bary; }
	const std::string& color_texture() const { return _color_texture; }
	const std::string& normal_texture() const { return _normal_texture; }
};

struct GLTFReadInfo {
	MatrixXi _F;
	MatrixX _V;
	MatrixX _UV;
	MatrixX _VD;
	MatrixX _VN;
	MatrixX _VT;
	//MatrixX _VC;
	VectorXu8 _subdivisions;
	VectorXu8 _topology_flags;

	SubdivisionMesh _umesh;

	std::vector<std::string> color_textures;
	std::vector<std::string> normal_textures;

	bool has_faces() const { return _F.size() > 0; }
	bool has_vertices() const { return _V.size() > 0; }
	bool has_uvs() const { return _UV.size() > 0; }
	bool has_directions() const { return _VD.size() > 0; }
	bool has_normals() const { return _VN.size() > 0; }
	bool has_tangents() const { return _VT.size() > 0; }
	//bool has_colors() const { return _VC.size() > 0; }
	bool has_subdivisions() const { return _subdivisions.size() > 0; }
	bool has_topology_flags() const { return _topology_flags.size() > 0; }
	bool has_subdivision_mesh() const { return _umesh.faces.size() > 0; }
	bool has_color_textures() const { return color_textures.size() > 0; }
	bool has_normal_textures() const { return normal_textures.size() > 0; }

	MatrixXi get_faces() { return std::move(_F); }
	MatrixX get_vertices() { return std::move(_V); }
	MatrixX get_uvs() { return std::move(_UV); }
	MatrixX get_directions() { return std::move(_VD); }
	MatrixX get_normals() { return std::move(_VN); }
	MatrixX get_tangents() { return std::move(_VT); }
	//MatrixX get_colors() { return std::move(_VC); }
	VectorXu8 get_subdivisions() { return std::move(_subdivisions); }
	VectorXu8 get_topology_flags() { return std::move(_topology_flags); }
	SubdivisionMesh get_subdivision_mesh() { return std::move(_umesh); }
};

bool read_gltf(const std::string& filename, GLTFReadInfo& read_info);
bool write_gltf(const std::string& name, const GLTFWriteInfo& write_info);

