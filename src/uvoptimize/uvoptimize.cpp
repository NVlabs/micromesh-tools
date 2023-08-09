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

#include "micro.h"
#include "tangent.h"
#include "bvh.h"
#include "mesh_io_gltf.h"
#include "mesh_io_bary.h"

#include "adjacency.h"
#include "arap.h"
#include "rectangle_packer.h"

#include "utils.h"

#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cerr << "Usage: " << fs::path(argv[0]).filename().string() << " MICROMESH" << std::endl;
		std::cerr << "    MICROMESH The micromesh glTF file" << std::endl;
		std::cerr << "This tool generates the following files:" << std::endl;
		std::cerr << "    uvbase_base_MICROMESH The base mesh with optimized UVs (reference: base geometry)" << std::endl;
		std::cerr << "    uvbase_MICROMESH A micromesh glTF file with optimized UVs (reference: base geometry)" << std::endl;
		std::cerr << "    uvopt_MICROMESH A micromesh glTF file with optimized UVs (reference: displaced microgeometry)" << std::endl;
		return -1;
	}

	fs::path input_mesh(argv[1]);

	GLTFReadInfo read_micromesh;
	if (!read_gltf(input_mesh.string(), read_micromesh)) {
		std::cerr << "Error reading gltf file " << argv[2] << std::endl;
		return -1;
	}

	Assert(read_micromesh.has_vertices());
	Assert(read_micromesh.has_faces());
	Assert(read_micromesh.has_uvs());
	Assert(read_micromesh.has_normals());
	Assert(read_micromesh.has_tangents());
	Assert(read_micromesh.has_directions());
	Assert(read_micromesh.has_subdivisions());
	Assert(read_micromesh.has_topology_flags());
	Assert(read_micromesh.has_subdivision_mesh());

	MatrixX V = read_micromesh.get_vertices();
	MatrixXi F = read_micromesh.get_faces();
	MatrixX UV = read_micromesh.get_uvs();
	MatrixX VN = read_micromesh.get_normals();
	MatrixX VT = read_micromesh.get_tangents();
	MatrixX VD = read_micromesh.get_directions();
	VectorXu8 S = read_micromesh.get_subdivisions();
	VectorXu8 E = read_micromesh.get_topology_flags();
	SubdivisionMesh umesh = read_micromesh.get_subdivision_mesh();

	// pack subdivisions and edge flags
	VectorXu8 SB = S;
	for (int i = 0; i < SB.size(); ++i)
		SB(i) = (S(i) << 3) | E(i);

	// generate a non-subdivided micromesh (used as reference for the 'base ARAP')
	SubdivisionMesh base;
	base.compute_mesh_structure(V, F, 0);
	// setup dummy micro-displacements
	std::vector<std::vector<Scalar>> dummy_displacements;
	for (int i = 0; i < F.rows(); ++i) {
		// each base face has only 3 zero displacements
		dummy_displacements.push_back({ 0, 0, 0 });
	}
	base.compute_micro_displacements(MatrixX::Constant(V.rows(), 3, 0), F, dummy_displacements);

	// compute uv scaling factor to match 3D area
	Scalar area_3d = 0;
	Scalar area_uv = 0;
	for (int fi = 0; fi < F.rows(); ++fi) {
		Matrix3 P;
		ARAP::Matrix32 U;
		for (int i = 0; i < 3; ++i) {
			P.row(i) = V.row(F(fi, i));
			U.row(i) = UV.row(F(fi, i));
		}
		area_3d += triangle_area(Vector3(V.row(F(fi, 0))), Vector3(V.row(F(fi, 1))), Vector3(V.row(F(fi, 2))));
		area_uv += triangle_area(Vector2(UV.row(F(fi, 0))), Vector2(UV.row(F(fi, 1))), Vector2(UV.row(F(fi, 2))));
	}

	bool mirrored = false;

	// check if UV needs mirroring
	if (area_uv < 0) {
		std::cout << "Mirroring UVs! (global atlas UV area is negative)" << std::endl;
		UV.col(0) *= -1;
		mirrored = true;
	}

	Scalar scale = std::sqrt(area_3d / std::abs(area_uv));

	MatrixX UV_init = scale * UV;

	// optimize UV atlas wrt base
	MatrixX UV_base = UV_init;
	ARAP arap_base(base, V, F);
	arap_base.solve(UV_base, 100);

	// optimize UV atlas wrt displaced micro-geometry
	MatrixX UV_opt = UV_base;
	ARAP arap(umesh, V, F);
	arap.solve(UV_opt, 100);

	if (mirrored) {
		UV_base.col(0) *= -1;
		UV_opt.col(0) *= -1;
	}

	fs::path gltf_name = input_mesh.filename().replace_extension();
	
	// pack base uvs and recompute tangents
	VFAdjacency VF = compute_adjacency_vertex_face(V, F);
	std::vector<Chart> charts = extract_charts(UV_base, F, VF);
	MatrixX UV_packed = pack_charts(charts, 2048, UV_base, F, VF);

	VT = compute_tangent_vectors(V, UV_packed, F);

	//write base
	//GLTFWriteInfo write_base;
	//write_base
	//	.write_faces(&F)
	//	.write_vertices(&V)
	//	.write_uvs(&UV_packed)
	//	.write_normals(&VN)
	//	.write_tangents(&VT);

	//write_gltf("uvbase_base_" + gltf_name.string(), write_base);

	Bary bary;
	extract_displacement_bary_data(umesh, &bary, true);

	// write the uv-mapped micromesh
	GLTFWriteInfo write_base_micro;
	write_base_micro
		.write_faces(&F)
		.write_vertices(&V)
		.write_uvs(&UV_packed)
		.write_normals(&VN)
		.write_tangents(&VT)
		.write_directions(&VD)
		.write_subdivision_bits(&SB)
		.write_bary(&bary);

	write_gltf("uvbase_" + gltf_name.string(), write_base_micro);

	// pack optimized uvs and recompute tangents
	charts = extract_charts(UV_opt, F, VF);
	UV_packed = pack_charts(charts, 2048, UV_opt, F, VF);

	VT = compute_tangent_vectors(V, UV_packed, F);

	// write the uv-mapped micromesh
	GLTFWriteInfo write_opt_micro;
	write_opt_micro
		.write_faces(&F)
		.write_vertices(&V)
		.write_uvs(&UV_packed)
		.write_normals(&VN)
		.write_tangents(&VT)
		.write_directions(&VD)
		.write_subdivision_bits(&SB)
		.write_bary(&bary);
	
	write_gltf("uvopt_" + gltf_name.string(), write_opt_micro);

	return 0;
}

