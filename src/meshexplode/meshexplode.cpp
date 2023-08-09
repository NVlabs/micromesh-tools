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

#include "tangent.h"
#include "bvh.h"
#include "mesh_io_gltf.h"
#include "mesh_io_bary.h"

#include "mesh_io.h"

#include "utils.h"

#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
	if (argc < 2) {
		std::cerr << "Usage: " << fs::path(argv[0]).filename().string() << " MICROMESH" << std::endl;
		std::cerr << "    MICROMESH The micromesh glTF file" << std::endl;
		std::cerr << std::endl;
		std::cerr << "This tool generates the following file" << std::endl;
		std::cerr << "    exploded_MICROMESH A glTF of the 'exploded' micromesh (subdivided, displaced and with per-vertex uvs)" << std::endl;
		return -1;
	}

	fs::path input_mesh(argv[1]);

	GLTFReadInfo read_micromesh;
	if (!read_gltf(input_mesh.string(), read_micromesh)) {
		std::cerr << "Error reading gltf file " << input_mesh << std::endl;
		return -1;
	}

	Assert(read_micromesh.has_vertices());
	Assert(read_micromesh.has_faces());
	Assert(read_micromesh.has_uvs());
	Assert(read_micromesh.has_normals());
	Assert(read_micromesh.has_tangents());
	Assert(read_micromesh.has_directions());
	Assert(read_micromesh.has_subdivision_mesh());

	Assert(read_micromesh.color_textures.size() <= 1);
	Assert(read_micromesh.normal_textures.size() <= 1);

	//MatrixX V = read_micromesh.get_vertices(); // not really needed
	MatrixXi F = read_micromesh.get_faces();
	MatrixX UV = read_micromesh.get_uvs();
	MatrixX VN = read_micromesh.get_normals();
	MatrixX VT = read_micromesh.get_tangents();

	SubdivisionMesh umesh = read_micromesh.get_subdivision_mesh();

	fs::path base = input_mesh.parent_path();
	fs::path gltf_name = input_mesh.filename().replace_extension();

	// write the exploded micromesh
	{
		MatrixX EV;
		MatrixX EUV;
		MatrixX EVN;
		MatrixX EVT;
		MatrixXi EF;
		umesh.extract_mesh_with_uvs(UV, VN, VT, F, EV, EF, EUV, EVN, EVT);

		EUV.col(1) = VectorX::Constant(EUV.rows(), 1) - EUV.col(1);

		//GLTFWriteInfo wi;
		//wi.write_faces(&EF).write_vertices(&EV).write_uvs(&EUV).write_normals(&EVN).write_tangents(&EVT);
		//if (read_micromesh.has_color_textures())
		//	wi.write_color_texture(read_micromesh.color_textures[0]);
		//if (read_micromesh.has_normal_textures())
		//	wi.write_normal_texture(read_micromesh.normal_textures[0]);

		//write_gltf("exploded_" + gltf_name.string(), wi);

		write_obj("exploded_" + gltf_name.string() + ".obj", EV, EF, EUV, EF);
	}

	return 0;
}

