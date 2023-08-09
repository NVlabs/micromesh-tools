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

#ifdef _MSC_VER

// Force use of discrete nvidia gpu
#include <windows.h>
extern "C" {
	_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}

#endif

#include "gui.h"
#include "session.h"
#include "mesh_io.h"
#include "mesh_io_gltf.h"
#include "mesh_io_bary.h"
#include "clean.h"

#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
	if (argc <= 2) {
		GUIApplication app;
		app.start(argc > 1 ? argv[1] : nullptr);
	}
	else {
		std::string mesh_file = argv[1];
		std::string cmd_file = argv[2];

		Session session;

		bool read_cmds = session.sequence.read_from_file(cmd_file);
		if (!read_cmds) {
			std::cerr << "Unable to read commands file " << cmd_file << std::endl;
			std::exit(-1);
		}

		bool mesh_loaded = session.load_mesh(mesh_file);
		if (!mesh_loaded) {
			std::cerr << "Unable to read mesh file " << mesh_file << std::endl;
			std::exit(-1);
		}
		else {
			std::cerr << "Loaded mesh " << mesh_file << std::endl;
		}

		std::string out_prefix = "out_";
		if (argc == 4)
			out_prefix = argv[3];
		
		session.save_prefix = out_prefix;
		session.execute();

		{
			VectorXu8 subdivision_bits = session.compute_subdivision_bits();

			Bary bary;
			extract_displacement_bary_data(session.micromesh, &bary, true);

			MatrixX V = session.base.V + bary.min_displacement * session.base.VD;
			MatrixX VD = (bary.max_displacement - bary.min_displacement) * session.base.VD;

			fs::path in_path = mesh_file;
			fs::path out_path = out_prefix + in_path.filename().string();
			out_path.replace_extension("gltf");

			GLTFWriteInfo write_info;
			write_info
				.write_faces(&session.base.F)
				.write_vertices(&V)
				.write_normals(&session.base.VN)
				.write_directions(&VD)
				.write_subdivision_bits(&subdivision_bits)
				.write_bary(&bary);

			write_gltf(out_path.string(), write_info);
		}

	}

	return 0;
}

