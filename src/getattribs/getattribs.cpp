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

#include "utils.h"

#include "space.h"

#include <filesystem>

namespace fs = std::filesystem;


// returns a vector that assign to each face in F1 the corresponding face in F2
// correspondences are computed by projecting face barycenters
//VectorXi matching_faces(const MatrixX& V1, const MatrixXi& F1, const MatrixX& V2, const MatrixXi& F2)
//{
//	Assert(F1.size() == F2.size());
//
//	VectorXi matching = VectorXi::Constant(F1.rows(), -1);
//
//	MatrixX VN1 = compute_vertex_normals(V1, F1);
//
//	BVHTree bvh;
//	bvh.build_tree(&V1, &F1, &VN1);
//
//	std::set<int> matched;
//
//	for (int f2i = 0; f2i < F2.rows(); ++f2i) {
//		Vector3 fp = (V2.row(F2(f2i, 0)) + V2.row(F2(f2i, 1)) + V2.row(F2(f2i, 2))) / (Scalar)3.0;
//		NearestInfo ni;
//		bvh.nearest_point(fp, &ni);
//
//		matching(ni.fi) = f2i;
//		matched.insert(ni.fi);
//	}
//
//	Assert(matched.size() == matching.size() && "Not all input faces were matched");
//	Assert(matching.minCoeff() == 0);
//
//	return matching;
//}

VectorXi matching_faces(const MatrixX& V1, const MatrixXi& F1, const MatrixX& V2, const MatrixXi& F2)
{
	Assert(F1.size() == F2.size());

	VectorXi matching = VectorXi::Constant(F1.rows(), -1);

	for (int f2i = 0; f2i < F2.rows(); ++f2i)
		matching(f2i) = f2i;
		
	return matching;
}


int main(int argc, char *argv[])
{
	if (argc != 3) {
		std::cerr << "Usage: " << fs::path(argv[0]).filename().string() << " MICROMESH BASE_WITH_ATTRIBS" << std::endl;
		std::cerr << "    MICROMESH         The micromesh glTF file" << std::endl;
		std::cerr << "    BASE_WITH_ATTRIBS The mesh whose base attributes (uvs and tangents) are imported" << std::endl;
		std::cerr << std::endl;
		std::cerr << "This tool generates the following file:" << std::endl;
		std::cerr << "    uv_MICROMESH A micromesh glTF file that includes the uvs, normals and tangents provided by BASE_WITH_ATTRIBS" << std::endl;
		return -1;
	}

	fs::path input_mesh(argv[1]);
	fs::path uv_mesh(argv[2]);

	// read input micromesh

	GLTFReadInfo read_micromesh;
	if (!read_gltf(input_mesh.string(), read_micromesh)) {
		std::cerr << "Error reading gltf file " << input_mesh << std::endl;
		return -1;
	}

	Assert(read_micromesh.has_vertices());
	Assert(read_micromesh.has_faces());
	Assert(read_micromesh.has_directions());
	Assert(read_micromesh.has_subdivisions());
	Assert(read_micromesh.has_topology_flags());
	Assert(read_micromesh.has_subdivision_mesh());

	MatrixX V1 = read_micromesh.get_vertices();
	MatrixXi F1 = read_micromesh.get_faces();
	MatrixX VD1 = read_micromesh.get_directions();
	VectorXu8 S1 = read_micromesh.get_subdivisions();
	VectorXu8 E1 = read_micromesh.get_topology_flags();

	SubdivisionMesh umesh = read_micromesh.get_subdivision_mesh();

	// pack subdivisions and edge flags
	VectorXu8 SB1 = S1;
	for (int i = 0; i < SB1.size(); ++i)
		SB1(i) = (S1(i) << 3) | E1(i);

	// read base mesh with extra attribs to import

	GLTFReadInfo read_attrib_mesh;
	if (!read_gltf(uv_mesh.string(), read_attrib_mesh)) {
		std::cerr << "Error reading gltf file " << uv_mesh << std::endl;
		return -1;
	}

	Assert(read_attrib_mesh.has_vertices());
	Assert(read_attrib_mesh.has_faces());
	Assert(read_attrib_mesh.has_uvs());
	Assert(read_attrib_mesh.has_normals());
	Assert(read_attrib_mesh.has_tangents());

	MatrixX V2 = read_attrib_mesh.get_vertices();
	MatrixXi F2 = read_attrib_mesh.get_faces();
	MatrixX UV2 = read_attrib_mesh.get_uvs();
	MatrixX VN2 = read_attrib_mesh.get_normals();
	MatrixX VT2 = read_attrib_mesh.get_tangents();

	Box3 box1;
	for (int i = 0; i < V1.rows(); ++i)
		box1.add(V1.row(i));

	Box3 box2;
	for (int i = 0; i < V2.rows(); ++i)
		box2.add(V2.row(i));

	std::cout << "Box 1 " << box1.cmin.transpose() << " " << box1.cmax.transpose() << std::endl;
	std::cout << "Box 2 " << box2.cmin.transpose() << " " << box2.cmax.transpose() << std::endl;
	
	
	// build a new micromesh combining the micromesh data
	// of the first mesh with the vertex attributes of the second

	VectorXi matching = matching_faces(V2, F2, V1, F1);

	// match directions
	MatrixX VD2 = MatrixX::Constant(V2.rows(), 3, 0);
	for (int i = 0; i < matching.rows(); ++i) {
		for (int j = 0; j < 3; ++j)
			VD2.row(F2(i, j)) = VD1.row(F1(matching(i), j));
	}

	// match subdivisions
	VectorXu8 S2 = VectorXu8::Constant(F2.rows(), 0);
	for (int i = 0; i < matching.rows(); ++i) {
		S2(i) = S1(matching(i));
	}

	// match topology flags
	VectorXu8 E2 = VectorXu8::Constant(F2.rows(), 0);
	for (int i = 0; i < matching.rows(); ++i) {
		E2(i) = E1(matching(i));
	}

	// pack subdivisions and edge flags
	VectorXu8 SB2 = S2;
	for (int i = 0; i < SB2.size(); ++i)
		SB2(i) = (S2(i) << 3) | E2(i);

	// extract microdisplacements (one vec per base face)
	std::vector<std::vector<Scalar>> micro_displacements(F2.rows());

	for (int i = 0; i < F2.rows(); ++i) {
		int f1i = matching(i);
		SubdivisionTri& st = umesh.faces[f1i];
		BarycentricGrid grid(st.subdivision_level());
		for (int uvi = 0; uvi < grid.num_samples(); ++uvi) {
			micro_displacements[i].push_back(st.scalar_displacement(uvi));
		}
	}

	// subdivide and displace the new micromesh
	SubdivisionMesh umesh2;
	umesh2.compute_mesh_structure(V2, F2, SB2);
	umesh2.compute_micro_displacements(VD2, F2, micro_displacements);

	// compute the new bary data (this may be necessary in the remote case
	// where the mapping between the base faces of the two mesh is not 
	// the identity)
	Bary bary;
	extract_displacement_bary_data(umesh2, &bary, true);

	// write the uv-mapped micromesh
	GLTFWriteInfo write_info;
	write_info
		.write_faces(&F2)
		.write_vertices(&V2)
		.write_uvs(&UV2)
		.write_normals(&VN2)
		.write_tangents(&VT2)
		.write_directions(&VD2)
		.write_subdivision_bits(&SB2)
		.write_bary(&bary);

	fs::path gltf_name = input_mesh.filename().replace_extension();

	write_gltf("uv_" + gltf_name.string(), write_info);

	return 0;
}

