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

#include "mesh_io.h"
#include "micro.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <vector>


static std::vector<std::string> tokenize_line(std::ifstream& ifs);
static std::vector<std::string> parse_face_indices(const std::string& token);

bool read_obj(const std::string& filename, MatrixX& V, MatrixXi& F, MatrixX& VT, MatrixXi& FT, MatrixX& VN, MatrixXi& FN)
{
	std::ifstream ifs(filename);
	if (!ifs)
		return false;

	std::vector<Vector3> vvec;
	std::vector<Vector2> vtvec;
	std::vector<Vector3> vnvec;
	std::vector<Vector3i> fvec;
	std::vector<Vector3i> ftvec;
	std::vector<Vector3i> fnvec;

	int nquad = 0;
	int npoly = 0;
	
	while (!ifs.eof()) {
		std::vector<std::string> tokens = tokenize_line(ifs);
	
		if (tokens.size() == 0)
			break;

		if (tokens[0] == "v") {
			if (tokens.size() <= 3) {
				std::cerr << "Warning: Vertex with dimension < 3, padding with 0s" << std::endl;
				for (uint64_t i = tokens.size(); i <= 3; ++i) {
					tokens.push_back("0");
				}
			}
			Vector3 v;
			v << std::atof(tokens[1].c_str()),
			     std::atof(tokens[2].c_str()),
			     std::atof(tokens[3].c_str());
			vvec.push_back(v);
		}
		else if (tokens[0] == "vt") {
			Assert(tokens.size() > 2);
			Vector2 vt;
			vt << std::atof(tokens[1].c_str()),
			      std::atof(tokens[2].c_str());
			vtvec.push_back(vt);
		}
		else if (tokens[0] == "vn") {
			Assert(tokens.size() > 3);
			Vector3 vn;
			vn << std::atof(tokens[1].c_str()),
				  std::atof(tokens[2].c_str()),
				  std::atof(tokens[3].c_str());
			vnvec.push_back(vn);
		}
		else if (tokens[0] == "f") {
			if (tokens.size() > 5) {
				npoly++;
			}
			else if (tokens.size() < 4) {
				std::cout << "Skipping face with too few indices" << std::endl;
			}
			else {
				Vector4i f(INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);
				Vector4i ft(INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);
				Vector4i fn(INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX);
				for (unsigned i = 1; i < tokens.size(); ++i) {
					std::vector<std::string> index_tuple = parse_face_indices(tokens[i]);
					Assert(index_tuple.size() == 3);
					f[i - 1] = std::atoi(index_tuple[0].c_str()) - 1;
					if (index_tuple[1].size() > 0)
						ft[i - 1] = std::atoi(index_tuple[1].c_str()) - 1;
					if (index_tuple[2].size() > 0)
						fn[i - 1] = std::atoi(index_tuple[2].c_str()) - 1;
				}
				fvec.push_back(Vector3i(f[0], f[1], f[2]));
				ftvec.push_back(Vector3i(ft[0], ft[1], ft[2]));
				fnvec.push_back(Vector3i(fn[0], fn[1], fn[2]));
				if (tokens.size() == 5) {
					fvec.push_back(Vector3i(f[0], f[2], f[3]));
					ftvec.push_back(Vector3i(ft[0], ft[2], ft[3]));
					fnvec.push_back(Vector3i(fn[0], fn[2], fn[3]));
					nquad++;
				}
			}
		}
		else if (tokens[0] == "mtllib") {

		}
		else if (tokens[0] == "usemtl") {

		}
	}

	if (npoly > 0)
		std::cout << "Skipped " << npoly << " polygonal faces" << std::endl;

	if (nquad > 0)
		std::cout << "Split " << nquad << " quadrilateral faces" << std::endl;

	ifs.close();

	int vn = int(vvec.size());
	int fn = int(fvec.size());

	V.resize(vn, 3);
	for (int i = 0; i < vn; ++i)
		V.row(i) = vvec[i];

	F.resize(fn, 3);
	for (int i = 0; i < fn; ++i) {
		F.row(i) = fvec[i];
		Assert(F(i, 0) >= 0);
		Assert(F(i, 1) >= 0);
		Assert(F(i, 2) >= 0);
		Assert(F(i, 0) < vn);
		Assert(F(i, 1) < vn);
		Assert(F(i, 2) < vn);
	}

	VT.resize(vtvec.size(), 2);
	for (int i = 0; i < vtvec.size(); ++i)
		VT.row(i) = vtvec[i];

	FT.resize(ftvec.size(), 3);
	for (int i = 0; i < ftvec.size(); ++i)
		FT.row(i) = ftvec[i];

	VN.resize(vnvec.size(), 3);
	for (int i = 0; i < vnvec.size(); ++i)
		VN.row(i) = vnvec[i];

	FN.resize(fnvec.size(), 3);
	for (int i = 0; i < fnvec.size(); ++i)
		FN.row(i) = fnvec[i];

	return true;
}

bool read_obj(const std::string& filename, MatrixX& V, MatrixXi& F)
{
	std::ifstream ifs(filename);
	if (!ifs)
		return false;

	std::vector<Vector3> vvec;
	std::vector<Vector3i> fvec;

	int nquad = 0;
	int npoly = 0;
	
	while (!ifs.eof()) {
		std::vector<std::string> tokens = tokenize_line(ifs);
	
		if (tokens.size() == 0)
			break;

		if (tokens[0] == "v") {
			if (tokens.size() <= 3) {
				std::cerr << "Warning: Vertex with dimension < 3, padding with 0s" << std::endl;
				for (uint64_t i = tokens.size(); i <= 3; ++i) {
					tokens.push_back("0");
				}
			}
			Vector3 v;
			v << std::atof(tokens[1].c_str()),
				 std::atof(tokens[2].c_str()),
				 std::atof(tokens[3].c_str());
			vvec.push_back(v);
		}
		else if (tokens[0] == "vt") {

		}
		else if (tokens[0] == "vn") {

		}
		else if (tokens[0] == "f") {
			if (tokens.size() > 5) {
				npoly++;
			}
			else if (tokens.size() < 4) {
				std::cout << "Skipping face with too few indices" << std::endl;
			}
			else {
				Vector4i f(0, 0, 0, 0);
				for (unsigned i = 1; i < tokens.size(); ++i) {
					std::vector<std::string> index_tuple = parse_face_indices(tokens[i]);
					f[i - 1] = std::atoi(index_tuple[0].c_str()) - 1;
				}
				fvec.push_back(Vector3i(f[0], f[1], f[2]));
				if (tokens.size() == 5) {
					fvec.push_back(Vector3i(f[0], f[2], f[3]));
					nquad++;
				}
			}
		}
		else if (tokens[0] == "mtllib") {

		}
		else if (tokens[0] == "usemtl") {

		}
	}

	if (npoly > 0)
		std::cout << "Skipped " << npoly << " polygonal faces" << std::endl;

	if (nquad > 0)
		std::cout << "Split " << nquad << " quadrilateral faces" << std::endl;

	ifs.close();

	int vn = int(vvec.size());
	int fn = int(fvec.size());

	V.resize(vn, 3);
	for (int i = 0; i < vn; ++i)
		V.row(i) = vvec[i];

	F.resize(fn, 3);
	for (int i = 0; i < fn; ++i) {
		F.row(i) = fvec[i];
		Assert(F(i, 0) >= 0);
		Assert(F(i, 1) >= 0);
		Assert(F(i, 2) >= 0);
		Assert(F(i, 0) < vn);
		Assert(F(i, 1) < vn);
		Assert(F(i, 2) < vn);
	}

	return true;
}

void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F)
{
	std::ofstream ofs(filename);
	if (!ofs) {
		std::cerr << "Error writing file " << filename << std::endl;
		std::exit(-1);
	}

	ofs.precision(8);
	//ofs.setf(std::ios::fixed);

	for (int i = 0; i < V.rows(); ++i) {
		ofs << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
	}

	for (int i = 0; i < F.rows(); ++i) {
		ofs << "f";
		for (int j = 0; j < F.cols(); ++j) {
			if (F(i, j) >= 0)
				ofs << " " << F(i, j) + 1;
			else
				ofs << " " << 1;
		}
		ofs << std::endl;
	}

	ofs.close();
}

void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& VN)
{
	std::ofstream ofs(filename);
	if (!ofs) {
		std::cerr << "Error writing file " << filename << std::endl;
		std::exit(-1);
	}

	ofs.precision(8);
	//ofs.setf(std::ios::fixed);

	for (int i = 0; i < V.rows(); ++i) {
		ofs << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
		ofs << "vn " << VN(i, 0) << " " << VN(i, 1) << " " << VN(i, 2) << std::endl;
	}

	for (int i = 0; i < F.rows(); ++i) {
		ofs << "f";
		for (int j = 0; j < F.cols(); ++j) {
			if (F(i, j) >= 0) {
				int idx = F(i, j) + 1;
				ofs << " " << idx << "//" << idx;
			}
			else {
				ofs << " " << 1 << "//" << 1;
			}
		}
		ofs << std::endl;
	}

	ofs.close();
}

void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& VT, const MatrixXi& FT)
{
	Assert(FT.size() == F.size());

	std::ofstream ofs(filename);
	if (!ofs) {
		std::cerr << "Error writing file " << filename << std::endl;
		std::exit(-1);
	}

	std::string mat_filename = filename + ".mtl";
	std::ofstream mat_ofs(mat_filename);
	if (!mat_ofs) {
		std::cerr << "Error writing file " << mat_filename << std::endl;
		std::exit(-1);
	}

	ofs.precision(8);
	mat_ofs.precision(8);

	ofs << "mtllib " << mat_filename << std::endl;

	for (int i = 0; i < V.rows(); ++i)
		ofs << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << '\n';

	for (int i = 0; i < VT.rows(); ++i)
		ofs << "vt " << VT(i, 0) << " " << VT(i, 1) << '\n';

	ofs << "usemtl Textured" << '\n';

	for (int i = 0; i < F.rows(); ++i) {
		ofs << "f";
		for (int j = 0; j < F.cols(); ++j) {
			if (F(i, j) >= 0)
				ofs << " " << F(i, j) + 1 << "/" << FT(i, j) + 1;
			else
				ofs << " " << 1 << "/" << 1;
		}
		ofs << '\n';
	}

	ofs.close();

	mat_ofs << "newmtl Textured" << std::endl;
	mat_ofs << "Ka " << 1.0 << " " << 1.0 << " " << 1.0 << std::endl;
	mat_ofs << "Kd " << 1.0 << " " << 1.0 << " " << 1.0 << std::endl;
	mat_ofs << "Ks " << 0.0 << " " << 0.0 << " " << 0.0 << std::endl;
	mat_ofs << "d 1.0" << std::endl;
	mat_ofs << "illum 2" << std::endl;
	mat_ofs << "map_Kd texture.png" << std::endl;

	mat_ofs.close();
}

void write_obj(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& VT, const MatrixXi& FT, const MatrixX& VN, const MatrixXi& FN)
{
	Assert(FT.size() == F.size());
	Assert(FN.size() == F.size());

	std::ofstream ofs(filename);
	if (!ofs) {
		std::cerr << "Error writing file " << filename << std::endl;
		std::exit(-1);
	}

	ofs.precision(8);

	for (int i = 0; i < V.rows(); ++i)
		ofs << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << '\n';

	for (int i = 0; i < VT.rows(); ++i)
		ofs << "vt " << VT(i, 0) << " " << VT(i, 1) << '\n';

	for (int i = 0; i < VN.rows(); ++i)
		ofs << "vn " << VN(i, 0) << " " << VN(i, 1) << " " << VN(i, 2) << '\n';

	for (int i = 0; i < F.rows(); ++i) {
		ofs << "f";
		for (int j = 0; j < F.cols(); ++j) {
			if (F(i, j) >= 0)
				ofs << " " << F(i, j) + 1 << "/" << FT(i, j) + 1 << "/" << FN(i, j) + 1;
			else
				ofs << " " << 1 << "/" << 1;
		}
		ofs << '\n';
	}

	ofs.close();

}

void write_obj(const std::string& filename, const SubdivisionMesh& umesh, Scalar displacement_factor)
{
	std::ofstream ofs(filename);
	if (!ofs) {
		std::cerr << "Error writing file " << filename << std::endl;
		std::exit(-1);
	}

	ofs.precision(8);

	int vertex_count = 1;

	for (const SubdivisionTri& st : umesh.faces) {
		for (int i = 0; i < st.V.rows(); ++i) {
			ofs << "v " << st.V(i, 0) + displacement_factor * st.VD(i, 0) << " "
			            << st.V(i, 1) + displacement_factor * st.VD(i, 1) << " "
			            << st.V(i, 2) + displacement_factor * st.VD(i, 2) << '\n';
		}

		for (int i = 0; i < st.F.rows(); ++i) {
			ofs << "f";
			for (int j = 0; j < st.F.cols(); ++j) {
				ofs << " " << st.F(i, j) + vertex_count;
			}
			ofs << '\n';
		}

		vertex_count += st.V.rows();
	}

	ofs.close();
}

void write_obj_lines(const std::string& filename, const std::vector<MatrixX>& Vs, const std::vector<std::vector<int>>& Ls)
{
	Assert(Vs.size() == Ls.size());
	std::ofstream ofs(filename);
	if (!ofs) {
		std::cerr << "Error writing file " << filename << std::endl;
		std::exit(-1);
	}

	ofs.precision(8);

	int vertex_count = 1;
	for (unsigned k = 0; k < Vs.size(); ++k) {
		const MatrixX& V = Vs[k];
		const std::vector<int>& L = Ls[k];
		for (int i = 0; i < V.rows(); ++i) {
			ofs << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
		}
		ofs << "l";
		for (int li : L)
			ofs << " " << li + vertex_count;
		ofs << "\n";

		vertex_count += V.rows();
	}

	ofs.close();
}

static std::vector<std::string> tokenize_line(std::ifstream& ifs)
{
	std::vector<std::string> tokens;
	std::string line;

	do {
		std::string input;
		std::getline(ifs, input);
		line = rtrim(input);
	} while ((line.empty() || line[0] == '#') && !ifs.eof());

	if (line.empty() || line[0] == '#')
		return {};
	
	auto it = line.begin();
	std::string tok;
	while (it != line.end()) {
		Assert(*it != '#');
		if (!whitespace(*it)) {
			tok.push_back(*it);
		}
		else {
			if (tok.size() > 0) {
				tokens.push_back(tok);
				tok.clear();
			}
		}
		it++;
	}

	if (tok.size() > 0)
		tokens.push_back(tok);

	return tokens;
}

static std::vector<std::string> parse_face_indices(const std::string& token)
{
	std::vector<std::string> index_tuple;
	std::string index_string;
	
	auto it = token.cbegin();

	while (it != token.cend()) {
		if (*it != '/') {
			index_string.push_back(*it);
		}
		else {
			index_tuple.push_back(index_string);
			index_string.clear();
		}
		it++;
	}

	if (index_string.size() > 0 || token.back() == '/')
		index_tuple.push_back(index_string);

	while (index_tuple.size() < 3)
		index_tuple.push_back("");
	
	return index_tuple;
}

inline int to256(float in01) { return int(in01 * 255); }

bool write_ply(const std::string& filename, const MatrixX& V, const MatrixXi& F, const MatrixX& C) {
	std::ofstream of;
	of.open(filename);
	if (!of.is_open()) return false;
	of << "ply\n"
		"format ascii 1.0\n"
		"element vertex " << V.rows() << "\n"
		"property float x\n"
		"property float y\n"
		"property float z\n"
		"property uchar red\n"
		"property uchar green\n"
		"property uchar blue\n"
		"property uchar alpha\n"
		"element face " << F.rows() << "\n"
		"property list uchar int vertex_indices\n"
		"end_header\n";
	for (uint32_t i = 0; i < V.rows(); i++) {
		of << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " ";
		of << to256(C(i, 0)) << " " << to256(C(i, 1)) << " " << to256(C(i, 2)) << " 255\n";
	}
	for (uint32_t i = 0; i < F.rows(); ++i) {
		of << "3 " << F(i, 0) << " " << F(i, 1) << " " << F(i, 2) << "\n";;
	}
	of.close();
	return true;
}

static bool read_stl_ascii(const std::string& filename, MatrixX& V, MatrixXi& F);
static bool read_stl_binary(const std::string& filename, MatrixX& V, MatrixXi& F);

bool read_stl(const std::string& filename, MatrixX& V, MatrixXi& F)
{
	std::ifstream ifs(filename, std::ios::binary);
	char header[6] = {};
	ifs.read(header, 5);
	if (!ifs) {
		std::cout << "Error reading STL file " << filename << std::endl;
		return false;
	}
	ifs.close();

	if (std::string(header) == "solid")
		return read_stl_ascii(filename, V, F);
	else
		return read_stl_binary(filename, V, F);

}

static bool read_stl_ascii(const std::string& filename, MatrixX& V, MatrixXi& F)
{
	std::ifstream ifs(filename);
	if (!ifs)
		return false;
	std::string str;
	ifs >> str;
	if (str != "solid")
		return false;
	ifs >> str;
	std::vector<Vector3> tris;
	while (ifs) {
		ifs >> str;
		if (str == "facet") {
			Vector3 v;
			// skip normal
			for (int i = 0; i < 4; ++i)
				ifs >> str;
			ifs >> str;
			if (str != "outer")
				return false;
			ifs >> str; // skip 'loop'
			for (int i = 0; i < 3; ++i) {
				ifs >> str; // 'vertex'
				ifs >> v(0);
				ifs >> v(1);
				ifs >> v(2);
				tris.push_back(v);
			}
			if (!ifs)
				return false;
			ifs >> str; // endloop
			ifs >> str; // endfacet
		}
	}
	if (tris.size() % 3 != 0)
		return false;

	uint32_t ntris = tris.size() / 3;

	V.resize(3 * ntris, 3);
	F.resize(ntris, 3);

	for (uint32_t i = 0; i < ntris; ++i) {
		for (uint32_t j = 0; j < 3; ++j) {
			V.row(3 * i + j) = tris[3 * i + j];
			F(i, j) = 3 * i + j;
		}
	}

	return true;
}

static bool read_stl_binary(const std::string& filename, MatrixX& V, MatrixXi& F)
{
	std::ifstream ifs(filename, std::ios::binary);
	bool fail = false;
	char header[80];
	uint32_t ntris;
	if (!ifs.read(header, 80)) {
		return false;
	}
	
	if (!ifs.read((char*)&ntris, sizeof(uint32_t))) {
		return false;
	}

	V.resize(3 * ntris, 3);
	F.resize(ntris, 3);

	for (uint32_t i = 0; i < ntris; ++i) {
		Vector3f n; // unused
		Vector3f v0;
		Vector3f v1;
		Vector3f v2;
		char attr[2]; // unused
		ifs.read(reinterpret_cast<char *>(&n), 3 * sizeof(float));
		ifs.read(reinterpret_cast<char *>(&v0), 3 * sizeof(float));
		ifs.read(reinterpret_cast<char *>(&v1), 3 * sizeof(float));
		ifs.read(reinterpret_cast<char *>(&v2), 3 * sizeof(float));
		V.row(3 * i + 0) = v0.cast<Scalar>();
		V.row(3 * i + 1) = v1.cast<Scalar>();
		V.row(3 * i + 2) = v2.cast<Scalar>();
		F(i, 0) = 3 * i + 0;
		F(i, 1) = 3 * i + 1;
		F(i, 2) = 3 * i + 2;
		ifs.read(attr, 2);
		if (!ifs) {
			V.resize(0, 0);
			F.resize(0, 0);
			return false;
		}
	}

	return true;
}

