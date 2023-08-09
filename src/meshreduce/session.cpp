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

#include "session.h"
#include "utils.h"

#include "clean.h"
#include "mesh_utils.h"
#include "direction_field.h"
#include "smooth.h"
#include "flip.h"
#include "visibility.h"
#include "mesh_io.h"
#include "mesh_io_gltf.h"
#include "mesh_io_bary.h"

#include "quality.h"

#include <sstream>
#include <thread>
#include <algorithm>
#include <execution>

//#define NO_PREPROCESS

std::string SessionCommand::to_string() const
{
	std::stringstream ss;

	if (Commands.find(name) != Commands.end()) {
		CommandType cmd = Commands[name];
		Assert(CommandStrings.find(cmd) != CommandStrings.end());

		ss << name;
		for (const CommandArgument& arg : args) {
			if (CommandArgTypes[name].find(arg.name) != CommandArgTypes[name].end()) {
				std::stringstream argss;
				argss << " -" << arg.name;
				ArgType arg_type = CommandArgTypes[name][arg.name];
				switch (arg_type) {
				case StringT:  argss << " " << arg.value.string_val; break;
				case ScalarT:  argss << " " << arg.value.scalar_val; break;
				case IntegerT: argss << " " << arg.value.integer_val; break;
				case Vector3T: argss << " " << arg.value.vector3_val[0] << ","
				                            << arg.value.vector3_val[1] << ","
				                            << arg.value.vector3_val[2]; break;
				case BoolT:    argss << " " << (arg.value.bool_val ? "true" : "false"); break;
				default: argss.clear(); break; // should never happen
				}
				ss << argss.str();
			}
			else {
				std::cerr << name << ": unrecognized argument " << arg.name << std::endl;
			}
		}

	}

	return ss.str();
}

static std::vector<std::string> tokenize_line(std::string line)
{
	std::vector<std::string> tokens;

	if (line.empty() || line[0] == '#')
		return {};
	
	auto it = line.begin();
	std::string tok;
	while (it != line.end()) {
		assert(*it != '#');
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

bool SessionCommand::from_string(const std::string& s)
{
	name = "";
	args.clear();

	std::vector<std::string> tokens = tokenize_line(s);
	if (tokens.size() > 0) {
		std::string cmdstring = tokens[0];

		if (Commands.find(cmdstring) != Commands.end()) {
			name = cmdstring;
			SessionCommand::_parse_args(tokens);
			return true;
		}
	}

	return false;
}

static void string_to_vec3(const std::string& s, double vec3[])
{
	std::string sv3[3];

	int i = 0;
	auto it = s.begin();
	while (it != s.end() && i < 3) {
		if (*it == ',')
			i++;
		else
			sv3[i].push_back(*it);
		it++;
	}

	for (unsigned i = 0; i < 3; ++i)
		vec3[i] = std::atof(sv3[i].c_str());
}

static bool string_to_bool(const std::string& s)
{
	if (s == "false" || s == "0")
		return false;

	return true;
}

void SessionCommand::_parse_args(const std::vector<std::string>& tokens)
{
	Assert(Commands.find(name) != Commands.end());
	CommandType cmd = Commands[name];

	unsigned token_index = 1;
	while (token_index < tokens.size()) {
		std::string argname = tokens[token_index];
		if (argname.starts_with('-') && ++token_index < tokens.size()) {
			// ok, parse arg
			argname = argname.substr(1);
			if (CommandArgTypes[name].find(argname) != CommandArgTypes[name].end()) {
				CommandArgument arg;
				arg.name = argname;
				ArgType arg_type = CommandArgTypes[name][arg.name];
				switch (arg_type) {
				case StringT:  std::snprintf(arg.value.string_val, CommandArgument::MAX_STRING_SIZE, "%s", tokens[token_index].c_str()); break;
				case ScalarT:  arg.value.scalar_val = std::atof(tokens[token_index].c_str()); break;
				case IntegerT: arg.value.integer_val = std::atoi(tokens[token_index].c_str()); break;
				case Vector3T: string_to_vec3(tokens[token_index], arg.value.vector3_val); break;
				case BoolT:    arg.value.bool_val = string_to_bool(tokens[token_index]); break;
				default: Assert(0); break; // should never happen
				}
				args.push_back(arg);
			}
		}
		else {
			token_index++;
		}
	}
}

void CommandSequence::add_command(const SessionCommand& cmd)
{
	commands.push_back(cmd);
}

void CommandSequence::clear_commands()
{
	commands.clear();
}

bool CommandSequence::read_from_file(const std::string& file_path)
{
	std::ifstream ifs(file_path);
	if (!ifs)
		return false;

	commands.clear();

	while (!ifs.eof()) {
		std::string line;
		std::getline(ifs, line);
		line = rtrim(line);

		if (!line.empty()) {
			SessionCommand cmd;
			if (cmd.from_string(line))
				commands.push_back(cmd);
		}
	}

	return true;
}

bool CommandSequence::write_to_file(const std::string& file_path) const
{
	std::ofstream ofs(file_path);
	if (!ofs)
		return false;

	ofs << to_string();
	ofs.close();

	return true;
}

std::string CommandSequence::to_string() const
{
	std::stringstream ss;
	for (const SessionCommand& cmd : commands)
		ss << cmd.to_string() << std::endl;
	return ss.str();
}

// used with tangent displacements
static void build_border_mesh_vertical(const MatrixX& V, const MatrixXi& F, const MatrixX& VN, Scalar scale, MatrixX& WV, MatrixXi& WF, MatrixX& WVN, bool orthogonal)
{
	MatrixXu8 FEB = per_face_edge_border_flag(V, F);

	int border_edges = FEB.count();

	WF = MatrixXi::Constant(4 * border_edges, 3, 0);
	WV = MatrixX::Constant(8 * border_edges, 3, 0);
	WVN = MatrixX::Constant(8 * border_edges, 3, 0);

	if (border_edges == 0)
		return;

	MatrixX H; // border mesh `height` vectors
	if (orthogonal) {
		H = VN;
	}
	else {
		H = MatrixX::Constant(VN.rows(), 3, 0);
		MatrixX FN = compute_face_normals(V, F);
		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < 3; ++j) {
				if (FEB(i, j)) {
					int e0 = F(i, j);
					int e1 = F(i, (j + 1) % 3);
					Vector3 e10 = V.row(e1) - V.row(e0);
					Vector3 n = FN.row(i);
					Vector3 t = e10.cross(n).normalized();
					H.row(e0) += t;
					H.row(e1) += t;
				}
			}
		}
		for (int vi = 0; vi < H.rows(); ++vi) {
			if (!H.row(vi).isZero()) {
				H.row(vi).normalize();
			}
		}
	}

	int fi = 0;
	int vi = 0;

	for (int i = 0; i < FEB.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			if (FEB(i, j)) {
				Assert(vi < WV.rows());
				Assert(fi < WF.rows());

				int v0 = F(i, j);
				int v1 = F(i, (j + 1) % 3);

				WV.row(vi) = V.row(v0);
				WV.row(vi + 1) = V.row(v0) + scale * H.row(v0);
				WVN.row(vi) = WVN.row(vi + 1) = VN.row(v0);
				WV.row(vi + 2) = V.row(v1);
				WV.row(vi + 3) = V.row(v1) + scale * H.row(v1);
				WVN.row(vi + 2) = WVN.row(vi + 3) = VN.row(v1);
				WF.row(fi) = Vector3i(vi, vi + 1, vi + 2);
				WF.row(fi + 1) = Vector3i(vi + 1, vi + 3, vi + 2);
				vi += 4;
				fi += 2;

				WV.row(vi) = V.row(v0);
				WV.row(vi + 1) = V.row(v0) - scale * H.row(v0);
				WVN.row(vi) = WVN.row(vi + 1) = VN.row(v0);
				WV.row(vi + 2) = V.row(v1);
				WV.row(vi + 3) = V.row(v1) - scale * H.row(v1);
				WVN.row(vi + 2) = WVN.row(vi + 3) = VN.row(v1);
				WF.row(fi) = Vector3i(vi, vi + 2, vi + 1);
				WF.row(fi + 1) = Vector3i(vi + 2, vi + 3, vi + 1);
				vi += 4;
				fi += 2;
			}
		}
	}
}

static void build_border_mesh(const MatrixX& V, const MatrixXi& F, const MatrixX& VN, Scalar scale, MatrixX& WV, MatrixXi& WF, MatrixX& WVN, bool orthogonal)
{
	MatrixXu8 FEB = per_face_edge_border_flag(V, F);

	int border_edges = FEB.count();

	WF = MatrixXi::Constant(4 * border_edges, 3, 0);
	WV = MatrixX::Constant(8 * border_edges, 3, 0);
	WVN = MatrixX::Constant(8 * border_edges, 3, 0);

	//WF = MatrixXi::Constant(2 * border_edges, 3, 0);
	//WV = MatrixX::Constant(4 * border_edges, 3, 0);
	//WVN = MatrixX::Constant(4 * border_edges, 3, 0);

	if (border_edges == 0)
		return;

	MatrixX H; // border mesh `height` vectors
	if (orthogonal) {
		H = VN;
	}
	else {
		H = MatrixX::Constant(VN.rows(), 3, 0);
		MatrixX FN = compute_face_normals(V, F);
		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < 3; ++j) {
				if (FEB(i, j)) {
					int e0 = F(i, j);
					int e1 = F(i, (j + 1) % 3);
					Vector3 e10 = V.row(e1) - V.row(e0);
					Vector3 n = FN.row(i);
					Vector3 t = e10.cross(n).normalized();
					H.row(e0) += t;
					H.row(e1) += t;
				}
			}
		}
		for (int vi = 0; vi < H.rows(); ++vi) {
			if (!H.row(vi).isZero()) {
				H.row(vi).normalize();
			}
		}
	}

	int fi = 0;
	int vi = 0;

	for (int i = 0; i < FEB.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			if (FEB(i, j)) {
				Assert(vi < WV.rows());
				Assert(fi < WF.rows());

				int v0 = F(i, j);
				int v1 = F(i, (j + 1) % 3);

				WV.row(vi) = V.row(v0);
				WV.row(vi + 1) = V.row(v0) + scale * H.row(v0);
				WVN.row(vi) = WVN.row(vi + 1) = VN.row(v0);
				WV.row(vi + 2) = V.row(v1);
				WV.row(vi + 3) = V.row(v1) + scale * H.row(v1);
				WVN.row(vi + 2) = WVN.row(vi + 3) = VN.row(v1);
				WF.row(fi) = Vector3i(vi, vi + 1, vi + 2);
				WF.row(fi + 1) = Vector3i(vi + 1, vi + 3, vi + 2);
				vi += 4;
				fi += 2;

				WV.row(vi) = V.row(v0);
				WV.row(vi + 1) = V.row(v0) - scale * H.row(v0);
				WVN.row(vi) = WVN.row(vi + 1) = VN.row(v0);
				WV.row(vi + 2) = V.row(v1);
				WV.row(vi + 3) = V.row(v1) - scale * H.row(v1);
				WVN.row(vi + 2) = WVN.row(vi + 3) = VN.row(v1);
				WF.row(fi) = Vector3i(vi, vi + 2, vi + 1);
				WF.row(fi + 1) = Vector3i(vi + 2, vi + 3, vi + 1);
				vi += 4;
				fi += 2;
			}
		}
	}
}

bool Session::load_mesh(const std::filesystem::path& mesh_path)
{
	// read into base, since Decimation needs both hi and base read anyway
	_timers.t_session.reset();

	bool read_mesh = false;
	if (lowercase(mesh_path.extension().string()) == ".obj") {
		read_mesh = read_obj(mesh_path.string(), base.V, base.F);
	}
	else if (lowercase(mesh_path.extension().string()) == ".stl") {
		read_mesh = read_stl(mesh_path.string(), base.V, base.F);
	}
	else if (lowercase(mesh_path.extension().string()) == ".gltf") {
		GLTFReadInfo read_info;
		read_mesh = read_gltf(mesh_path.string(), read_info);
		if (read_mesh) {
			base.V = read_info.get_vertices();
			base.F = read_info.get_faces();
		}
	}

	_timers.t_proc.reset();

	if (!read_mesh) {
		std::cerr << "Error reading mesh file " << mesh_path << std::endl;
		current_mesh.clear();
		return false;
	}

	std::cout << "Reading mesh file took " << _timers.t_session.time_elapsed() << " seconds" << std::endl;

	MatrixXu8 FEB = per_face_edge_border_flag(base.V, base.F);
	VectorXu8 VB = per_vertex_border_flag(base.V, base.F, FEB);

	unify_vertices(base.V, base.F, VB, FEB);

	base.VF = compute_adjacency_vertex_face(base.V, base.F);

#ifndef NO_PREPROCESS
	
	//std::cout << "Removing near-degenerate triangles";
	//int n = 0;
	//while (n++ < 5 && remove_thin_triangles(base.V, base.F, base.VF, VB))
	//	std::cout << '.';
	//std::cout << " done." << std::endl;


	//int nzero = remove_zero_area_faces(base.V, base.F);
	//if (nzero > 0) {
	//	std::cout << "Removed " << nzero << " null faces" << std::endl;
	//	remove_degenerate_faces_inplace(base.F);
	//}

#endif

	remove_degenerate_faces_inplace(base.F);
	compact_vertex_data(base.F, base.V, VB);

	base.VF = compute_adjacency_vertex_face(base.V, base.F);
	base.VN = compute_vertex_normals(base.V, base.F);

	base.FR = compute_face_aspect_ratios(base.V, base.F);
	base.FN = compute_face_normals(base.V, base.F);

	// init hi mesh
	hi.F = base.F;
	hi.V = base.V;
	hi.VN = base.VN;
	hi.VQ = VectorX();
	hi.VF = base.VF;

	hi.D = MatrixX();
	hi.DW = VectorX();

	//compute_direction_field(hi.V, hi.F, hi.VF, hi.D, hi.DW);
	//for (int i = 0; i < 3; ++i)
	//	smooth_direction_field(hi.F, hi.VF, hi.D, hi.DW);

	//base.D = hi.D;
	//base.DW = hi.DW;

	hi.bvh = BVHTree();
	hi.bvh.build_tree(&hi.V, &hi.F, &hi.VN, 32);

	hi.bvh_test = [&](const IntersectionInfo& ii) -> bool {
		return ii.d.dot(compute_face_normal(ii.fi, hi.V, hi.F)) >= 0;
	};

	hi.box = Box3();
	for (int i = 0; i < hi.V.rows(); ++i)
		hi.box.add(hi.V.row(i));
	hi.avg_edge_length = average_edge(hi.V, hi.F);

	//border.F.resize(0, 0);
	//border.V.resize(0, 0);
	//border.VN.resize(0, 0);
	//build_border_mesh_vertical(hi.V, hi.F, hi.VN, 0.003 * hi.box.diagonal().norm(), border.V, border.F, border.VN, true);

	border.F.resize(0, 0);
	border.V.resize(0, 0);
	border.VN.resize(0, 0);

	build_border_mesh(hi.V, hi.F, hi.VN, 0.002 * hi.box.diagonal().norm(), border.V, border.F, border.VN, false);

	// init Decimation
	algo.handle = std::make_shared<Decimation>(hi.V, hi.F, base.V, base.F, base.VF, base.D, base.DW, base.FR, base.FN);

	proxy.V = MatrixX();
	proxy.F = MatrixXi();
	proxy.VN = MatrixX();
	proxy.VF = VFAdjacency();
	proxy.Q.clear();
	proxy.QW.clear();
	proxy.FR = VectorX();
	proxy.FN = MatrixX();

	proxy.D = MatrixX();
	proxy.DW = VectorX();

	// clear rest of base mesh data since base at first is 'hidden'
	base.VD = MatrixX();
	base.valid_fn = 0;
	base.top_VD = MatrixX();
	base.bottom_VD = MatrixX();

	base.subdivisions = VectorXu8();
	base.corrections = VectorXi8();
	base.flags = VectorXu8();

	// clear micro mesh
	micromesh = SubdivisionMesh();

	// set current mesh path
	current_mesh = mesh_path;

	std::cout << "Loading mesh took " << _timers.t_session.time_elapsed() << std::endl;

	return true;
}

bool Session::read_commands(const std::string& file_path)
{
	return sequence.read_from_file(file_path);
}

void Session::add_command(const SessionCommand& cmd)
{
	sequence.add_command(cmd);
}

void Session::clear_commands()
{
	sequence.clear_commands();
}

void Session::execute()
{
	for (const SessionCommand& cmd : sequence.commands)
		execute(cmd);
}

void Session::execute(const SessionCommand& cmd)
{
	if (Commands.find(cmd.name) != Commands.end()) {
		std::cerr << "Executing command: " << cmd.to_string() << std::endl;
		CommandType cmd_type = Commands[cmd.name];
		switch (cmd_type) {
		case InitProxy: _execute_init_proxy(cmd); break;
		case LoadBaseMesh: _execute_load_base_mesh(cmd); break;
		case Decimate: _execute_decimate(cmd); break;
		case Tessellate: _execute_tessellate(cmd); break;
		case SetDisplacementDirs: _execute_set_displacement_dirs(cmd); break;
		case Displace: _execute_displace(cmd); break;
		case TweakTessellation: _execute_tweak_tessellation(cmd); break;
		case TweakDisplacementDir: _execute_tweak_displacement_dir(cmd); break;
		case OptimizeBaseTopology: _execute_optimize_base_topology(cmd); break;
		case OptimizeBasePositions: _execute_optimize_base_positions(cmd); break;
		case MinimizePrismoids: _execute_minimize_prismoids(cmd); break;
		case ResetTessellationOffsets: _execute_reset_tessellation_offsets(cmd); break;
		case FlipEdge: _execute_flip_edge(cmd); break;
		case SplitEdge: _execute_split_edge(cmd); break;
		case MoveVertex: _execute_move_vertex(cmd); break;
		case SplitVertex: _execute_split_vertex(cmd); break;
		case Save: _execute_save(cmd); break;
		case SaveStats: _execute_save_stats(cmd); break;
		default: Assert(0 && "Missing cmd dispatch case"); break; // should never happen
		}
	}
	else {
		std::cerr << "Unrecognized command: " << cmd.to_string() << std::endl;
	}
}

void Session::_base_mesh_changed(bool reset_directions)
{
	algo.handle->compact_faces();

	// snap border to wall mesh
	if (border.V.size() > 0) {
		BVHTree bvh_border;
		bvh_border.build_tree(&border.V, &border.F, &border.VN);

		const VectorXu8& base_VB = algo.handle->dcb->VB;
		for (int vi = 0; vi < base_VB.rows(); ++vi) {
			if (base_VB(vi)) {
				Vector3 p_from = base.V.row(vi);

				NearestInfo ni;
				if (bvh_border.nearest_point(p_from, &ni))
					base.V.row(vi) = ni.p;
			}
		}
	}

	base.VF = compute_adjacency_vertex_face(base.V, base.F);
	base.VN = compute_vertex_normals(base.V, base.F);

	// default to smooth vertex normals for displacement directions
	if (reset_directions)
		base.VD = base.VN;

	base.valid_fn = long(base.F.rows());

	base.corrections = VectorXi8::Constant(base.F.rows(), 0);

	if (micromesh.is_displaced()) {
		micromesh.clear_micro_displacements();
	}
}

void Session::_proxy_mesh_changed()
{
	remove_degenerate_faces_inplace(proxy.F, proxy.FR, proxy.FN);
	compact_vertex_data(proxy.F, proxy.V, proxy.Q, proxy.QW, proxy.D, proxy.DW);

	proxy.VF = compute_adjacency_vertex_face(proxy.V, proxy.F);
	proxy.VN = compute_vertex_normals(proxy.V, proxy.F);
}

VectorXu8 Session::compute_subdivision_bits() const
{
	if (base.subdivisions.size() == 0 || base.corrections.size() == 0 || base.flags.size() == 0)
		return VectorXu8();

	VectorXu8 subdivision_bits = VectorXu8::Constant(base.F.rows(), 0);
	for (int i = 0; i < base.F.rows(); ++i) {
		subdivision_bits(i) = (uint8_t(base.subdivisions(i) + base.corrections(i)) << 3) | base.flags(i);
	}

	return subdivision_bits;
}

void Session::_execute_init_proxy(const SessionCommand& cmd)
{
	std::cerr << "_execute_init_proxy()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::InitProxy);

	namespace cap = cmdargs::init_proxy;

	int smoothing_iterations = 0;
	int anisotropic_smoothing_iterations = 0;
	Scalar anisotropic_smoothing_weight = 0;

	DecimationParameters dparams;

	bool has_reduction = false;
	int fn_target = -1;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cap::smoothing_iterations) {
			if (arg.value.integer_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				smoothing_iterations = arg.value.integer_val;
			}
		}
		else if (arg.name == cap::anisotropic_smoothing_iterations) {
			if (arg.value.integer_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				anisotropic_smoothing_iterations = arg.value.integer_val;
			}
		}
		else if (arg.name == cap::anisotropic_smoothing_weight) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				anisotropic_smoothing_weight = arg.value.scalar_val;
			}
		}
		else if (arg.name == cap::border_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.border_error_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cap::aspect_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.ar_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cap::visibility_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.visibility_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cap::normals_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.normals_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cap::vertex_smoothing) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.use_vertex_smoothing = true;
				dparams.smoothing_coefficient = arg.value.scalar_val;
			}
		}
		else if (arg.name == cap::max_error) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.bound_geometric_error = true;
				dparams.max_relative_error = arg.value.scalar_val;
			}
		}
		else if (arg.name == cap::min_aspect_ratio) {
			if (arg.value.scalar_val < 0 || arg.value.scalar_val > 1) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.bound_aspect_ratio = true;
				dparams.min_aspect_ratio = arg.value.scalar_val;
			}
		}
		// target face count can be specified in two different ways
		else if (arg.name == cap::min_fn) {
			if (arg.value.integer_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				if (!has_reduction)
					fn_target = arg.value.integer_val;
			}
		}
		else if (arg.name == cap::reduction_fn) {
			if (arg.value.integer_val <= 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				has_reduction = true;
				fn_target = int(hi.F.rows() / (Scalar)arg.value.integer_val);
			}
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (fn_target < 0) {
		std::cerr << "ERROR " << cmd.name << ": unspecified argument `" << cap::min_fn
		          << "` or `" << cap::reduction_fn << "`" << std::endl;
		return;
	}

	Timer t;

	proxy.V = hi.V;
	proxy.F = hi.F;

	proxy.VF = hi.VF;// compute_adjacency_vertex_face(proxy.V, proxy.F);

	//proxy.D = hi.D;
	//proxy.DW = hi.DW;
	
	VectorXu8 VB;
	per_vertex_border_flag(proxy.V, proxy.F, proxy.VF, VB);

	std::cout << "Proxy mesh setup: " << t.time_since_last_check() << " seconds" << std::endl;

	// apply topological filtering

#ifndef NO_PREPROCESS

	// get rid of low-valence and non-manifold faces, and crease artifacts
	int n_collapsed = squash_low_valence_vertices(proxy.V, proxy.F, proxy.VF, VB);
	n_collapsed += collapse_onto_creases(proxy.V, proxy.F, proxy.VF, VB, radians(170));
	if (n_collapsed > 0) {
		remove_degenerate_faces_inplace(proxy.F);
		compact_vertex_data(proxy.F, proxy.V, VB, proxy.D, proxy.DW);
		proxy.VF = compute_adjacency_vertex_face(proxy.V, proxy.F);
	}

	int nflip = flip_pass(proxy.V, proxy.F, proxy.VF, VB, proxy.D, proxy.DW);
	std::cout << "Flipped " << nflip << " edges" << std::endl;

	// target edge length for the remeshing pass
	Scalar target_len = 5 * hi.avg_edge_length;

	// select faces with long edges to split
	std::unordered_set<int> remesh_faces;
	for (int fi = 0; fi < proxy.F.rows(); ++fi) {
		if (proxy.F(fi, 0) != INVALID_INDEX) {
			for (int i = 0; i < 3; ++i) {
				Edge e(proxy.F(fi, i), proxy.F(fi, (i + 1) % 3));
				if ((proxy.V.row(e.first) - proxy.V.row(e.second)).norm() > 2 * target_len)
					remesh_faces.insert(fi);
			}
		}
	}

	if (remesh_faces.size() > 0) {
		std::cout << "Remeshing " << remesh_faces.size() << " faces" << std::endl;
		// perform some remeshing operations to get rid of very large triangles
		for (int i = 0; i < 10; ++i) {
			std::cout << "Remeshing (iteration " << i << ")" << std::endl;
			int nsplit = split_pass(proxy.V, proxy.F, proxy.VF, VB, target_len, remesh_faces);
			std::cout << "\tSplit " << nsplit << " edges" << std::endl;

			// Apply topological filtering
			int nflip = flip_pass(proxy.V, proxy.F, proxy.VF, VB, remesh_faces);
			std::cout << "\tFlipped " << nflip << " edges" << std::endl;

			int ncollapse = collapse_pass(proxy.V, proxy.F, proxy.VF, VB, target_len, remesh_faces);
			std::cout << "\tcollapsed " << ncollapse << " edges" << std::endl;
		}
	}

#endif

	remove_degenerate_faces_inplace(proxy.F);
	compact_vertex_data(proxy.F, proxy.V, VB, proxy.D, proxy.DW);
	proxy.VF = compute_adjacency_vertex_face(proxy.V, proxy.F);

	// record face orientations *before* filtering
	proxy.FN = compute_face_normals(proxy.V, proxy.F);

	if (smoothing_iterations > 0)
		for (int i = 0; i < smoothing_iterations; ++i)
			flip_guarded_laplacian_smooth(proxy.V, proxy.F, proxy.VF, VB, proxy.FN);

	if (anisotropic_smoothing_iterations > 0)
		flip_guarded_anisotropic_smoothing(proxy.V, proxy.F, proxy.VF, VB, proxy.FN, anisotropic_smoothing_iterations, dparams.border_error_scaling, anisotropic_smoothing_weight);

	proxy.FR = compute_face_aspect_ratios(proxy.V, proxy.F);
	
	std::cout << "Proxy mesh filtering: " << t.time_since_last_check() << " seconds" << std::endl;

	if (parallel_proxy)
		decimate_mesh_parallel(proxy.V, proxy.F, proxy.VF, fn_target, proxy.Q, proxy.QW, proxy.D, proxy.DW, proxy.FR, proxy.FN, dparams, nw);
	else
		decimate_mesh_fast(proxy.V, proxy.F, proxy.VF, fn_target, proxy.Q, proxy.QW, proxy.D, proxy.DW, proxy.FR, proxy.FN, dparams);
	
	std::cout << "Proxy mesh decimation: " << t.time_since_last_check() << " seconds" << std::endl;

	_proxy_mesh_changed();
}

void Session::_execute_load_base_mesh(const SessionCommand& cmd)
{
	std::cerr << "_execute_load_base_mesh()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::LoadBaseMesh);

	std::string path = "";

	namespace cal = cmdargs::load_base_mesh;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cal::path) {
			path = arg.value.string_val;
		}
		else {
			std::cerr << cmd.name << ": Unrecognized argument " << arg.name << std::endl;
		}
	}

	if (path == "") {
		std::cerr << cmd.name << ": Missing required parameter " << std::quoted(cal::path) << std::endl;
		return;
	}
	
	MatrixX V;
	MatrixXi F;

	bool read_mesh = false;
	std::filesystem::path base_mesh_file(path);
	std::string fmt = lowercase(base_mesh_file.extension().string());

	if (fmt == ".obj") {
		read_mesh = read_obj(base_mesh_file.string(), V, F);
	}
	else if (fmt == ".gltf") {
		GLTFReadInfo read_info;
		read_mesh = read_gltf(base_mesh_file.string(), read_info);

		V = read_info.get_vertices();
		F = read_info.get_faces();
	}

	if (!read_mesh) {
		std::cerr << cmd.name << ": Error reading the specified mesh file " << base_mesh_file << std::endl;
		return;
	}

	MatrixXu8 FEB = per_face_edge_border_flag(V, F);
	VectorXu8 VB = per_vertex_border_flag(V, F, FEB);

	unify_vertices(V, F, VB, FEB);

	compact_vertex_data(F, V, VB);

	VFAdjacency VF = compute_adjacency_vertex_face(V, F);

	std::vector<Quadric> Q;
	std::vector<Scalar> QW;
	compute_quadrics_per_vertex(V, F, VF, VB, Scalar(1), Q, QW);

	MatrixX FN = compute_face_normals(V, F);
	VectorX FR = compute_face_aspect_ratios(V, F);

	MatrixX D;
	VectorX DW;

	DecimationParameters dparams;

	algo.handle->_import_decimation_mesh(V, F, VF, Q, QW, D, DW, FR, FN);
	algo.handle->init(dparams);

	_base_mesh_changed(true);
}

void Session::_execute_decimate(const SessionCommand& cmd)
{
	std::cerr << "_execute_decimate()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::Decimate);

	namespace cad = cmdargs::decimate;

	DecimationParameters dparams;

	bool use_proxy = false;
	bool has_reduction = false;
	int fn_target = 1;

	// configure decimation
	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cad::proxy) {
			use_proxy = arg.value.bool_val;
		}
		else if (arg.name == cad::border_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.border_error_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cad::aspect_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.ar_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cad::visibility_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.visibility_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cad::normals_multiplier) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.normals_scaling = arg.value.scalar_val;
			}
		}
		else if (arg.name == cad::vertex_smoothing) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.use_vertex_smoothing = true;
				dparams.smoothing_coefficient = arg.value.scalar_val;
			}
		}
		else if (arg.name == cad::max_error) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.bound_geometric_error = true;
				dparams.max_relative_error = arg.value.scalar_val;
			}
		}
		else if (arg.name == cad::min_aspect_ratio) {
			if (arg.value.scalar_val < 0 || arg.value.scalar_val > 1) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				dparams.bound_aspect_ratio = true;
				dparams.min_aspect_ratio = arg.value.scalar_val;
			}
		}
		// target face count can be specified in two different ways
		else if (arg.name == cad::min_fn) {
			if (arg.value.integer_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				if (!has_reduction)
					fn_target = arg.value.integer_val;
			}
		}
		else if (arg.name == cad::reduction_fn) {
			if (arg.value.integer_val <= 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				has_reduction = true;
				fn_target = int(hi.F.rows() / (Scalar)arg.value.integer_val);
			}
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (use_proxy) {
		if (proxy.V.rows() > 0) {
			algo.handle->_import_decimation_mesh(proxy.V, proxy.F, proxy.VF, proxy.Q, proxy.QW, proxy.D, proxy.DW, proxy.FR, proxy.FN);
		}
		else {
			std::cerr << cmd.name << ": proxy mesh does not exist, using input" << std::endl;
		}
	}

	// execute
	algo.handle->init(dparams);
	algo.handle->execute(fn_target);

	int nzero = remove_zero_area_faces(base.V, base.F);
	std::cout << "Removed " << nzero << " zero-area faces" << std::endl;
	algo.handle->compact_faces();

	_base_mesh_changed(true);
}

static void cap_subdivision_level(VectorXu8& subdivisions, uint8_t max_level)
{
	std::vector<int> ind = vector_of_indices(subdivisions.size());
	std::sort(ind.begin(), ind.end(), [&](int i, int j) { return subdivisions(i) > subdivisions(j); });

	int n_excess_faces = 0;
	int i = 0;
	while (i < ind.size() && subdivisions(ind[i]) >= max_level) {
		n_excess_faces += (1 << (2 * subdivisions(ind[i]))) - (1 << (2 * max_level));
		subdivisions(ind[i]) = max_level;
		i++;
	}

	while (n_excess_faces > 0 && i < ind.size()) {
		int j = i;
		bool updated = false;

		while (n_excess_faces > 0 && j < ind.size()) {
			uint8_t sj = subdivisions(ind[j]);
			Assert(sj < max_level);
			n_excess_faces += (1 << 2 * sj) - (1 << 2 * (sj + 1));
			subdivisions(ind[j])++;
			if (subdivisions[ind[j]] == max_level)
				i++;
			j++;
		}

		if (!updated)
			break;
	}
}

void Session::_execute_tessellate(const SessionCommand& cmd)
{
	std::cerr << "_execute_tessellate()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::Tessellate);

	std::string mode = "";
	int level = -1;
	int microfn = -1;
	Scalar microexpansion = -1;
	Scalar max_error = -1;

	int max_level = 8;

	namespace cat = cmdargs::tessellate;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cat::mode) {
			mode = arg.value.string_val;
			if (mode != cat::modes::constant && mode != cat::modes::uniform && mode != cat::modes::adaptive) {
				std::cerr << "ERROR " << cmd.name << ": unrecognized tessellation mode " << mode
					<< ". Valid modes are "
					<< std::quoted(cat::modes::constant) << ", "
					<< std::quoted(cat::modes::uniform) << ", "
					<< std::quoted(cat::modes::adaptive) << std::endl;
			}
		}
		else if (arg.name == cat::level) {
			if (arg.value.integer_val < 0 || arg.value.integer_val > 8) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				level = arg.value.integer_val;
			}
		}
		else if (arg.name == cat::max_level) {
			if (arg.value.integer_val < 0 || arg.value.integer_val > 8) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				max_level = arg.value.integer_val;
			}
		}
		else if (arg.name == cat::microfn) {
			if (arg.value.integer_val <= 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				microexpansion = arg.value.integer_val;
			}
		}
		else if (arg.name == cat::microexpansion) {
			if (arg.value.scalar_val < 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				microexpansion = arg.value.scalar_val;
			}
		}
		else if (arg.name == cat::max_error) {
			if (arg.value.scalar_val <= 0) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.scalar_val << std::endl;
			}
			else {
				max_error = arg.value.scalar_val;
			}
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (mode == cat::modes::constant) {
		if (level < 0) {
			std::cerr << "ERROR " << cmd.name << " (" << mode << "): unspecified argument `" << cat::level << "`" << std::endl;
			return;
		}
		else {
			base.subdivisions = compute_subdivision_levels_constant(base.F, uint8_t(level));
		}
	}
	else if (mode == cat::modes::uniform) {
		if (microfn < 0 && microexpansion < 0) {
			std::cerr << "ERROR " << cmd.name << ": unspecified argument `" << cat::microexpansion
			          << "` or `" << cat::microfn << "`" << std::endl;
			return;
		}
		else {
			// microfn has priority over microfn if both are specified
			if (microfn > 0)
				microexpansion = microfn / (Scalar)hi.F.rows();
			else
				Assert(microexpansion > 0);

			Scalar subdivision_level = (std::log2(microexpansion) + std::log2(hi.F.rows() / (Scalar)base.valid_fn)) / 2;
			Scalar avg_area_inv = 1.0 / average_area(base.V, base.F);
			base.subdivisions = compute_subdivision_levels_uniform_area(base.V, base.F, avg_area_inv, subdivision_level);
		}
		cap_subdivision_level(base.subdivisions, uint8_t(max_level));
	}
	else if (mode == cat::modes::adaptive) {
		if (microfn < 0 && microexpansion < 0 && max_error < 0) {
			std::cerr << "ERROR " << cmd.name << ": unspecified argument `" << cat::microexpansion
			          << "` or `" << cat::microexpansion << "` or `" << cat::max_error << "`" << std::endl;
			return;
		}

		max_error *= (max_error < 0) ? 0 : hi.avg_edge_length;

		if (microfn < 0) {
			if (microexpansion > 0) {
				microfn = microexpansion * hi.F.rows();
			}
			else {
				microfn = std::numeric_limits<int>::max();
			}
		}

		Assert(!(max_error == 0 && microfn == std::numeric_limits<int>::max()));

		base.subdivisions = compute_mesh_subdivision_adaptive(base.V, base.VD, base.F, hi.V, hi.F, hi.VN, microfn, max_error);
		cap_subdivision_level(base.subdivisions, uint8_t(max_level));
	}
	else {
		std::cerr << "ERROR " << cmd.name << ": missing required argument `" << cat::mode << "`" << std::endl;
		return;
	}

	base.flags = adjust_subdivision_levels(base.F, base.subdivisions, base.corrections, RoundMode::Up);
	VectorXu8 subdivision_bits = compute_subdivision_bits();
	
	micromesh.compute_mesh_structure(base.V, base.F, subdivision_bits);
}

void Session::_execute_set_displacement_dirs(const SessionCommand& cmd)
{
	std::cerr << "_execute_set_displacement_dirs()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::SetDisplacementDirs);

	namespace cas = cmdargs::set_displacement_dirs;

	std::string type = cas::types::normals;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cas::type) {
			type = arg.value.string_val;
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (type == cas::types::max_visibility) {
		VectorX VIS;
		std::tie(base.VD, VIS) = compute_optimal_visibility_directions(base.V, base.F, base.VF);
		for (int i = 0; i < base.VD.rows(); ++i)
			if (VIS(i) <= 0)
				base.VD.row(i).setZero();
	}
	else if (type == cas::types::normals) {
		base.VD = base.VN;
	}
	else if (type == cas::types::tangent) {
		MatrixX border_FN = compute_face_normals(border.V, border.F);
		BVHTree bvh_border;
		bvh_border.build_tree(&border.V, &border.F, &border.VN);

		base.VD = base.VN;
		VectorXu8 base_VB;
		per_vertex_border_flag(base.V, base.F, base_VB);
		for (int vi = 0; vi < base.V.rows(); ++vi) {
			if (base_VB(vi)) {
				NearestInfo ni;
				bool hit = bvh_border.nearest_point(base.V.row(vi), &ni);
				Assert(hit);
				base.VD.row(vi) = -border_FN.row(ni.fi);
			}
		}
	}
	else {
		std::cerr << "ERROR " << cmd.name << ": unrecognized displacement type " << type
			<< ". Valid types are "
			<< std::quoted(cas::types::max_visibility) << ", "
			<< std::quoted(cas::types::normals) << ", "
			<< std::quoted(cas::types::tangent) << std::endl;
	}
}

void Session::_execute_displace(const SessionCommand& cmd)
{
	std::cerr << "_execute_displace()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::Displace);

	namespace cad = cmdargs::displace;
	
	for (const CommandArgument& arg : cmd.args) {
		std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
	}

	micromesh.compute_micro_displacements(base.V, base.VD, base.F, hi.bvh, hi.bvh_test, border.V, border.F, border.VN);
}

void Session::_execute_tweak_tessellation(const SessionCommand& cmd)
{
	std::cerr << "_execute_tweak_tessellation()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::TweakTessellation);

	namespace ctt = cmdargs::tweak_tessellation;

	int base_fi = -1;
	int delta = 0;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == ctt::base_fi) {
			if (arg.value.integer_val < 0 || arg.value.integer_val >= int(base.F.rows())) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				base_fi = arg.value.integer_val;
			}
		}
		else if (arg.name == ctt::delta) {
			delta = arg.value.integer_val;
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (base_fi < 0) {
		std::cerr << "ERROR " << cmd.name << ": missing " << ctt::base_fi << " argument" << std::endl;
		return;
	}

	RoundMode rm = delta > 0 ? RoundMode::Up : RoundMode::Down;

	base.corrections(base_fi) += int8_t(delta);
	base.flags = adjust_subdivision_levels(base.F, base.subdivisions, base.corrections, rm);

	VectorXu8 subdivision_bits = compute_subdivision_bits();
	micromesh.update_mesh_structure(subdivision_bits, base.V, base.F, hi.bvh, hi.bvh_test, border.V, border.F, border.VN);
}

void Session::_execute_tweak_displacement_dir(const SessionCommand& cmd)
{
	std::cerr << "_execute_tweak_displacement_dir()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::TweakDisplacementDir);

	int base_vi = -1;
	Vector3 dir = Vector3::Zero();
	bool has_dir = false;

	namespace ctd = cmdargs::tweak_displacement_dir;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == ctd::base_vi) {
			if (arg.value.integer_val < 0 || arg.value.integer_val >= int(base.VD.rows())) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				base_vi = arg.value.integer_val;
			}
		}
		else if (arg.name == ctd::direction) {
			has_dir = true;
			for (int i = 0; i < 3; ++i)
				dir(i) = Scalar(arg.value.vector3_val[i]);
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}
	
	if (base_vi < 0) {
		std::cerr << "ERROR " << cmd.name << ": missing " << ctd::base_vi << " argument" << std::endl;
		return;
	}

	if (!has_dir) {
		std::cerr << "ERROR " << cmd.name << ": missing " << ctd::direction << " argument" << std::endl;
		return;
	}

	base.VD.row(base_vi) = dir;

	if (micromesh.is_displaced()) {
		VectorXi update = VectorXi::Constant(base.F.rows(), 0);
		for (const VFEntry vfe : base.VF[base_vi])
			update(vfe.first) = 1;

		micromesh.compute_micro_displacements(hi.bvh, hi.bvh_test, base.V, base.VD, base.F, update, border.V, border.F, border.VN);
	}
}

void Session::_execute_optimize_base_topology(const SessionCommand& cmd)
{
	std::cerr << "_execute_optimize_base_topology()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::OptimizeBaseTopology);

	if (base.valid_fn == 0) {
		std::cerr << "ERROR " << cmd.name << ": base mesh does not exist yet." << std::endl;
		return;
	}

	VectorXu8 base_VB;
	per_vertex_border_flag(base.V, base.F, base_VB);

	flip_to_reduce_error(base.V, base.F, base.VF, base_VB, hi.V, hi.F);
}

void Session::_execute_optimize_base_positions(const SessionCommand& cmd)
{
	std::cerr << "_execute_optimize_base_positions()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::OptimizeBasePositions);

	namespace cobp = cmdargs::optimize_base_positions;

	std::string mode = "";

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cobp::mode) {
			mode = arg.value.string_val;
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (mode == cobp::modes::least_squares) {
		base.V = optimize_base_mesh_positions_ls(base.V, base.F, base.VN, hi.V, hi.F, hi.VN);
		_base_mesh_changed(true);
	}
	else if (mode == cobp::modes::reprojected_quadrics) {
		algo.handle->_refresh_quadrics();
		_base_mesh_changed(true);
	}
	else if (mode == cobp::modes::clear_smoothing_term) {
		algo.handle->_move_vertices_in_quadrics_min();
		_base_mesh_changed(true);
	}
	else {
		std::cerr << "ERROR " << cmd.name << ": missing or invalid" << cobp::mode << " argument" << std::endl;
		return;
	}
}

void Session::_execute_minimize_prismoids(const SessionCommand& cmd)
{
	std::cerr << "_execute_minimize_prismoids()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::MinimizePrismoids);

	if (!micromesh.is_displaced()) {
		std::cerr << "ERROR " << cmd.name << ": micromesh is not displaced" << std::endl;
		return;
	}

	MatrixX new_V;
	MatrixX new_VD;

	micromesh.normalize_displacements(base.V, base.VD, base.F, new_V, new_VD);
	
	// minimize the displacements, writing new displacement vectors into mesh_data.VD
	micromesh.minimize_base_directions_length(base.V, base.VD, base.F, new_V, new_VD, base.top_VD, base.bottom_VD);

	base.V = new_V;
	base.VD = new_VD;

	auto reproject = [&](SubdivisionTri& face) -> void {
		int base_fi = face.base_fi;

		// target positions (old microvertex positions)
		MatrixX V_old = face.V + face.VD;
		MatrixX VN_old = face.VN;

		// subdivide with new base coordinates, leaving the subdivision level/bits the same
		face.subdivide(face.subdivision_bits, base.V.row(base.F(base_fi, 0)), base.V.row(base.F(base_fi, 1)), base.V.row(base.F(base_fi, 2)));
		face.VN = VN_old;

		// set the new direction vectors
		for (int i = 0; i < 3; ++i)
			face.base_VD.row(i) = base.VD.row(base.F(base_fi, i));

		// note that base.V and base.VD at this point encode a 'volume' that should fully contain the micro-surface

		// for each microvertex, compute the new displacement by projecting the old point along
		// the new displacement ray

		BarycentricGrid grid(face.subdivision_level());
		for (int uvi = 0; uvi < face.V.rows(); ++uvi) {
			Vector3 w = grid.barycentric_coord(uvi);
			Vector3 d = face.interpolate_direction(w);
			Scalar t = project_onto_ray(V_old.row(uvi), face.V.row(uvi), d);
			t = clamp(t, Scalar(0), Scalar(1));
			face.VD.row(uvi) = t * d;
		}
	};
	std::for_each(std::execution::par_unseq, micromesh.faces.begin(), micromesh.faces.end(), reproject);
}

void Session::_execute_reset_tessellation_offsets(const SessionCommand& cmd)
{
	std::cerr << "_execute_reset_tessellation_offsets()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::ResetTessellationOffsets);

	base.corrections = VectorXi8::Constant(base.F.rows(), 0);
}

void Session::_execute_flip_edge(const SessionCommand& cmd)
{
	std::cerr << "_execute_flip_edge()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::FlipEdge);

	namespace cfe = cmdargs::flip_edge;

	int base_fi = -1;
	int edge = -1;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cfe::base_fi) {
			if (arg.value.integer_val < 0 || arg.value.integer_val >= int(base.F.rows())) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				base_fi = arg.value.integer_val;
			}
		}
		else if (arg.name == cfe::edge) {
			if (arg.value.integer_val < 0 || arg.value.integer_val > 2) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				edge = arg.value.integer_val;
			}
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (base_fi < 0) {
		std::cerr << "ERROR " << cmd.name << ": missing " << cfe::base_fi << " argument" << std::endl;
		return;
	}

	if (edge < 0) {
		std::cerr << "ERROR " << cmd.name << ": missing " << cfe::edge << " argument" << std::endl;
		return;
	}

	Edge e(base.F(base_fi, edge), base.F(base_fi, (edge + 1) % 3));
	Flap flap = compute_flap(e, base.F, base.VF);
	if (flip_preserves_topology(flap, base.F, base.VF)) {
		FlipInfo flip = flip_edge(flap, base.V, base.F, base.VF);
		Assert(flip.ok);
		_base_mesh_changed(true);
	}
}

void Session::_execute_split_edge(const SessionCommand& cmd)
{
	std::cerr << "_execute_split_edge()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::SplitEdge);

	namespace cse = cmdargs::split_edge;

	int base_fi = -1;
	int edge = -1;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cse::base_fi) {
			if (arg.value.integer_val < 0 || arg.value.integer_val >= int(base.F.rows())) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				base_fi = arg.value.integer_val;
			}
		}
		else if (arg.name == cse::edge) {
			if (arg.value.integer_val < 0 || arg.value.integer_val > 2) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				edge = arg.value.integer_val;
			}
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (base_fi < 0) {
		std::cerr << "ERROR " << cmd.name << ": missing " << cse::base_fi << " argument" << std::endl;
		return;
	}

	if (edge < 0) {
		std::cerr << "ERROR " << cmd.name << ": missing " << cse::edge << " argument" << std::endl;
		return;
	}

	Edge e(base.F(base_fi, edge), base.F(base_fi, (edge + 1) % 3));
	Flap flap = compute_flap(e, base.F, base.VF);

	SplitInfo si = split_edge(flap, base.V, base.F, base.VF, algo.handle->dcb->VB);
	if (!si.ok) {
		std::cerr << "Cannot split non-manifold edge" << std::endl;
	}
	else {
		algo.handle->manage_split(si);
		_base_mesh_changed(true);
	}
}

void Session::_execute_move_vertex(const SessionCommand& cmd)
{
	std::cerr << "_execute_move_vertex()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::MoveVertex);

	int base_vi = -1;
	Vector3 pos = Vector3::Zero();
	bool has_pos = false;

	namespace cmv = cmdargs::move_vertex;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cmv::base_vi) {
			if (arg.value.integer_val < 0 || arg.value.integer_val >= int(base.V.rows())) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				base_vi = arg.value.integer_val;
			}
		}
		else if (arg.name == cmv::position) {
			has_pos = true;
			for (int i = 0; i < 3; ++i)
				pos(i) = Scalar(arg.value.vector3_val[i]);
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}
	
	if (!has_pos) {
		std::cerr << "ERROR " << cmd.name << ": missing " << cmv::position << " argument" << std::endl;
		return;
	}

	base.V.row(base_vi) = pos;
	_base_mesh_changed(true);
}

void Session::_execute_split_vertex(const SessionCommand& cmd)
{
	std::cerr << "_execute_split_vertex()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::SplitVertex);

	int base_vi = -1;

	namespace cms = cmdargs::split_vertex;

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cms::base_vi) {
			if (arg.value.integer_val < 0 || arg.value.integer_val >= int(base.V.rows())) {
				std::cerr << cmd.name << ": out-of-range value for argument " << arg.name << " " << arg.value.integer_val << std::endl;
			}
			else {
				base_vi = arg.value.integer_val;
			}
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	VFEntry vfe = *(base.VF[base_vi].begin());

	std::vector<EFEntry> face_circle = compute_face_circle(EFEntry(vfe.fi(), vfe.vi()), base.V, base.F, base.VF);

	int fn = face_circle.size();

	int best_i = -1;
	int best_fn = -1;
	Scalar best_visibility = 0;

	std::vector<Vector3> area_vectors;
	for (const EFEntry& ef : face_circle) {
		area_vectors.push_back(compute_area_vector(
			base.V.row(base.F(ef.fi(), ef.ei(0))),
			base.V.row(base.F(ef.fi(), ef.ei(1))),
			base.V.row(base.F(ef.fi(), ef.ei(2)))));
	}

	for (int i = 0; i < fn; ++i) { // for all possible starting positions
		for (int fni = 1; fni < fn - 1; ++fni) { // for all possible cardinalities of the set pairs 
			std::vector<Vector3> area_vectors1;
			std::vector<Vector3> area_vectors2;
			int j = 0;
			for (; j < fni; ++j)
				area_vectors1.push_back(area_vectors[(i + j) % fn]);
			for (; j < fn; ++j)
				area_vectors2.push_back(area_vectors[(i + j) % fn]);

			Vector3 n1, n2;
			Scalar vis1, vis2;
			std::tie(n1, vis1) = compute_positive_visibility_from_directions(area_vectors1);
			std::tie(n2, vis2) = compute_positive_visibility_from_directions(area_vectors2);

			Scalar vis = std::min(vis1, vis2);

			if (std::min(vis1, vis2) > best_visibility) {
				best_i = i;
				best_fn = fni;
				best_visibility = vis;
			}
		}
	}

	int i1 = best_i;
	int i2 = (best_i + best_fn) % fn;
	Edge e1(base.F(face_circle[i1].fi(), face_circle[i1].ei()), base.F(face_circle[i1].fi(), face_circle[i1].ei(1)));
	Edge e2(base.F(face_circle[i2].fi(), face_circle[i2].ei()), base.F(face_circle[i2].fi(), face_circle[i2].ei(1)));

	VertexSplitInfo vsi = split_vertex(base_vi, e1, e2, base.V, base.F, base.VF, algo.handle->dcb->VB);
	if (!vsi.ok) {
		std::cerr << "Cannot split non-manifold vertex" << std::endl;
	}
	else {
		// apply a very small displacement along neighboring edges to ensure the new faces are not zero area
		Vector3 d_old = base.V.row(base.F(face_circle[i1].fi(), face_circle[i1].ei(2))) - base.V.row(vsi.old_vertex);
		Vector3 d_new = base.V.row(base.F(face_circle[i2].fi(), face_circle[i2].ei(2))) - base.V.row(vsi.new_vertex);
		//base.V.row(vsi.old_vertex) = base.V.row(vsi.old_vertex) + 0.01 * d_old.transpose();
		//base.V.row(vsi.new_vertex) = base.V.row(vsi.new_vertex) + 0.01 * d_new.transpose();
		algo.handle->manage_vertex_split(vsi);
		_base_mesh_changed(true);
	}
}

void Session::_execute_save(const SessionCommand& cmd) const
{
	std::cerr << "_execute_save()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::Save);

	namespace cas = cmdargs::save;

	std::string tag = "";

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cas::tag) {
			tag = arg.value.string_val;
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (tag == "") {
		std::cerr << "ERROR " << cmd.name << ": missing " << cas::tag << " argument." << std::endl;
		return;
	}

	std::filesystem::path out_path = tag + save_prefix + current_mesh.filename().string();
	out_path.replace_extension("gltf");

	GLTFWriteInfo write_info;
	write_info
		.write_faces(&base.F)
		.write_vertices(&base.V)
		.write_normals(&base.VN);

	write_gltf(out_path.string(), write_info);
}

void Session::_execute_save_stats(const SessionCommand& cmd)
{
	std::cerr << "_execute_save_stats()" << std::endl;

	Assert(Commands.find(cmd.name) != Commands.end());
	Assert(Commands[cmd.name] == CommandType::SaveStats);

	if (!micromesh.is_subdivided() || !micromesh.is_displaced()) {
		std::cerr << "ERROR " << cmd.name << ": micromesh does not exist" << std::endl;
		return;
	}

	namespace cas = cmdargs::save_stats;

	std::string tag = "";

	for (const CommandArgument& arg : cmd.args) {
		if (arg.name == cas::tag) {
			tag = arg.value.string_val;
		}
		else {
			std::cerr << cmd.name << ": unrecognized argument " << arg.name << std::endl;
		}
	}

	if (tag == "") {
		std::cerr << "ERROR " << cmd.name << ": missing " << cas::tag << " argument." << std::endl;
		return;
	}

	std::filesystem::path out_path = tag + save_prefix + current_mesh.filename().string();
	out_path.replace_extension("stats");

	// Compute geometric error

	MatrixX V;
	MatrixXi F;

	Distribution d_error;
	Distribution d_isotropy;

	micromesh.extract_mesh(V, F);

	// just to be safe and avoid possible issues when computing the bvh for the
	// exploded micromesh, remove any zero-area face
	int nzero = remove_zero_area_faces(V, F);
	if (nzero > 0) {
		std::cout << "Micromesh had " << nzero << " zero area faces!" << std::endl;
		remove_degenerate_faces_inplace(F);
	}

	MatrixX VN = compute_vertex_normals(V, F);
	compute_vertex_quality_hausdorff_distance(hi.V, hi.VN, hi.F, V, VN, F, hi.VQ);
	for (int vi = 0; vi < (int)hi.VQ.rows(); ++vi)
		d_error.add(hi.VQ(vi));

	Histogram h_error(0, d_error.percentile(0.99), 256);
	VectorX vertex_areas = compute_voronoi_vertex_areas(hi.V, hi.F);
	for (int vi = 0; vi < (int)hi.VQ.rows(); ++vi)
		if (std::isfinite(hi.VQ(vi)) && std::isfinite(vertex_areas(vi)))
			if (hi.VQ(vi) <= d_error.percentile(0.99))
				h_error.add(hi.VQ(vi), vertex_areas(vi));

	// aspect ratio
	micromesh.compute_face_quality_aspect_ratio();
	Histogram h_isotropy(0, 1, 256);
	for (const SubdivisionTri& uface : micromesh.faces) {
		for (int ufi = 0; ufi < uface.F.rows(); ++ufi) {
			if (std::isfinite(uface.FQ(ufi))) {
				h_isotropy.add(uface.FQ(ufi), triangle_area(
					Vector3(uface.V.row(uface.F(ufi, 0)) + uface.VD.row(uface.F(ufi, 0))),
					Vector3(uface.V.row(uface.F(ufi, 1)) + uface.VD.row(uface.F(ufi, 1))),
					Vector3(uface.V.row(uface.F(ufi, 2)) + uface.VD.row(uface.F(ufi, 2)))));
				d_isotropy.add(uface.FQ(ufi));
			}
			else {
				std::cout << "NAN ISOTROPY AT BASE FACE " << uface.base_fi << std::endl;
			}
		}
	}

	// input:vn, fn, size
	// output: udisp encoding, size, time, base_fn, ufn, isotropy, error
	std::ofstream ofs(out_path);
	ofs << "in_vn " << hi.VN.rows() << std::endl;
	ofs << "in_fn " << hi.F.rows() << std::endl;
	std::size_t in_sz = 3 * sizeof(float) * hi.V.rows() + 3 * sizeof(int) * hi.F.rows();
	ofs << "in_size " << in_sz << std::endl;

	{
		std::pair<MatrixX, VectorX> P = compute_optimal_visibility_directions(hi.V, hi.F, hi.VF);
		int in_negative_vis = 0;
		for (int i = 0; i < P.second.rows(); ++i) {
			if (P.second(i) <= 0)
				in_negative_vis++;
		}
		ofs << "in_negative_vis " << in_negative_vis << std::endl;
	}

	// base data: xyz + dir per vertex, plus the index
	std::size_t out_sz = 6 * sizeof(float) * base.V.rows() + 3 * sizeof(int) * base.F.rows();
	uint32_t micro_vn = 0;

	// umesh data: 1 byte per base face (topology) plus the microdisplacements

	Bary bary;
	extract_displacement_bary_data(micromesh, &bary, true);
	out_sz += bary.data.valuesInfo.valueCount; // unorm11_pack32 data size in bytes

	for (const SubdivisionTri& st : micromesh.faces) {
		out_sz += 1; // base topology
		micro_vn += st.V.rows();
	}

	ofs << "out_udisp_encoding UNORM11_PACK32" << std::endl;
	//ofs << "out_udisp_size " << 2 << std::endl; // 2 bytes per microdisplacement
	ofs << "out_size " << out_sz << std::endl;
	ofs << "base_fn " << base.F.rows() << std::endl;
	ofs << "micro_fn " << micromesh.micro_fn << std::endl;
	ofs << "micro_vn " << micro_vn << std::endl;
	ofs << "time " << _timers.t_proc.time_elapsed() << std::endl;

	std::vector<int>percentiles = { 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99 };

	ofs << "isotropy_min " << h_isotropy.min() << std::endl;
	ofs << "isotropy_max " << h_isotropy.max() << std::endl;
	ofs << "isotropy_avg " << h_isotropy.avg() << std::endl;

	for (int p : percentiles) {
		ofs << "isotropy_p" << p << " " << d_isotropy.percentile(p / Scalar(100)) << std::endl;
	}

	ofs << "error_min " << d_error.min() << std::endl; // use distribution for min/max
	ofs << "error_max " << d_error.max() << std::endl;
	ofs << "error_avg " << h_error.avg() << std::endl;

	for (int p : percentiles) {
		ofs << "error_p" << p << " " << d_error.percentile(p / Scalar(100)) << std::endl;
	}
	
}


