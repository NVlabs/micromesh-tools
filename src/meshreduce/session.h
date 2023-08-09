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
#include "decimation.h"
#include "adjacency.h"
#include "aabb.h"
#include "bvh.h"
#include "micro.h"

#include <map>
#include <vector>
#include <set>
#include <filesystem>

//#define CMDARG_STRING(argname, strval) {.name = argname, .value = {.string_val = strval}}
#define CMDARG_INTEGER(argname, val) {.name = argname, .value = {.integer_val = val}}
#define CMDARG_SCALAR(argname, val) {.name = argname, .value = {.scalar_val = val}}
#define CMDARG_VEC3(argname, vx, vy, vz) {.name = argname, .value = {.vector3_val = {vx, vy, vz}}}
#define CMDARG_BOOL(argname, val) {.name = argname, .value = {.bool_val = val}}

struct CommandArgument {
	static constexpr unsigned MAX_STRING_SIZE = 256;
	std::string name;
	union {
		char string_val[MAX_STRING_SIZE] = {};
		double scalar_val;
		long integer_val;
		double vector3_val[3];
		bool bool_val;
	} value;
};

enum CommandType {
	InitProxy,
	LoadBaseMesh,
	Decimate,
	Tessellate,
	SetDisplacementDirs,
	Displace,
	TweakTessellation,
	TweakDisplacementDir,
	OptimizeBasePositions,
	OptimizeBaseTopology,
	MinimizePrismoids,
	ResetTessellationOffsets,
	FlipEdge,
	SplitEdge,
	MoveVertex,
	SplitVertex,
	Save,
	SaveStats
};

struct SessionCommand {
	std::string name;
	std::vector<CommandArgument> args;

	bool from_string(const std::string& s);
	std::string to_string() const;

	void _parse_args(const std::vector<std::string>& tokens);
};

typedef std::map<std::string, CommandType> CommandMap;
typedef std::map<CommandType, std::string> CommandString;

static CommandString CommandStrings = {
	{InitProxy, "init_proxy"},
	{LoadBaseMesh, "load_base_mesh"},
	{Decimate, "decimate"},
	{Tessellate, "tessellate"},
	{SetDisplacementDirs, "set_displacement_dirs"},
	{Displace, "displace"},
	{TweakTessellation, "tweak_tessellation"},
	{TweakDisplacementDir, "tweak_displacement_dir"},
	{OptimizeBaseTopology, "optimize_base_topology"},
	{OptimizeBasePositions, "optimize_base_positions"},
	{MinimizePrismoids, "minimize_prismoids"},
	{ResetTessellationOffsets, "reset_tessellation_offsets"},
	{FlipEdge, "flip_edge"},
	{SplitEdge, "split_edge"},
	{MoveVertex, "move_vertex"},
	{SplitVertex, "split_vertex"},
	{Save, "save"},
	{SaveStats, "save_stats"}
};

static CommandMap Commands = {
	{CommandStrings[InitProxy], InitProxy},
	{CommandStrings[LoadBaseMesh], LoadBaseMesh},
	{CommandStrings[Decimate], Decimate},
	{CommandStrings[Tessellate], Tessellate},
	{CommandStrings[SetDisplacementDirs], SetDisplacementDirs},
	{CommandStrings[Displace], Displace},
	{CommandStrings[TweakTessellation], TweakTessellation},
	{CommandStrings[TweakDisplacementDir], TweakDisplacementDir},
	{CommandStrings[OptimizeBaseTopology], OptimizeBaseTopology},
	{CommandStrings[OptimizeBasePositions], OptimizeBasePositions},
	{CommandStrings[MinimizePrismoids], MinimizePrismoids},
	{CommandStrings[ResetTessellationOffsets], ResetTessellationOffsets},
	{CommandStrings[FlipEdge], FlipEdge},
	{CommandStrings[SplitEdge], SplitEdge},
	{CommandStrings[MoveVertex], MoveVertex},
	{CommandStrings[SplitVertex], SplitVertex},
	{CommandStrings[Save], Save},
	{CommandStrings[SaveStats], SaveStats}
};

enum ArgType {
	StringT,
	ScalarT,
	IntegerT,
	Vector3T,
	BoolT
};

typedef std::map<std::string, ArgType> ArgTypeMap;
typedef std::map<std::string, ArgTypeMap> CommandArgsMap;

#define ARGDECL(argname, type) {argname, type}

namespace cmdargs {
	namespace init_proxy {
		const std::string smoothing_iterations = "smoothing_iterations";
		const std::string anisotropic_smoothing_iterations = "anisotropic_smoothing_iterations";
		const std::string anisotropic_smoothing_weight = "anisotropic_smoothing_weight";

		const std::string vertex_smoothing = "vertex_smoothing";

		const std::string border_multiplier = "border_multiplier";
		const std::string aspect_multiplier = "aspect_multiplier";
		const std::string visibility_multiplier = "visibility_multiplier";
		const std::string normals_multiplier = "normals_multiplier";

		const std::string max_error = "max_error";
		const std::string min_aspect_ratio = "min_aspect_ratio";
		const std::string min_fn = "min_fn";
		const std::string reduction_fn = "reduction_fn";
	}

	namespace load_base_mesh {
		const std::string path = "path";
	}

	namespace decimate {
		const std::string proxy = "proxy";
		
		const std::string vertex_smoothing = "vertex_smoothing";

		const std::string border_multiplier = "border_multiplier";
		const std::string aspect_multiplier = "aspect_multiplier";
		const std::string visibility_multiplier = "visibility_multiplier";
		const std::string normals_multiplier = "normals_multiplier";

		const std::string max_error = "max_error";
		const std::string min_aspect_ratio = "min_aspect_ratio";
		const std::string min_fn = "min_fn";
		const std::string reduction_fn = "reduction_fn";
	}

	namespace tessellate {
		const std::string mode = "mode";
		const std::string level = "level";
		const std::string max_level = "max_level";
		const std::string microexpansion = "microexpansion";
		const std::string microfn = "microfn";
		const std::string max_error = "max_error";

		namespace modes {
			const std::string constant = "constant";
			const std::string uniform = "uniform";
			const std::string adaptive = "adaptive";
		}
	}

	namespace set_displacement_dirs {
		const std::string type = "type";
		namespace types {
			const std::string max_visibility = "max_visibility";
			const std::string normals = "normals";
			const std::string tangent = "tangent";
		}
	}

	namespace displace {
	}

	namespace tweak_tessellation {
		const std::string base_fi = "base_fi";
		const std::string delta = "delta";
	}

	namespace tweak_displacement_dir {
		const std::string base_vi = "base_vi";
		const std::string direction = "direction";
	}

	namespace optimize_base_topology {

	}

	namespace optimize_base_positions {
		const std::string mode = "mode";
		namespace modes {
			const std::string least_squares = "global_least_squares";
			const std::string reprojected_quadrics = "reprojected_quadrics";
			const std::string clear_smoothing_term = "clear_smoothing_term";
		}
	}

	namespace minimize_prismoids {

	}

	namespace reset_tessellation_offsets {

	}

	namespace flip_edge {
		const std::string base_fi = "base_fi";
		const std::string edge = "edge";
	}

	namespace split_edge {
		const std::string base_fi = "base_fi";
		const std::string edge = "edge";
	}

	namespace move_vertex {
		const std::string base_vi = "base_vi";
		const std::string position = "position";
	}

	namespace split_vertex {
		const std::string base_vi = "base_vi";
	}

	namespace save {
		const std::string tag = "tag";
	}

	namespace save_stats {
		const std::string tag = "tag";
	}
}

static CommandArgsMap CommandArgTypes = {
	{
		CommandStrings[InitProxy], {
			ARGDECL(cmdargs::init_proxy::smoothing_iterations, IntegerT),
			ARGDECL(cmdargs::init_proxy::anisotropic_smoothing_iterations, IntegerT),
			ARGDECL(cmdargs::init_proxy::anisotropic_smoothing_weight, ScalarT),
			
			ARGDECL(cmdargs::init_proxy::vertex_smoothing, ScalarT),
			ARGDECL(cmdargs::init_proxy::border_multiplier, ScalarT),
			ARGDECL(cmdargs::init_proxy::aspect_multiplier, ScalarT),
			ARGDECL(cmdargs::init_proxy::visibility_multiplier, ScalarT),
			ARGDECL(cmdargs::init_proxy::normals_multiplier, ScalarT),
			ARGDECL(cmdargs::init_proxy::max_error, ScalarT),
			ARGDECL(cmdargs::init_proxy::min_aspect_ratio, ScalarT),
			ARGDECL(cmdargs::init_proxy::min_fn, IntegerT),
			ARGDECL(cmdargs::init_proxy::reduction_fn, IntegerT)
		}
	},
	{
		CommandStrings[LoadBaseMesh], {
			ARGDECL(cmdargs::load_base_mesh::path, StringT)
		}
	},
	{
		CommandStrings[Decimate], {
			ARGDECL(cmdargs::decimate::proxy, BoolT),
			ARGDECL(cmdargs::decimate::vertex_smoothing, ScalarT),
			ARGDECL(cmdargs::decimate::border_multiplier, ScalarT),
			ARGDECL(cmdargs::decimate::aspect_multiplier, ScalarT),
			ARGDECL(cmdargs::decimate::visibility_multiplier, ScalarT),
			ARGDECL(cmdargs::decimate::normals_multiplier, ScalarT),
			ARGDECL(cmdargs::decimate::max_error, ScalarT),
			ARGDECL(cmdargs::decimate::min_aspect_ratio, ScalarT),
			ARGDECL(cmdargs::decimate::min_fn, IntegerT),
			ARGDECL(cmdargs::decimate::reduction_fn, IntegerT)
		}
	},
	{
		CommandStrings[Tessellate], {
			ARGDECL(cmdargs::tessellate::mode, StringT),
			ARGDECL(cmdargs::tessellate::level, IntegerT),
			ARGDECL(cmdargs::tessellate::max_level, IntegerT),
			ARGDECL(cmdargs::tessellate::microexpansion, ScalarT),
			ARGDECL(cmdargs::tessellate::microfn, IntegerT),
			ARGDECL(cmdargs::tessellate::max_error, ScalarT)
		}
	},
	{
		CommandStrings[SetDisplacementDirs], {
			ARGDECL(cmdargs::set_displacement_dirs::type, StringT)
		}
	},
	{
		CommandStrings[Displace], {
		}
	},
	{
		CommandStrings[TweakTessellation], {
			ARGDECL(cmdargs::tweak_tessellation::base_fi, IntegerT),
			ARGDECL(cmdargs::tweak_tessellation::delta, IntegerT)
		}
	},
	{
		CommandStrings[TweakDisplacementDir], {
			ARGDECL(cmdargs::tweak_displacement_dir::base_vi, IntegerT),
			ARGDECL(cmdargs::tweak_displacement_dir::direction, Vector3T)
		}
	},
	{
		CommandStrings[OptimizeBaseTopology], {
		}
	},
	{
		CommandStrings[OptimizeBasePositions], {
			ARGDECL(cmdargs::optimize_base_positions::mode, StringT)
		}
	},
	{
		CommandStrings[MinimizePrismoids], {
		}
	},
	{
		CommandStrings[ResetTessellationOffsets], {
		}
	},
	{
		CommandStrings[FlipEdge], {
			ARGDECL(cmdargs::flip_edge::base_fi, IntegerT),
			ARGDECL(cmdargs::flip_edge::edge, IntegerT)
		}
	},
	{
		CommandStrings[SplitEdge], {
			ARGDECL(cmdargs::split_edge::base_fi, IntegerT),
			ARGDECL(cmdargs::split_edge::edge, IntegerT)
		}
	},
	{
		CommandStrings[MoveVertex], {
			ARGDECL(cmdargs::move_vertex::base_vi, IntegerT),
			ARGDECL(cmdargs::move_vertex::position, Vector3T)
		}
	},
	{
		CommandStrings[SplitVertex], {
			ARGDECL(cmdargs::split_vertex::base_vi, IntegerT),
		}
	},
	{
		CommandStrings[Save], {
			ARGDECL(cmdargs::save::tag, StringT)
		}
	},
	{
		CommandStrings[SaveStats], {
			ARGDECL(cmdargs::save_stats::tag, StringT)
		}
	}
};

struct CommandSequence {
	std::vector<SessionCommand> commands;

	void add_command(const SessionCommand& cmd);
	void clear_commands();

	bool read_from_file(const std::string& file_path);
	bool write_to_file(const std::string& file_path) const;
	std::string to_string() const;
};

struct Session {

	std::filesystem::path current_mesh;

	CommandSequence sequence;

	std::string save_prefix = "";

	bool parallel_proxy = false;
	int nw = 4;

	struct {
		std::shared_ptr<Decimation> handle = nullptr;
	} algo;

	struct {
		MatrixX V;
		MatrixXi F;
		MatrixX VN;
		VectorX VQ;
		VFAdjacency VF;

		Box3 box;
		Scalar avg_edge_length;

		MatrixX D;
		VectorX DW;

		BVHTree bvh;
		BVHTree::IntersectionFilter bvh_test;
	} hi;

	struct {
		MatrixX V;
		MatrixXi F;
		MatrixX VN;
	} border;

	struct {
		MatrixX V;
		MatrixXi F;
		MatrixX VN;
		VFAdjacency VF;
		std::vector<Quadric> Q;
		std::vector<Scalar> QW;

		VectorX FR;
		MatrixX FN;

		MatrixX D;
		VectorX DW;
	} proxy;

	struct {
		MatrixX V;
		MatrixXi F;
		MatrixX VN;
		MatrixX VD;
		VFAdjacency VF;

		MatrixX D;
		VectorX DW;

		VectorX FR;
		MatrixX FN;

		long valid_fn;

		MatrixX top_VD;
		MatrixX bottom_VD;
		
		Scalar subdivision_level = 0;

		// the subdivision level 'baseline', can have arbitrary jumps
		VectorXu8 subdivisions;

		// the subdivision level corrections, used to manipulate levels and ensure watertightness
		VectorXi8 corrections;

		// the edge decimation flags
		VectorXu8 flags;
		
		// The subdivision bits are obtained as B(i) = (uint8_t(subdivision(i) + correction(i)) << 3) | flags(i)
	} base;

	SubdivisionMesh micromesh;

	struct {
		Timer t_session;
		Timer t_proc;
	} _timers;

	bool load_mesh(const std::filesystem::path& mesh_path);

	bool read_commands(const std::string& cmd_file_path);

	void add_command(const SessionCommand& cmd);
	void clear_commands();

	void execute();
	void execute(const SessionCommand& cmd);

	void _base_mesh_changed(bool reset_directions);
	void _proxy_mesh_changed();

	VectorXu8 compute_subdivision_bits() const;

	void _execute_init_proxy(const SessionCommand& cmd);
	void _execute_load_base_mesh(const SessionCommand& cmd);
	void _execute_decimate(const SessionCommand& cmd);
	void _execute_tessellate(const SessionCommand& cmd);
	void _execute_set_displacement_dirs(const SessionCommand& cmd);
	void _execute_displace(const SessionCommand& cmd);
	void _execute_tweak_tessellation(const SessionCommand& cmd);
	void _execute_tweak_displacement_dir(const SessionCommand& cmd);
	void _execute_optimize_base_topology(const SessionCommand& cmd);
	void _execute_optimize_base_positions(const SessionCommand& cmd);
	void _execute_minimize_prismoids(const SessionCommand& cmd);
	void _execute_reset_tessellation_offsets(const SessionCommand& cmd);
	void _execute_flip_edge(const SessionCommand& cmd);
	void _execute_split_edge(const SessionCommand& cmd);
	void _execute_move_vertex(const SessionCommand& cmd);
	void _execute_split_vertex(const SessionCommand& cmd);
	void _execute_save(const SessionCommand& cmd) const;
	void _execute_save_stats(const SessionCommand& cmd);
};

