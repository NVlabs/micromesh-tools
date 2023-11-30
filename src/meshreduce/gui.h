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
#include "micro.h"
#include "aabb.h"
#include "bvh.h"

#include "gui_controls.h"
#include "utils.h"

#include "quality.h"
#include "color.h"

#include "session.h"

#include <glad/glad.h>

#include <deque>
#include <filesystem>

struct GLFWwindow;

struct Quality {
	static constexpr int None = 0;
	static constexpr int DistanceInputToMicro = 1;
	static constexpr int MicrofaceAspect = 2;
	static constexpr int MicrofaceStretch = 3;
	static constexpr int MicrodisplacementDistance = 4;
	static constexpr int BaseVertexVisibility = 5;

	const char* modes = "None\0"
	                    "Distance Input -> Micro\0"
	                    "Aspect Ratio (displaced)\0"
	                    "Displacement Stretch\0"
	                    "Displacement Distance\0"
	                    "Base Vertex Visibility\0"
	                    "\0";

	int current_mode = None;
	int computed_mode = None;
	int goto_rank = 1;
	Distribution distribution;

	float max_hi_quality = 0.001f;

	std::vector<std::pair<Vector3, Scalar>> points;
};

struct RenderingOptions {
	static constexpr int RenderLayer_InputMesh = 0;
	static constexpr int RenderLayer_BaseMesh = 1;
	static constexpr int RenderLayer_MicroMesh = 2;
	static constexpr int RenderLayer_ProxyMesh = 3;

	int current_layer = RenderLayer_InputMesh;

	bool render_border = false;

	float shading_weight = 1.0f;
	bool flat = false;

	float metallic = 0.25f;
	float roughness = 0.45f;
	float ambient = 0.1f;

	//Vector3f color = Vector3f(0.8f, 0.8f, 0.8f);
	Vector3f color = Vector3f(0.51f, 0.62f, 0.82f); // 0x839ed1, rgb = 131, 158, 209
	Vector3f lightColor = Vector3f(1.0f, 1.0f, 1.0f);
	float light_intensity = 22.0f;
	
	bool wire = false;
	bool draw_umesh_wire = true;
	float wire_line_width = 1.0f;
	Vector4f wire_color = Vector4f(0.235f, 0.225f, 0.235f, 1.0f);
	Vector4f wire_color2 = Vector4f(0.235f, 0.235f, 0.235f, 1.0f);
	Vector4f displacements_color = Vector4f(1.0f, 1.0f, 0.0f, 1.0f);
	Vector4f direction_field_color = Vector4f(0.22f, 0.32f, 0.92f, 1.0f);

	Vector4f border_wire_color = Vector4f(0.1f, 0.85f, 0.2f, 1.0f);

	bool visible_directions = false;
	bool visible_displacements = false;
	bool visible_direction_field = false;

	bool visible_op_status = false;

	float displacement_scale = 1.0f;
	float displacement_direction_scale = 1.0f;
	float direction_field_scale = 1.0f;
};

struct SolidMaterial {
	float roughness = 0.0f;
	float shading_weight = 1.0f;
	bool flat = false;
	bool lambert = true;
	Vector3f color = Vector3f(0, 1, 0);
};

struct WireMaterial {
	float line_width = 1.0f;
	Vector4f color1 = Vector4f(1, 1, 1, 1);
};

struct GLMeshInfo {
	GLuint vao_solid = 0;
	GLuint vao_wire = 0;
	GLuint buffer = 0;

	bool use_color_attribute = false;
	intptr_t color_offset = 0;

	int n_solid = 0;
	int n_wire = 0;
};

struct GLSubdivisionMeshInfo {
	int n_solid = 0;
	int n_wire_border = 0;
	int n_wire_inner = 0;

	bool use_color_attribute = false;
	intptr_t color_offset = 0;
	int fn = 0;

	GLuint vao_solid = 0;
	GLuint vao_wire_border = 0;
	GLuint vao_wire_inner = 0;
	GLuint buffer_solid = 0;
	GLuint buffer_wire_border = 0;
	GLuint buffer_wire_inner = 0;
};

struct GLLineInfo {
	int n_lines = 0;
	GLuint vao_line = 0;
	GLuint buffer_line = 0;
};

enum DirectionsType : int {
	MaximalVisibility,
	BaseVertexNormals,
	BaseVertexNormalsWithTangentsOnBorder,
	_END
};

struct GUIApplication {

	Session s;

	CommandSequence sequence;
	std::vector<std::string> cmd_strings;
	std::deque<SessionCommand> cmd_queue;

	struct {
		GLuint vao;
		GLuint buffer;
	} gl_selection;

	struct {
		GLuint vao;
		GLuint buffer;
		int n = 0;
	} gl_edit;

	struct {
		GLuint quad_vao;
		GLuint quad_buffer;

		GLuint fbo;
		GLuint color_buffer;
		GLuint id_buffer;
		GLuint depth_buffer;
		GLenum draw_buffers[2];
	} render_target;

	struct {
		int width;
		int height;

		double mx;
		double my;

		GLFWwindow* handle = nullptr;

		Vector4f background0;
		Vector4f background1;
	} window;

	Quality quality;

	bool file_loaded = false;
	bool quit = false;

	struct {
		struct {
			const int menu_width = 350;
			const float pad = 5.0f;
			const float widget_indentation = 12.0f;

			const int tweak_marker_x = 295;
		} layout;

		struct {
			int fi = -1;
			int e = 0;
			int status = 0;
		} selection;

		struct {
			int vi = -1;
			bool active = false;
			Vector3 offset;
		} edit;

		struct {
			bool enabled = false;
			bool fn_flag = true;
			int fn_target = 500000;
			int target_reduction_factor = 0;

			int smoothing_iterations = 0;
			int anisotropic_smoothing_iterations = 0;
			Scalar anisotropic_smoothing_weight = 0.95;

			DecimationParameters dparams;
		} proxy;

		struct {
			struct {
				bool fn_flag = true;
				int fn_target = 0;
				int target_reduction_factor = 0;

				DecimationParameters dparams;
			} decimation;
			
			BVHTree bvh;
		} base;

		struct {
			const int TARGET_PRIMITIVE_COUNT = 0;
			const int TARGET_ERROR = 1;
			const int TARGET_LEVEL = 2;

			const char* mode = "Constant Level\0"
			                   "Uniform Microface Size\0"
			                   "Adaptive Microface Size\0"
			                   "\0";
			int current_mode = 1;

			Scalar microexpansion = 2.0;
			int ufn = 1000000;
			int max_level = 5;
			int level = 3;
			float max_error = 0.15;

			struct {
				int target = 0;
			} adaptive;

			struct {
				int target = 0;
			} adaptive2;

			struct {
				int target = 2;
			} constant;

			struct {
				int target = 0;
			} uniform;
		} tessellate;

		struct {
			const char* types = "MaxVisibility\0"
			                    "As normals\0"
			                    "Tangent\0"
			                    "\0";

			int current_type = DirectionsType::MaximalVisibility;
			bool changed = false;
		} directions;

		bool micromesh_exists = false;

		struct {
			bool enabled = false;
		} displace;

		bool show_rendering_options = false;
		bool show_top_mesh = false;
		bool lock_view = false;

		struct {
			bool on = false;
			char string[256] = "shot";
			int counter = 0;
			int multiplier = 1;
		} screenshot;

		struct {
			int current_map = 4; // default is RdPu
			const char* maps = "Viridis\0"
			                   "Plasma\0"
			                   "Cividis\0"
			                   "Turbo\0"
			                   "RdPu\0"
			                   "PuRd\0"
			                   "GnBu\0"
			                   "PuBu\0"
			                   "YlGnBu\0"
			                   "GistGray\0"
			                   "Hot\0"
			                   "GistHeat\0"
			                   "Copper\0"
			                   "\0";
			const std::vector<ColorMap> cmaps = {
				ColorMap::Viridis,
				ColorMap::Plasma,
				ColorMap::Cividis,
				ColorMap::Turbo,
				ColorMap::RdPu,
				ColorMap::PuRd,
				ColorMap::GnBu,
				ColorMap::PuBu,
				ColorMap::YlGnBu,
				ColorMap::GistGray,
				ColorMap::Hot,
				ColorMap::GistHeat,
				ColorMap::Copper
			};
		} color;
	} gui;

	GLMeshInfo gl_mesh;
	GLMeshInfo gl_mesh_in;
	GLMeshInfo gl_mesh_border;
	GLMeshInfo gl_mesh_proxy;
	GLSubdivisionMeshInfo gl_umesh;

	GLLineInfo gl_top_mesh;

	GLLineInfo gl_directions;
	GLLineInfo gl_displacements;

	struct {
		GLLineInfo hi;
		GLLineInfo proxy;
		GLLineInfo base;
	} gl_direction_field;

	// colors are stored as ABGR u32 for ImGui compatibility
	struct {
		std::map<int, GLLineInfo> gl_edges;
		std::map<int, uint32_t> colors;
	} operations;

	bool _full_draw = true;

	RenderingOptions ro;

	OrbitControl control;
	bool control_drag = false;
	bool control_pan = false;

	Timer click_timer;

	GLuint solid_program;
	GLuint solid_displacement_program;
	GLuint wire_program;
	GLuint wire_displacement_program;
	GLuint quad_program;

	void start(const char *meshfile);

	void load_mesh(const std::string& meshfile, bool reset_controls = true);

	void _init_gl_proxy_colors_from_vertex_area();

	void _init_transforms();
	void _init_gui();
	void _init_glfw();
	void _init_gl();
	void _init_gl_buffers_on_mesh_load();
	void _init_gl_border_mesh();
	void _init_gl_umesh();
	void _init_gl_top_mesh();
	void _init_gl_directions();
	void _init_gl_displacements();
	void _init_gl_direction_field(GLLineInfo& gl_direction_field, const MatrixX& V, const MatrixX& D);
	void _init_gl_umesh_colors_from_face_quality();
	void _init_gl_umesh_colors_from_vertex_quality();
	void _init_gl_hi_colors_from_vertex_quality();
	void _init_gl_hi_colors_from_ambient_occlusion(int num_samples);
	void _init_gl_base_colors_from_vertex_scalar_field(const VectorX& Q, Scalar min_q = 1, Scalar max_q = 0);
	void _resize_offscreen_buffers();

	static constexpr uint8_t KeepDirectionsOnBaseChanged = 1;
	static constexpr uint8_t KeepSubdivisionOnBaseChanged = 1 << 1;

	void _require_full_draw();

	void _update_transforms();
	void _draw_offscreen();
	void _draw_onscreen();
	void _draw_background();
	void _draw_mesh_border(const GLMeshInfo& mesh);
	void _draw_mesh(const GLMeshInfo& mesh, bool force_no_wireframe = false);
	void _draw_mesh(const GLSubdivisionMeshInfo& umesh);
	void _draw_lines(const GLLineInfo& lines, const Vector4f& color);
	void _base_mesh_changed(bool reset_directions = true, bool tessellate = true);
	void _proxy_mesh_changed();
	void _set_base_directions(const MatrixX& VD);
	void _update_umesh_subdivision();
	void _set_selected_dir_toward_eye();

	void _compute_quality();

	void _draw_gui();
		void _main_menu();
			void _input_widgets();
			void _proxy_mesh_widgets();
			void _base_mesh_generation_widgets();
			void _base_mesh_optimization_widgets();
			void _micromesh_generation_widgets();
			void _micromesh_optimization_widgets();
			void _assessment_widgets();
			void _save_widgets();
		
		void _view_overlay();
			void _layer_widgets();
			void _rendering_widgets();
		void _log_overlay();
		void _extra_overlay();

	void _get_mouse_ray(Vector3* o, Vector3* dir) const;

	void select_edge_near_cursor();
	void select_vertex_near_cursor();
	void center_on_mouse_cursor();
	void set_view_from_quality_rank();

	void log_collapse_info_near_cursor() const;

	void tweak_subdivision_level_under_cursor(int val);
	void flip_edge_near_cursor();
	void split_edge_near_cursor();
	void split_vertex_near_cursor();

	void _edit_init();
	void _edit_update();
	void _edit_finalize();

	void _screenshot();

	void _log_init_proxy();
	void _log_load_base_mesh(const std::string& file_path);
	void _log_decimate();
	void _log_tessellate();
	void _log_set_displacement_dirs();
	void _log_displace();
	void _log_tweak_tessellation(int fi, int delta);
	void _log_tweak_displacement_dir(int vi, Vector3 dir);
	void _log_optimize_base_topology();
	void _log_optimize_base_positions(const std::string& optim_mode);
	void _log_minimize_prismoids();
	void _log_reset_tessellation_offsets();
	void _log_flip_edge(int fi, int e);
	void _log_split_edge(int fi, int e);
	void _log_move_vertex();
	void _log_split_vertex(int vi);

	void _execute_pending_command();

	static void key_callback(GLFWwindow* window, int key, int, int action, int);
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void framebuffer_size_callback(GLFWwindow* window, int w, int h);
};

