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
#include "aabb.h"
#include "bvh.h"

#include "gui_controls.h"
#include "utils.h"

#include "quality.h"
#include "color.h"

#include <glad/glad.h>

#include <filesystem>

struct GLFWwindow;

struct RenderingOptions {
	static constexpr int RenderLayer_InputMesh = 0;
	static constexpr int RenderLayer_BaseMesh = 0;
	static constexpr int RenderLayer_MicroMesh = 2;

	int current_layer = RenderLayer_MicroMesh;

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
	float wire_line_width = 1.0f;
	Vector4f wire_color = Vector4f(0.235f, 0.235f, 0.235f, 1.0f);
	Vector4f wire_color2 = Vector4f(0.435f, 0.435f, 0.435f, 0.5f);
	Vector4f displacements_color = Vector4f(1.0f, 1.0f, 0.0f, 1.0f);
	Vector4f direction_field_color = Vector4f(0.22f, 0.32f, 0.92f, 1.0f);

	Vector4f border_wire_color = Vector4f(0.1f, 0.85f, 0.2f, 1.0f);

	float displacement_scale = 1.0f;
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

struct GUIApplication {

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

	//Quality quality;

	bool file_loaded = false;
	bool quit = false;

	struct {
		struct {
			const int menu_width = 350;
			const float pad = 10.0f;
			const float widget_indentation = 12.0f;
		} layout;

		bool show_rendering_options = false;

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

	GLMeshInfo _gl_mesh_input;
	GLMeshInfo _gl_mesh_base;
	GLSubdivisionMeshInfo _gl_umesh;

	struct {
		MatrixX V;
		MatrixXi F;
	} input_mesh;

	struct {
		MatrixX V;
		MatrixXi F;
		BVHTree bvh;
		Box3 box;
	} base;

	SubdivisionMesh umesh;

	//GLLineInfo gl_top_mesh;

	//GLLineInfo gl_directions;
	//GLLineInfo gl_displacements;

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

	void _init_transforms();
	void _init_gui();
	void _init_glfw();
	void _init_gl();
	void _init_gl_buffers(const MatrixX& V, const MatrixXi& F, GLMeshInfo& gl_mesh) const;
	void _init_gl_buffers(GLSubdivisionMeshInfo& gl_umesh) const;
	//void _init_gl_umesh();
	//void _init_gl_umesh_colors_from_face_quality();
	//void _init_gl_umesh_colors_from_vertex_quality();
	//void _init_gl_hi_colors_from_vertex_quality();
	//void _init_gl_base_colors_from_vertex_scalar_field(const VectorX& Q, Scalar min_q = 1, Scalar max_q = 0);
	void _resize_offscreen_buffers();

	void _require_full_draw();

	void _update_transforms();
	void _draw_offscreen();
	void _draw_onscreen();
	void _draw_mesh(const GLMeshInfo& mesh);
	void _draw_mesh(const GLSubdivisionMeshInfo& umesh);
	void _draw_lines(const GLLineInfo& lines, const Vector4f& color);

	//void _compute_quality();

	void _draw_gui();
		void _main_menu();
			void _input_widgets();
	//		void _proxy_mesh_widgets();
	//		void _base_mesh_generation_widgets();
	//		void _base_mesh_optimization_widgets();
	//		void _micromesh_generation_widgets();
	//		void _micromesh_optimization_widgets();
	//		void _assessment_widgets();
	//		void _save_widgets();
	//	
		void _view_overlay();
			void _layer_widgets();
			void _rendering_widgets();
	//	void _log_overlay();
	//	void _extra_overlay();

	void _get_mouse_ray(Vector3* o, Vector3* dir) const;

	void center_on_mouse_cursor();
	//void set_view_from_quality_rank();

	void _screenshot();

	static void key_callback(GLFWwindow* window, int key, int, int action, int);
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void framebuffer_size_callback(GLFWwindow* window, int w, int h);
};

