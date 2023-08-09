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

#include "space.h"
#include "gui.h"
#include "camera.h"
#include "gl_utils.h"
#include "utils.h"
#include "mesh_io.h"
#include "mesh_io_gltf.h"
#include "tangent.h"
#include "intersection.h"
#include "mesh_utils.h"
#include "clean.h"

#include "quality.h"
#include "color.h"

#include "flip.h"

#include "micro.h"
#include "bvh.h"

#include "visibility.h"

#include "ambient_occlusion.h"

#include "stb_image_write.h"
#include "font_droidsans.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <vector>

void glfw_error_callback(int err, const char* description)
{
	std::cerr << "GLFW Error: " << description << std::endl;
}

void GUIApplication::start(const char *meshfile)
{
	_init_glfw();
	//_init_transforms();
	_init_gui();
	_init_gl();

	check_gl_error();

	// decimation data initialization
	if (meshfile)
		load_mesh(meshfile);

	check_gl_error();

	std::cout << "Starting render loop..." << std::endl;

	while (!glfwWindowShouldClose(window.handle) && !quit) {
		glfwPollEvents();

		if (gui.screenshot.on)
			framebuffer_size_callback(window.handle, window.width * gui.screenshot.multiplier, window.height * gui.screenshot.multiplier);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		_update_transforms();

		_draw_offscreen();

		if (gui.screenshot.on) {
			_screenshot();
		}

		_draw_onscreen();

		if (gui.screenshot.on) {
			framebuffer_size_callback(window.handle, window.width / gui.screenshot.multiplier, window.height / gui.screenshot.multiplier);
			gui.screenshot.on = false;
		}

		// make sure all the screenshot stuff is dealt with *before* drawing the gui
		// so the screenshot can also be issued by clicking a button
		_draw_gui();
		
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		check_gl_error();

		glBindVertexArray(0);
		glfwSwapBuffers(window.handle);

		_execute_pending_command();
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	
	glfwDestroyWindow(window.handle);

	glfwTerminate();
}

void GUIApplication::load_mesh(const std::string& meshfile, bool reset_controls)
{
	// set some default parameters on the first mesh load
	if (!s.algo.handle) {
		gui.base.decimation.dparams.use_vertex_smoothing = true;
		gui.base.decimation.dparams.smoothing_coefficient = 0.1;

		gui.base.decimation.dparams.bound_geometric_error = true;
		gui.base.decimation.dparams.max_relative_error = 1;

		gui.base.decimation.dparams.bound_aspect_ratio = true;
	}

	bool new_mesh = meshfile != s.current_mesh.string();

	s.load_mesh(meshfile);

	gui.micromesh_exists = false;

	gui.proxy.fn_target = std::min(500000, int(s.hi.F.rows()));
	gui.proxy.target_reduction_factor = int(std::round(s.hi.F.rows() / (Scalar)gui.proxy.fn_target));

	gui.base.bvh = BVHTree();
	gui.base.bvh.build_tree(&s.base.V, &s.base.F, &s.base.VN, 64);

	if (reset_controls)
		_init_transforms();

	sequence.clear_commands();
	cmd_strings.clear();

	gl_mesh.use_color_attribute = false;
	gl_mesh.color_offset = 0;
	gl_mesh_in.use_color_attribute = false;
	gl_mesh_in.color_offset = 0;
	gl_umesh.use_color_attribute = false;
	gl_umesh.color_offset = 0;

	gui.selection.fi = -1;

	quality.current_mode = Quality::None;
	quality.computed_mode = Quality::None;
	quality.goto_rank = 1;
	quality.distribution = Distribution();

	ro.current_layer = RenderingOptions::RenderLayer_InputMesh;

	if (new_mesh) {
		gui.base.decimation.target_reduction_factor = 32;
		gui.base.decimation.fn_target = (1 / (float)gui.base.decimation.target_reduction_factor) * s.hi.F.rows();
	}
		
	_init_gl_buffers_on_mesh_load();
	_init_gl_direction_field(gl_direction_field.hi, s.hi.V, s.hi.D);
	
	file_loaded = true;

	ro.visible_op_status = false;

	_require_full_draw();
}

void GUIApplication::_init_transforms()
{
	control.offset = s.hi.box.center().cast<float>();

	//control.theta = 0.0f;
	//control.phi = M_PI_2;
	control.r = s.hi.box.diagonal().norm();

	control.camera.fov = 39.6f;
	control.camera.aspect = window.width / (float)window.height;
	control.camera.near_clip = 0.01f;
	control.camera.far_clip = 2000.0f;
}

void GUIApplication::_init_gui()
{
	window.mx = 0;
	window.my = 0;
	
	file_loaded = false;
	quit = false;

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window.handle, true);
	ImGui_ImplOpenGL3_Init();

	ImGuiStyle& style = ImGui::GetStyle();
	style.FrameRounding = 2.0f;
	style.GrabRounding = style.FrameRounding;

	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontFromMemoryCompressedTTF(font_droidsans_compressed_data, font_droidsans_compressed_size, 15);

	ro.visible_displacements = false;

	window.background0 = Vector4f(0.2f, 0.2f, 0.2f, 1.0f);
	window.background1 = Vector4f(0.3f, 0.3f, 0.3f, 1.0f);
}

void GUIApplication::_init_glfw()
{
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		std::exit(-1);
	}
	glfwSetErrorCallback(glfw_error_callback);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 1);

	//glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

	const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

	window.width = int(mode->width * 0.85f);
	window.height = int(mode->height * 0.85f);

	std::string title = std::string("MeshReduce");

	window.handle = glfwCreateWindow(window.width, window.height, title.c_str(), NULL, NULL);
	if (!window.handle) {
		std::cerr << "Failed to create window or context" << std::endl;
		std::exit(-1);
	}

	// set callbacks
	glfwSetMouseButtonCallback(window.handle, mouse_button_callback);
	glfwSetCursorPosCallback(window.handle, cursor_position_callback);
	glfwSetScrollCallback(window.handle, scroll_callback);
	glfwSetKeyCallback(window.handle, key_callback);
	//glfwSetCharCallback(window.handle, ImGui_ImplGlfwGL3_CharCallback);
	glfwSetFramebufferSizeCallback(window.handle, framebuffer_size_callback);

	glfwMakeContextCurrent(window.handle);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize OpenGL context" << std::endl;
		std::exit(-1);
	}
	
	glfwSwapInterval(1);

	glfwSetWindowUserPointer(window.handle, this);
	
	framebuffer_size_callback(window.handle, window.width, window.height);
	
	std::cout << "Initialized GLFW" << std::endl;
}

#include "shader_strings.h"

void GUIApplication::_init_gl()
{
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.0f, 1.0f);

	const char* svsptr = solid_v;
	const char* svdsptr = solid_v_d;
	const char* sfsptr = solid_f;
	const char* wvsptr = wire_v;
	const char* wfsptr = wire_f;
	const char* qvsptr = quad_v;
	const char* qfsptr = quad_f;

	solid_program = compile_shader_program(&svsptr, &sfsptr);
	check_gl_error();
	solid_displacement_program = compile_shader_program(&svdsptr, &sfsptr);
	check_gl_error();
	wire_program = compile_shader_program(&wvsptr, &wfsptr);
	check_gl_error();
	wire_displacement_program = compile_shader_program(&svdsptr, &wfsptr);
	check_gl_error();
	quad_program = compile_shader_program(&qvsptr, &qfsptr);
	check_gl_error();
	
	std::cout << "Compiled shaders" << std::endl;

	{
		glGenVertexArrays(1, &gl_selection.vao);
		glBindVertexArray(gl_selection.vao);
		glGenBuffers(1, &gl_selection.buffer);
		glBindBuffer(GL_ARRAY_BUFFER, gl_selection.buffer);
		glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), nullptr, GL_STATIC_DRAW);

		GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
		glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(wire_pos_location);
		check_gl_error();

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenVertexArrays(1, &gl_edit.vao);
		glBindVertexArray(gl_edit.vao);
		glGenBuffers(1, &gl_edit.buffer);
		glBindBuffer(GL_ARRAY_BUFFER, gl_edit.buffer);
		glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

		gl_edit.n = 0;

		wire_pos_location = glGetAttribLocation(wire_program, "position");
		glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(wire_pos_location);
		check_gl_error();

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// setup buffers for rendering the 'onscreen' quad
	// position and uvs, all 2D
	std::vector<float> quad_vertex_data = {
		-1.0f, -1.0f, 0.0f, 0.0f,
		 1.0f, -1.0f, 1.0f, 0.0f,
		 1.0f,  1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 1.0f, 1.0f,
		-1.0f,  1.0f, 0.0f, 1.0f,
	};

	glGenVertexArrays(1, &render_target.quad_vao);
	glBindVertexArray(render_target.quad_vao);
	glGenBuffers(1, &render_target.quad_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, render_target.quad_buffer);
	glBufferData(GL_ARRAY_BUFFER, quad_vertex_data.size() * sizeof(float), quad_vertex_data.data(), GL_STATIC_DRAW);
	check_gl_error();

	GLint quad_pos_location = glGetAttribLocation(quad_program, "position");
	glVertexAttribPointer(quad_pos_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
	glEnableVertexAttribArray(quad_pos_location);

	GLint quad_tc_location = glGetAttribLocation(quad_program, "texcoord");
	glVertexAttribPointer(quad_tc_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(quad_tc_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	check_gl_error();

	std::cout << "Initialized window buffers" << std::endl;
}

void GUIApplication::_init_gl_buffers_on_mesh_load()
{
	// generate buffer data
	int vertex_size = 6 * sizeof(float); // position and normal
	std::vector<float> buffer_data;
	buffer_data.reserve(3 * s.hi.F.rows() * 6);
	for (int i = 0; i < s.hi.F.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3f p = s.hi.V.row(s.hi.F(i, j)).cast<float>();
			Vector3f n = s.hi.VN.row(s.hi.F(i, j)).cast<float>();
			buffer_data.push_back(p(0));
			buffer_data.push_back(p(1));
			buffer_data.push_back(p(2));
			buffer_data.push_back(n(0));
			buffer_data.push_back(n(1));
			buffer_data.push_back(n(2));
		}
	}

	// setup base mesh buffers
	{
		if (!gl_mesh.buffer)
			glGenBuffers(1, &gl_mesh.buffer);

		glBindBuffer(GL_ARRAY_BUFFER, gl_mesh.buffer);
		//glBufferData(GL_ARRAY_BUFFER, buffer_data.size() * sizeof(float), buffer_data.data(), GL_STATIC_DRAW);
		glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

		// setup 'solid' rendering vao
		if (!gl_mesh.vao_solid) {
			glGenVertexArrays(1, &gl_mesh.vao_solid);
			glBindVertexArray(gl_mesh.vao_solid);

			GLint solid_pos_location = glGetAttribLocation(solid_program, "position");
			glVertexAttribPointer(solid_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
			glEnableVertexAttribArray(solid_pos_location);
			check_gl_error();

			GLint solid_normal_location = glGetAttribLocation(solid_program, "normal");
			glVertexAttribPointer(solid_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(solid_normal_location);
			check_gl_error();

			GLint color_location = glGetAttribLocation(solid_program, "color");
			glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_mesh.color_offset));
			glEnableVertexAttribArray(color_location);
			check_gl_error();

			glBindVertexArray(0);
		}

		// reuse the buffers for wire rendering

		if (!gl_mesh.vao_wire) {
			glGenVertexArrays(1, &gl_mesh.vao_wire);
			glBindVertexArray(gl_mesh.vao_wire);

			GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
			glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
			glEnableVertexAttribArray(wire_pos_location);
			check_gl_error();

			GLint wire_normal_location = glGetAttribLocation(wire_program, "normal");
			glVertexAttribPointer(wire_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(wire_normal_location);
			check_gl_error();

			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		gl_mesh.n_solid = gl_mesh.n_wire = 0;
	}

	// setup input mesh buffers
	{
		if (!gl_mesh_in.buffer)
			glGenBuffers(1, &gl_mesh_in.buffer);

		// pre-allocate color storage
		glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_in.buffer);
		glBufferData(GL_ARRAY_BUFFER, buffer_data.size() * sizeof(float) + (3 * s.hi.F.rows() * sizeof(uint32_t)), 0, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_data.size() * sizeof(float), buffer_data.data());
		gl_mesh_in.color_offset = buffer_data.size() * sizeof(float);

		// setup 'solid' rendering vao

		if (!gl_mesh_in.vao_solid) {
			glGenVertexArrays(1, &gl_mesh_in.vao_solid);
			glBindVertexArray(gl_mesh_in.vao_solid);

			GLint solid_pos_location = glGetAttribLocation(solid_program, "position");
			glVertexAttribPointer(solid_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
			glEnableVertexAttribArray(solid_pos_location);
			check_gl_error();

			GLint solid_normal_location = glGetAttribLocation(solid_program, "normal");
			glVertexAttribPointer(solid_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(solid_normal_location);
			check_gl_error();

			glBindVertexArray(0);
		}

		// reuse the buffers for wire rendering
		if (!gl_mesh_in.vao_wire) {
			glGenVertexArrays(1, &gl_mesh_in.vao_wire);
			glBindVertexArray(gl_mesh_in.vao_wire);

			GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
			glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
			glEnableVertexAttribArray(wire_pos_location);
			check_gl_error();

			GLint wire_normal_location = glGetAttribLocation(wire_program, "normal");
			glVertexAttribPointer(wire_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(wire_normal_location);
			check_gl_error();

			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		
		gl_mesh_in.n_solid = gl_mesh_in.n_wire = (int)3 * s.base.F.rows();
	}

	// setup proxy mesh buffers
	{
		if (!gl_mesh_proxy.buffer)
			glGenBuffers(1, &gl_mesh_proxy.buffer);

		// pre-allocate color storage
		glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_proxy.buffer);
		glBufferData(GL_ARRAY_BUFFER, buffer_data.size() * sizeof(float) + (3 * s.hi.F.rows() * sizeof(uint32_t)), 0, GL_STATIC_DRAW);
		//glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_data.size() * sizeof(float), buffer_data.data());
		gl_mesh_proxy.color_offset = buffer_data.size() * sizeof(float);

		if (!gl_mesh_proxy.vao_solid) {
			glGenVertexArrays(1, &gl_mesh_proxy.vao_solid);
			glBindVertexArray(gl_mesh_proxy.vao_solid);

			GLint solid_pos_location = glGetAttribLocation(solid_program, "position");
			glVertexAttribPointer(solid_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
			glEnableVertexAttribArray(solid_pos_location);
			check_gl_error();

			GLint solid_normal_location = glGetAttribLocation(solid_program, "normal");
			glVertexAttribPointer(solid_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(solid_normal_location);
			check_gl_error();

			glBindVertexArray(0);
		}

		if (!gl_mesh_proxy.vao_wire) {
			glGenVertexArrays(1, &gl_mesh_proxy.vao_wire);
			glBindVertexArray(gl_mesh_proxy.vao_wire);

			GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
			glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
			glEnableVertexAttribArray(wire_pos_location);
			check_gl_error();

			GLint wire_normal_location = glGetAttribLocation(wire_program, "normal");
			glVertexAttribPointer(wire_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(wire_normal_location);
			check_gl_error();

			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		
		gl_mesh_proxy.n_solid = gl_mesh_proxy.n_wire = (int)3 * s.base.F.rows();
	}

	check_gl_error();

	_init_gl_border_mesh();

	std::cout << "Initialized GL buffers" << std::endl;
}

void GUIApplication::_init_gl_border_mesh()
{
	// generate buffer data
	int vertex_size = 6 * sizeof(float); // position and normal
	std::vector<float> buffer_data;
	buffer_data.reserve(3 * s.border.F.rows() * 6);
	for (int i = 0; i < s.border.F.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 p = s.border.V.row(s.border.F(i, j));
			Vector3 n = s.border.VN.row(s.border.F(i, j));
			buffer_data.push_back(p.x());
			buffer_data.push_back(p.y());
			buffer_data.push_back(p.z());
			buffer_data.push_back(n.x());
			buffer_data.push_back(n.y());
			buffer_data.push_back(n.z());
		}
	}

	if (!gl_mesh_border.buffer)
		glGenBuffers(1, &gl_mesh_border.buffer);

	glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_border.buffer);
	glBufferData(GL_ARRAY_BUFFER, buffer_data.size() * sizeof(float), buffer_data.data(), GL_STATIC_DRAW);

	// setup 'solid' rendering vao
	if (!gl_mesh_border.vao_solid) {
		glGenVertexArrays(1, &gl_mesh_border.vao_solid);
		glBindVertexArray(gl_mesh_border.vao_solid);

		GLint solid_pos_location = glGetAttribLocation(solid_program, "position");
		glVertexAttribPointer(solid_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
		glEnableVertexAttribArray(solid_pos_location);
		check_gl_error();

		GLint solid_normal_location = glGetAttribLocation(solid_program, "normal");
		glVertexAttribPointer(solid_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(solid_normal_location);
		check_gl_error();

		glBindVertexArray(0);
	}

	// reuse the buffers for wire rendering
	if (!gl_mesh_border.vao_wire) {
		glGenVertexArrays(1, &gl_mesh_border.vao_wire);
		glBindVertexArray(gl_mesh_border.vao_wire);

		GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
		glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
		glEnableVertexAttribArray(wire_pos_location);
		check_gl_error();

		GLint wire_normal_location = glGetAttribLocation(wire_program, "normal");
		glVertexAttribPointer(wire_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(wire_normal_location);
		check_gl_error();

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	gl_mesh_border.n_solid = gl_mesh_border.n_wire = (int)3 * s.border.F.rows();

	check_gl_error();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_init_gl_umesh()
{
	Assert(s.micromesh.micro_fn > 0);

	Timer t;

	if (gl_umesh.vao_solid)
		glDeleteVertexArrays(1, &gl_umesh.vao_solid);
	if (gl_umesh.vao_wire_border)
		glDeleteVertexArrays(1, &gl_umesh.vao_wire_border);
	if (gl_umesh.vao_wire_inner)
		glDeleteVertexArrays(1, &gl_umesh.vao_wire_inner);

	//if (gl_umesh.buffer_solid)
	//	glDeleteBuffers(1, &gl_umesh.buffer_solid);
	//if (gl_umesh.buffer_wire_border)
	//	glDeleteBuffers(1, &gl_umesh.buffer_wire_border);
	//if (gl_umesh.buffer_wire_inner)
	//	glDeleteBuffers(1, &gl_umesh.buffer_wire_inner);

	constexpr int vertex_size = 9 * sizeof(float);
	constexpr int wire_vertex_size = 6 * sizeof(float);

	std::vector<float> solid_buffer;
	std::vector<float> wire_buffer_border;
	std::vector<float> wire_buffer_inner;

	solid_buffer.reserve(uint64_t(3) * s.micromesh.micro_fn * 9);
	wire_buffer_border.reserve(uint64_t(3) * s.micromesh.micro_fn * 6);
	wire_buffer_inner.reserve(s.micromesh.micro_fn);

	gl_umesh.fn = 0;
	for (const SubdivisionTri& st : s.micromesh.faces) {
		BarycentricGrid bary_grid(st.subdivision_level());
		std::vector<uint8_t> border(bary_grid.num_samples(), false);
		for (int i = 0; i < bary_grid.samples_per_side(); ++i)
			for (int j = 0; j <= i; ++j)
				if (i == 0 || i == bary_grid.samples_per_side() - 1 || i == j)
					border[bary_grid.index(i, j)] = true;

		int deg = st.F.cols();
		Assert(deg == 3);
		for (int fi = 0; fi < st.F.rows(); ++fi) {
			gl_umesh.fn++;
			for (int j = 0; j < deg; ++j) {
				Vector3 p = st.V.row(st.F(fi, j));
				Vector3 n = st.VN.row(st.F(fi, j));
				Vector3 d = st.VD.row(st.F(fi, j));
				solid_buffer.push_back(p.x());
				solid_buffer.push_back(p.y());
				solid_buffer.push_back(p.z());
				solid_buffer.push_back(n.x());
				solid_buffer.push_back(n.y());
				solid_buffer.push_back(n.z());
				solid_buffer.push_back(d.x());
				solid_buffer.push_back(d.y());
				solid_buffer.push_back(d.z());

				Edge e(st.F(fi, j), st.F(fi, (j + 1) % deg));
				Vector3 e0 = st.V.row(e.first);
				Vector3 e1 = st.V.row(e.second);
				Vector3 n0 = st.VN.row(e.first);
				Vector3 n1 = st.VN.row(e.second);
				Vector3 d0 = st.VD.row(e.first);
				Vector3 d1 = st.VD.row(e.second);

				int e0i, e0j;
				int e1i, e1j;
				bary_grid.inverted_index(e.first, &e0i, &e0j);
				bary_grid.inverted_index(e.second, &e1i, &e1j);

				if ((e0j == 0 && e1j == 0) // both on first column
					|| (e0i == bary_grid.samples_per_side() - 1 && e1i == bary_grid.samples_per_side() - 1) // both on last row
					|| (e0i == e0j && e1i == e1j)) { // both on diagonal
					wire_buffer_border.push_back(e0.x());
					wire_buffer_border.push_back(e0.y());
					wire_buffer_border.push_back(e0.z());
					//wire_buffer_border.push_back(n0.x());
					//wire_buffer_border.push_back(n0.y());
					//wire_buffer_border.push_back(n0.z());
					wire_buffer_border.push_back(d0.x());
					wire_buffer_border.push_back(d0.y());
					wire_buffer_border.push_back(d0.z());
					wire_buffer_border.push_back(e1.x());
					wire_buffer_border.push_back(e1.y());
					wire_buffer_border.push_back(e1.z());
					//wire_buffer_border.push_back(n1.x());
					//wire_buffer_border.push_back(n1.y());
					//wire_buffer_border.push_back(n1.z());
					wire_buffer_border.push_back(d1.x());
					wire_buffer_border.push_back(d1.y());
					wire_buffer_border.push_back(d1.z());
				}
				else {
					wire_buffer_inner.push_back(e0.x());
					wire_buffer_inner.push_back(e0.y());
					wire_buffer_inner.push_back(e0.z());
					//wire_buffer_inner.push_back(n0.x());
					//wire_buffer_inner.push_back(n0.y());
					//wire_buffer_inner.push_back(n0.z());
					wire_buffer_inner.push_back(d0.x());
					wire_buffer_inner.push_back(d0.y());
					wire_buffer_inner.push_back(d0.z());
					wire_buffer_inner.push_back(e1.x());
					wire_buffer_inner.push_back(e1.y());
					wire_buffer_inner.push_back(e1.z());
					//wire_buffer_inner.push_back(n1.x());
					//wire_buffer_inner.push_back(n1.y());
					//wire_buffer_inner.push_back(n1.z());
					wire_buffer_inner.push_back(d1.x());
					wire_buffer_inner.push_back(d1.y());
					wire_buffer_inner.push_back(d1.z());
				}
			}
		}
	}

	gl_umesh.n_solid = solid_buffer.size() / 9;
	gl_umesh.n_wire_border = wire_buffer_border.size() / 6;
	gl_umesh.n_wire_inner = wire_buffer_inner.size() / 6;

	gl_umesh.color_offset = int64_t(3) * gl_umesh.fn * vertex_size;

	// solid buffer (pre-allocate color storage at the end of the buffer)

	if (!gl_umesh.buffer_solid)
		glGenBuffers(1, &gl_umesh.buffer_solid);
	glBindBuffer(GL_ARRAY_BUFFER, gl_umesh.buffer_solid);
	glBufferData(GL_ARRAY_BUFFER, solid_buffer.size() * sizeof(float) + (3 * gl_umesh.fn * sizeof(uint32_t)), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, solid_buffer.size() * sizeof(float), solid_buffer.data());

	glGenVertexArrays(1, &gl_umesh.vao_solid);
	check_gl_error();
	glBindVertexArray(gl_umesh.vao_solid);
	check_gl_error();

	GLint solid_pos_location = glGetAttribLocation(solid_displacement_program, "position");
	check_gl_error();
	glVertexAttribPointer(solid_pos_location, 3, GL_FLOAT, GL_FALSE, vertex_size, 0);
	check_gl_error();
	glEnableVertexAttribArray(solid_pos_location);
	check_gl_error();

	GLint solid_normal_location = glGetAttribLocation(solid_displacement_program, "normal");
	glVertexAttribPointer(solid_normal_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(solid_normal_location);
	check_gl_error();

	GLint solid_displacement_location = glGetAttribLocation(solid_displacement_program, "displacement");
	glVertexAttribPointer(solid_displacement_location, 3, GL_FLOAT, GL_FALSE, vertex_size, (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(solid_displacement_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// wire border buffer

	if (!gl_umesh.buffer_wire_border)
		glGenBuffers(1, &gl_umesh.buffer_wire_border);
	glBindBuffer(GL_ARRAY_BUFFER, gl_umesh.buffer_wire_border);
	glBufferData(GL_ARRAY_BUFFER, wire_buffer_border.size() * sizeof(float), wire_buffer_border.data(), GL_STATIC_DRAW);

	glGenVertexArrays(1, &gl_umesh.vao_wire_border);
	glBindVertexArray(gl_umesh.vao_wire_border);

	GLint wire_border_pos_location = glGetAttribLocation(wire_displacement_program, "position");
	glVertexAttribPointer(wire_border_pos_location, 3, GL_FLOAT, GL_FALSE, wire_vertex_size, 0);
	glEnableVertexAttribArray(wire_border_pos_location);

	check_gl_error();

	//GLint wire_border_normal_location = glGetAttribLocation(wire_displacement_program, "normal");
	//glVertexAttribPointer(wire_border_normal_location, 3, GL_FLOAT, GL_FALSE, wire_vertex_size, (void*)(3 * sizeof(float)));
	//glEnableVertexAttribArray(wire_border_normal_location);
	//check_gl_error();

	GLint wire_border_displacement_location = glGetAttribLocation(wire_displacement_program, "displacement");
	glVertexAttribPointer(wire_border_displacement_location, 3, GL_FLOAT, GL_FALSE, wire_vertex_size, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(wire_border_displacement_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// wire inner buffer

	if (!gl_umesh.buffer_wire_inner)
		glGenBuffers(1, &gl_umesh.buffer_wire_inner);
	glBindBuffer(GL_ARRAY_BUFFER, gl_umesh.buffer_wire_inner);
	glBufferData(GL_ARRAY_BUFFER, wire_buffer_inner.size() * sizeof(float), wire_buffer_inner.data(), GL_STATIC_DRAW);

	glGenVertexArrays(1, &gl_umesh.vao_wire_inner);
	glBindVertexArray(gl_umesh.vao_wire_inner);

	GLint wire_inner_pos_location = glGetAttribLocation(wire_displacement_program, "position");
	glVertexAttribPointer(wire_inner_pos_location, 3, GL_FLOAT, GL_FALSE, wire_vertex_size, 0);
	glEnableVertexAttribArray(wire_inner_pos_location);
	check_gl_error();

	//GLint wire_inner_normal_location = glGetAttribLocation(wire_displacement_program, "normal");
	//glVertexAttribPointer(wire_inner_normal_location, 3, GL_FLOAT, GL_FALSE, wire_vertex_size, (void*)(3 * sizeof(float)));
	//glEnableVertexAttribArray(wire_inner_normal_location);
	//check_gl_error();

	GLint wire_inner_displacement_location = glGetAttribLocation(wire_displacement_program, "displacement");
	glVertexAttribPointer(wire_inner_displacement_location, 3, GL_FLOAT, GL_FALSE, wire_vertex_size, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(wire_inner_displacement_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (s.micromesh.is_displaced()) {
		_init_gl_displacements();
	}
	
	gl_umesh.use_color_attribute = false;

	if (quality.computed_mode != Quality::None && s.micromesh.is_displaced()) {
		quality.current_mode = quality.computed_mode;
		_compute_quality();
	}

	_init_gl_top_mesh();
}

void GUIApplication::_init_gl_top_mesh()
{
	if (gl_top_mesh.vao_line) {
		glDeleteVertexArrays(1, &gl_top_mesh.vao_line);
		gl_top_mesh.vao_line = 0;
	}
	
	if (gl_top_mesh.buffer_line) {
		glDeleteBuffers(1, &gl_top_mesh.buffer_line);
		gl_top_mesh.buffer_line = 0;
	}

	gl_top_mesh.n_lines = 0;

	MatrixX V = s.base.V + s.base.VD;

	std::vector<float> buffer;
	buffer.reserve(18 * s.base.F.rows());

	for (const SubdivisionTri& st : s.micromesh.faces) {
		Matrix3 T = st.base_V + st.base_VD;
		MatrixX V;
		MatrixXi F;
		//subdivide_tri(T.row(0), T.row(1), T.row(2), V, F, st.subdivision_bits);
		subdivide_tri(T.row(0), T.row(1), T.row(2), V, F, 0);
		for (int fi = 0; fi < F.rows(); ++fi) {
			for (int j = 0; j < 3; ++j) {

				buffer.push_back(V.row(F(fi, j))(0));
				buffer.push_back(V.row(F(fi, j))(1));
				buffer.push_back(V.row(F(fi, j))(2));

				buffer.push_back(V.row(F(fi, (j + 1) % 3))(0));
				buffer.push_back(V.row(F(fi, (j + 1) % 3))(1));
				buffer.push_back(V.row(F(fi, (j + 1) % 3))(2));

				gl_top_mesh.n_lines++;
			}
		}
	}

	glGenVertexArrays(1, &gl_top_mesh.vao_line);
	glBindVertexArray(gl_top_mesh.vao_line);

	glGenBuffers(1, &gl_top_mesh.buffer_line);
	glBindBuffer(GL_ARRAY_BUFFER, gl_top_mesh.buffer_line);
	glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float), buffer.data(), GL_STATIC_DRAW);

	GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
	glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(wire_pos_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_init_gl_directions()
{
	if (gl_directions.vao_line) {
		glDeleteVertexArrays(1, &gl_directions.vao_line);
		gl_directions.vao_line = 0;
	}
	
	if (gl_directions.buffer_line) {
		glDeleteBuffers(1, &gl_directions.buffer_line);
		gl_directions.buffer_line = 0;
	}

	std::vector<float> lines;
	gl_directions.n_lines = 0;

	Scalar scale = ro.displacement_direction_scale;

	Assert(s.base.VN.rows() > 0);
	Assert(s.base.VN.rows() == s.base.V.rows());
	for (int i = 0; i < s.base.V.rows(); ++i) {
		Vector3 pj = s.base.V.row(i);
		Vector3 pjd = s.base.V.row(i) + s.base.VD.row(i) * scale;
		lines.push_back(pj.x());
		lines.push_back(pj.y());
		lines.push_back(pj.z());
		lines.push_back(pjd.x());
		lines.push_back(pjd.y());
		lines.push_back(pjd.z());
		gl_directions.n_lines++;
	}

	glGenVertexArrays(1, &gl_directions.vao_line);
	glBindVertexArray(gl_directions.vao_line);

	glGenBuffers(1, &gl_directions.buffer_line);
	glBindBuffer(GL_ARRAY_BUFFER, gl_directions.buffer_line);
	glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);

	GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
	glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(wire_pos_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_init_gl_displacements()
{
	Assert(s.micromesh.is_displaced());

	if (gl_displacements.vao_line) {
		glDeleteVertexArrays(1, &gl_displacements.vao_line);
		gl_displacements.vao_line = 0;
	}
	
	if (gl_displacements.buffer_line) {
		glDeleteBuffers(1, &gl_displacements.buffer_line);
		gl_displacements.buffer_line = 0;
	}

	int vertex_size = 6 * sizeof(float);

	std::vector<float> lines;
	gl_displacements.n_lines = 0;

	for (const SubdivisionTri& uface : s.micromesh.faces) {
		BarycentricGrid grid(uface.subdivision_level());

		//for (int i = 0; i < grid.num_samples(); ++i) {
		//	Vector3 pi = uface.V.row(i);
		//	Vector3 pid = uface.V.row(i) + uface.VD.row(i);
		//	lines.push_back(pi.x());
		//	lines.push_back(pi.y());
		//	lines.push_back(pi.z());
		//	lines.push_back(pid.x());
		//	lines.push_back(pid.y());
		//	lines.push_back(pid.z());
		//	gl_displacements.n_lines++;
		//}

		int n = grid.samples_per_side();
		for (int i = 0; i < n; ++i) {
			Vector3 pi = uface.V.row(grid.index(i, 0));
			Vector3 pid = uface.V.row(grid.index(i, 0)) + uface.VD.row(grid.index(i, 0));
			lines.push_back(pi.x());
			lines.push_back(pi.y());
			lines.push_back(pi.z());
			lines.push_back(pid.x());
			lines.push_back(pid.y());
			lines.push_back(pid.z());
			gl_displacements.n_lines++;

			pi = uface.V.row(grid.index(n - 1, i));
			pid = uface.V.row(grid.index(n - 1, i)) + uface.VD.row(grid.index(n - 1, i));
			lines.push_back(pi.x());
			lines.push_back(pi.y());
			lines.push_back(pi.z());
			lines.push_back(pid.x());
			lines.push_back(pid.y());
			lines.push_back(pid.z());
			gl_displacements.n_lines++;

			pi = uface.V.row(grid.index(i, i));
			pid = uface.V.row(grid.index(i, i)) + uface.VD.row(grid.index(i, i));
			lines.push_back(pi.x());
			lines.push_back(pi.y());
			lines.push_back(pi.z());
			lines.push_back(pid.x());
			lines.push_back(pid.y());
			lines.push_back(pid.z());
			gl_displacements.n_lines++;
		}
	}

	glGenVertexArrays(1, &gl_displacements.vao_line);
	glBindVertexArray(gl_displacements.vao_line);

	glGenBuffers(1, &gl_displacements.buffer_line);
	glBindBuffer(GL_ARRAY_BUFFER, gl_displacements.buffer_line);
	glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);

	GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
	glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(wire_pos_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_init_gl_direction_field(GLLineInfo& gl_direction_field, const MatrixX& V, const MatrixX& D)
{
	//Assert(D.rows() > 0);
	//Assert(V.rows() == D.rows());

	if (gl_direction_field.vao_line) {
		glDeleteVertexArrays(1, &gl_direction_field.vao_line);
		gl_direction_field.vao_line = 0;
	}
	
	if (gl_direction_field.buffer_line) {
		glDeleteBuffers(1, &gl_direction_field.buffer_line);
		gl_direction_field.buffer_line = 0;
	}

	int vertex_size = 6 * sizeof(float);

	std::vector<float> lines;
	gl_direction_field.n_lines = 0;

	for (long i = 0; i < D.rows(); ++i) {
		Vector3 d0 = V.row(i);
		Vector3 d1 = V.row(i) + 0.5 * D.row(i);
		Vector3 d2 = V.row(i) - 0.5 * D.row(i);
		lines.push_back(d0.x()); lines.push_back(d0.y()); lines.push_back(d0.z());
		lines.push_back(d1.x()); lines.push_back(d1.y()); lines.push_back(d1.z());
		gl_direction_field.n_lines++;
		lines.push_back(d0.x()); lines.push_back(d0.y()); lines.push_back(d0.z());
		lines.push_back(d2.x()); lines.push_back(d2.y()); lines.push_back(d2.z());
		gl_direction_field.n_lines++;
	}

	glGenVertexArrays(1, &gl_direction_field.vao_line);
	glBindVertexArray(gl_direction_field.vao_line);

	glGenBuffers(1, &gl_direction_field.buffer_line);
	glBindBuffer(GL_ARRAY_BUFFER, gl_direction_field.buffer_line);
	glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);

	GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
	glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(wire_pos_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_init_gl_umesh_colors_from_face_quality()
{
	if (s.micromesh.has_face_quality()) {
		quality.distribution = Distribution();
		for (const SubdivisionTri& uface : s.micromesh.faces) {
			for (int ufi = 0; ufi < uface.F.rows(); ++ufi) {
				quality.distribution.add(uface.FQ(ufi));
			}
		}

		std::cout << "Distribution info" << std::endl;
		std::cout << "  MIN " << quality.distribution.min() << std::endl;;
		std::cout << "  MAX " << quality.distribution.max() << std::endl;;
		std::cout << "  AVG " << quality.distribution.avg() << std::endl;;

		Histogram h(0, 1, 256);
		for (const SubdivisionTri& uface : s.micromesh.faces) {
			for (int ufi = 0; ufi < uface.F.rows(); ++ufi) {
				h.add(uface.FQ(ufi), triangle_area(
					Vector3(uface.V.row(uface.F(ufi, 0))),
					Vector3(uface.V.row(uface.F(ufi, 1))),
					Vector3(uface.V.row(uface.F(ufi, 2)))));
			}
		}

		std::cout << "Histogram info" << std::endl;
		std::cout << "  MIN " << h.min() << std::endl;;
		std::cout << "  MAX " << h.max() << std::endl;;
		std::cout << "  AVG " << h.avg() << std::endl;;

		std::vector<Vector4u8> colors;
		for (const SubdivisionTri& uface : s.micromesh.faces) {
			for (int ufi = 0; ufi < uface.F.rows(); ++ufi) {
				Scalar q = (uface.FQ(ufi) - quality.distribution.min()) / (quality.distribution.max() - quality.distribution.min());
				for (int i = 0; i < 3; ++i) {
					Vector4u8 c = get_color_mapping(1 - q, 0, 1, gui.color.cmaps[gui.color.current_map]);
					colors.push_back(c);
				}
			}
		}

		glBindVertexArray(gl_umesh.vao_solid);
		glBindBuffer(GL_ARRAY_BUFFER, gl_umesh.buffer_solid);

		glBufferSubData(GL_ARRAY_BUFFER, gl_umesh.color_offset, colors.size() * sizeof(Vector4u8), colors.data());

		GLint color_location = glGetAttribLocation(solid_displacement_program, "color");
		glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_umesh.color_offset));
		glEnableVertexAttribArray(color_location);
		check_gl_error();

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else {
		std::cout << "WARNING: uMesh has no face quality computed" << std::endl;
	}
}

void GUIApplication::_init_gl_umesh_colors_from_vertex_quality()
{
	if (s.micromesh.has_vertex_quality()) {
		quality.distribution = Distribution();
		for (const SubdivisionTri& uface : s.micromesh.faces) {
			std::set<int> vset;
			for (int ufi = 0; ufi < uface.F.rows(); ++ufi) {
				for (int j = 0; j < 3; ++j) {
					int vi = uface.F(ufi, j);
					if (vset.find(vi) == vset.end()) {
						vset.insert(vi);
						quality.distribution.add(uface.VQ(vi));
					}
				}
			}
		}

		std::cout << "Distribution info" << std::endl;
		std::cout << "  MIN " << quality.distribution.min() << std::endl;
		std::cout << "  MAX " << quality.distribution.max() << std::endl;
		std::cout << "  AVG " << quality.distribution.avg() << std::endl;

		std::vector<Vector4u8> colors;
		for (const SubdivisionTri& uface : s.micromesh.faces) {
			for (int ufi = 0; ufi < uface.F.rows(); ++ufi) {
				for (int j = 0; j < 3; ++j) {
					int vi = uface.F(ufi, j);
					Scalar q = (uface.VQ(vi) - quality.distribution.min()) / (quality.distribution.max() - quality.distribution.min());
					Vector4u8 c = get_color_mapping(1 - q, 0, 1, gui.color.cmaps[gui.color.current_map]);
					colors.push_back(c);
				}
			}
		}

		glBindVertexArray(gl_umesh.vao_solid);
		glBindBuffer(GL_ARRAY_BUFFER, gl_umesh.buffer_solid);

		glBufferSubData(GL_ARRAY_BUFFER, gl_umesh.color_offset, colors.size() * sizeof(Vector4u8), colors.data());

		GLint color_location = glGetAttribLocation(solid_displacement_program, "color");
		glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_umesh.color_offset));
		glEnableVertexAttribArray(color_location);
		check_gl_error();

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else {
		std::cout << "WARNING: uMesh has no vertex quality computed" << std::endl;
	}
}

void GUIApplication::_init_gl_hi_colors_from_vertex_quality()
{
	Assert(s.hi.VQ.rows() == s.hi.V.rows());
	quality.distribution = Distribution();
	for (int vi = 0; vi < (int)s.hi.VQ.rows(); ++vi)
		quality.distribution.add(s.hi.VQ(vi));

	std::cout << "Distribution info" << std::endl;
	std::cout << "  MIN " << quality.distribution.min() << std::endl;
	std::cout << "  MAX " << quality.distribution.max() << std::endl;
	std::cout << "  AVG " << quality.distribution.avg() << std::endl;

	Histogram h(0, quality.distribution.max(), 256);
	VectorX vertex_areas = compute_voronoi_vertex_areas(s.hi.V, s.hi.F);
	for (int vi = 0; vi < (int)s.hi.VQ.rows(); ++vi)
		h.add(s.hi.VQ(vi), vertex_areas(vi));

	std::cout << "Histogram info" << std::endl;
	std::cout << "  MIN " << h.min() << std::endl;;
	std::cout << "  MAX " << h.max() << std::endl;;
	std::cout << "  AVG " << h.avg() << std::endl;;

	std::vector<Vector4u8> colors;
	for (int i = 0; i < (int)s.hi.F.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int vi = s.hi.F(i, j);
			Scalar q = s.hi.VQ(vi);
			Vector4u8 c = get_color_mapping(q, 0, Scalar(quality.max_hi_quality), gui.color.cmaps[gui.color.current_map]);
			colors.push_back(c);
		}
	}

	glBindVertexArray(gl_mesh_in.vao_solid);
	glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_in.buffer);

	glBufferSubData(GL_ARRAY_BUFFER, gl_mesh_in.color_offset, colors.size() * sizeof(Vector4u8), colors.data());

	GLint color_location = glGetAttribLocation(solid_program, "color");
	glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_mesh_in.color_offset));
	glEnableVertexAttribArray(color_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

#include "smooth.h"
void GUIApplication::_init_gl_hi_colors_from_ambient_occlusion(int n_samples)
{
	Timer t;
	VectorX AO;
	compute_ambient_occlusion(s.hi.V, s.hi.F, s.hi.VN, AO, n_samples);
	std::cout << "AO took " << t.time_elapsed() << " seconds" << std::endl;

	VectorXu8 VB;
	per_vertex_border_flag(s.hi.V, s.hi.F, s.hi.VF, VB);

	for (int i = 0; i < 3; ++i)
		laplacian_smooth(AO, s.hi.F, s.hi.VF, VB);

	std::vector<Vector4u8> colors;
	for (int i = 0; i < (int)s.hi.F.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int vi = s.hi.F(i, j);
			Vector4u8 c = get_color_mapping(1 - std::pow(AO(vi), 2), 0, 1, ColorMap::GistGray);
			colors.push_back(c);
		}
	}

	glBindVertexArray(gl_mesh_in.vao_solid);
	glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_in.buffer);

	glBufferSubData(GL_ARRAY_BUFFER, gl_mesh_in.color_offset, colors.size() * sizeof(Vector4u8), colors.data());

	GLint color_location = glGetAttribLocation(solid_program, "color");
	glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_mesh_in.color_offset));
	glEnableVertexAttribArray(color_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_init_gl_base_colors_from_vertex_scalar_field(const VectorX& Q, Scalar min_q, Scalar max_q)
{
	Assert(Q.size() == s.base.V.rows());

	quality.distribution = Distribution();
	for (int vi = 0; vi < Q.size(); ++vi) {
		quality.distribution.add(Q(vi));
	}

	if (min_q >= max_q) {
		min_q = quality.distribution.min();
		max_q = quality.distribution.max();
	}

	std::vector<Vector4u8> colors;
	for (int fi = 0; fi < s.base.F.rows(); ++fi) {
		for (int i = 0; i < 3; ++i) {
			Vector4u8 c = get_color_mapping(-Q(s.base.F(fi, i)), -max_q, -min_q, gui.color.cmaps[gui.color.current_map]);
			colors.push_back(c);
		}
	}

	glBindVertexArray(gl_mesh.vao_solid);
	glBindBuffer(GL_ARRAY_BUFFER, gl_mesh.buffer);

	glBufferSubData(GL_ARRAY_BUFFER, gl_mesh.color_offset, colors.size() * sizeof(Vector4u8), colors.data());

	//GLint color_location = glGetAttribLocation(solid_program, "color");
	//glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_mesh_in.color_offset));
	//glEnableVertexAttribArray(color_location);
	//check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_init_gl_proxy_colors_from_vertex_area()
{
	check_gl_error();
	Distribution distrib = Distribution();
	for (int vi = 0; vi < (int)s.proxy.V.rows(); ++vi)
		distrib.add(std::log(s.proxy.QW[vi]));

	std::vector<uint8_t> colors;
	for (int i = 0; i < (int)s.proxy.F.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int vi = s.proxy.F(i, j);
			Scalar q = (std::log(s.proxy.QW[vi]) - distrib.min()) / (distrib.max() - distrib.min());
			colors.push_back(uint8_t((1 - q) * 255));
			colors.push_back(uint8_t((1 - q) * 255));
			colors.push_back(uint8_t(255));
			colors.push_back(uint8_t(255));
		}
	}

	glBindVertexArray(gl_mesh_proxy.vao_solid);
	check_gl_error();
	glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_proxy.buffer);
	check_gl_error();

	glBufferSubData(GL_ARRAY_BUFFER, gl_mesh_proxy.color_offset, colors.size() * sizeof(uint8_t), colors.data());
	check_gl_error();

	GLint color_location = glGetAttribLocation(solid_program, "color");
	check_gl_error();
	glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_mesh_proxy.color_offset));
	check_gl_error();
	glEnableVertexAttribArray(color_location);
	check_gl_error();

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_resize_offscreen_buffers()
{
	glDeleteTextures(1, &render_target.color_buffer);
	glDeleteTextures(1, &render_target.id_buffer);
	glDeleteTextures(1, &render_target.depth_buffer);
	glDeleteFramebuffers(1, &render_target.fbo);

	glGenFramebuffers(1, &render_target.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, render_target.fbo);

	glGenTextures(1, &render_target.color_buffer);
	glBindTexture(GL_TEXTURE_2D, render_target.color_buffer);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, window.width, window.height);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, render_target.color_buffer, 0);

	glGenTextures(1, &render_target.id_buffer);
	glBindTexture(GL_TEXTURE_2D, render_target.id_buffer);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, window.width, window.height);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, render_target.id_buffer, 0);

	glGenTextures(1, &render_target.depth_buffer);
	glBindTexture(GL_TEXTURE_2D, render_target.depth_buffer);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, window.width, window.height);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, render_target.depth_buffer, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	
	check_gl_error();

	render_target.draw_buffers[0] = GL_COLOR_ATTACHMENT0;
	render_target.draw_buffers[1] = GL_COLOR_ATTACHMENT1;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	_require_full_draw();
}

void GUIApplication::_require_full_draw()
{
	_full_draw = true;
}

void GUIApplication::_update_transforms()
{
	control.update_camera_transform();
}

void GUIApplication::_draw_offscreen()
{
	// these are needed to ensure the png alpha and colors are correct
	glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_COLOR, GL_DST_COLOR);
	
	static const float depth_clear = 1.0f;
	static const float color_clear[] = { 0.33f, 0.33f, 0.33f, 1.0f };
	static const float color_clear_screenshot[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	static const int id_clear = -1;

	check_gl_error();

	glBindFramebuffer(GL_FRAMEBUFFER, render_target.fbo);
	check_gl_error();

	glDrawBuffers(2, render_target.draw_buffers);
	check_gl_error();

	glViewport(0, 0, window.width, window.height);
	check_gl_error();

	bool draw_mesh = _full_draw || gui.edit.active;

	if (draw_mesh) {
		glClearBufferfv(GL_DEPTH, 0, &depth_clear);

		if (!gui.screenshot.on)
			glClearBufferfv(GL_COLOR, 0, color_clear);
		else
			glClearBufferfv(GL_COLOR, 0, color_clear_screenshot);

		glClearBufferiv(GL_COLOR, 1, &id_clear);

		//if (!gui.screenshot.on)
		//	_draw_background();
	}

	check_gl_error();

	if (file_loaded) {

		glEnable(GL_DEPTH_TEST);

		if (draw_mesh) {
			if (ro.current_layer == RenderingOptions::RenderLayer_BaseMesh) {
				_draw_mesh(gl_mesh, ro.visible_op_status);
				if (ro.visible_direction_field)
					_draw_lines(gl_direction_field.base, ro.direction_field_color);
			}

			if (ro.current_layer == RenderingOptions::RenderLayer_InputMesh) {
				_draw_mesh(gl_mesh_in);
				if (ro.visible_direction_field)
					_draw_lines(gl_direction_field.hi, ro.direction_field_color);
			}

			if (ro.current_layer == RenderingOptions::RenderLayer_ProxyMesh) {
				_draw_mesh(gl_mesh_proxy);
				if (ro.visible_direction_field)
					_draw_lines(gl_direction_field.proxy, ro.direction_field_color);
			}

			if (ro.current_layer == RenderingOptions::RenderLayer_MicroMesh) {
				_draw_mesh(gl_umesh);
				if (gui.show_top_mesh)
					_draw_lines(gl_top_mesh, Vector4f(0.46f, 1.0f, 0.15f, 1.0f));
			}

			if (ro.visible_op_status && ro.current_layer == RenderingOptions::RenderLayer_BaseMesh) {
				for (auto& entry : operations.gl_edges)
					if (entry.second.n_lines > 0)
						_draw_lines(entry.second, color_from_u32_abgr(operations.colors[entry.first]));
			}
		}

		if (ro.render_border) {
			_draw_mesh_border(gl_mesh_border);
		}

		if (ro.visible_displacements && s.micromesh.base_fn > 0 && s.micromesh.is_displaced())
			_draw_lines(gl_displacements, ro.displacements_color);
		
		if (ro.visible_directions && s.micromesh.base_fn > 0)
			_draw_lines(gl_directions, ro.displacements_color);

		check_gl_error();

		// DEBUG
		glUseProgram(wire_program);
		glDrawBuffers(1, render_target.draw_buffers);

		check_gl_error();
		float red[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
		GLuint loc_color1 = glGetUniformLocation(wire_program, "color1");
		glUniform4fv(loc_color1, 1, red);

		check_gl_error();
		GLuint loc_model_mat = glGetUniformLocation(wire_program, "modelMatrix");
		GLuint loc_view_mat = glGetUniformLocation(wire_program, "viewMatrix");
		GLuint loc_proj_mat = glGetUniformLocation(wire_program, "projectionMatrix");
		glUniformMatrix4fv(loc_model_mat, 1, GL_FALSE, control.matrix.data());
		glUniformMatrix4fv(loc_view_mat, 1, GL_FALSE, control.camera.view.data());
		glUniformMatrix4fv(loc_proj_mat, 1, GL_FALSE, control.camera.projection.data());

		check_gl_error();
		if (gui.selection.fi != -1) {
			glBindVertexArray(gl_selection.vao);
			glDepthFunc(GL_LEQUAL);
			glDepthFunc(GL_ALWAYS);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, 1);
			glDepthFunc(GL_LESS);
		}

		if (gui.edit.active) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			GLuint loc_color1 = glGetUniformLocation(wire_program, "color1");
			float edit_color[4] = { 0.55f, 0.9f, 0.2f, 1.0f };
			glUniform4fv(loc_color1, 1, edit_color);

			glBindVertexArray(gl_edit.vao);
			
			glLineWidth(2);

			glDisable(GL_DEPTH_TEST);
			glDrawArrays(GL_TRIANGLES, 0, gl_edit.n);
			glDisable(GL_DEPTH_TEST);

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			glBindVertexArray(0);
			glUseProgram(0);
		}

		glBindVertexArray(0);
		glUseProgram(0);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	check_gl_error();

	_full_draw = false;
}

void GUIApplication::_draw_mesh_border(const GLMeshInfo& mesh)
{
	glDrawBuffers(1, render_target.draw_buffers);

	glUseProgram(wire_program);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GLuint loc_color1 = glGetUniformLocation(wire_program, "color1");
	glUniform4fv(loc_color1, 1, ro.border_wire_color.data());

	GLuint loc_model_mat = glGetUniformLocation(wire_program, "modelMatrix");
	GLuint loc_view_mat = glGetUniformLocation(wire_program, "viewMatrix");
	GLuint loc_proj_mat = glGetUniformLocation(wire_program, "projectionMatrix");
	glUniformMatrix4fv(loc_model_mat, 1, GL_FALSE, control.matrix.data());
	glUniformMatrix4fv(loc_view_mat, 1, GL_FALSE, control.camera.view.data());
	glUniformMatrix4fv(loc_proj_mat, 1, GL_FALSE, control.camera.projection.data());

	glBindVertexArray(mesh.vao_wire);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LINE_SMOOTH);
	glLineWidth(ro.wire_line_width);
	glDrawArrays(GL_TRIANGLES, 0, mesh.n_wire);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_BLEND);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);
	glUseProgram(0);
}

void GUIApplication::_draw_mesh(const GLMeshInfo& mesh, bool force_no_wireframe)
{
	glDrawBuffers(2, render_target.draw_buffers);

	glHint(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, GL_NICEST);
	check_gl_error();

	glUseProgram(solid_program);
	GLuint loc_metallic = glGetUniformLocation(solid_program, "metallic");
	glUniform1f(loc_metallic, ro.metallic);
	GLuint loc_roughness = glGetUniformLocation(solid_program, "roughness");
	glUniform1f(loc_roughness, ro.roughness);
	GLuint loc_ao = glGetUniformLocation(solid_program, "ao");
	glUniform1f(loc_ao, ro.ambient);
	GLuint loc_shading_weight = glGetUniformLocation(solid_program, "shadingWeight");
	glUniform1f(loc_shading_weight, ro.shading_weight);
	GLuint loc_flat = glGetUniformLocation(solid_program, "flatShading");
	glUniform1i(loc_flat, ro.flat ? 1 : 0);
	GLuint loc_mesh_color = glGetUniformLocation(solid_program, "meshColor");
	glUniform3fv(loc_mesh_color, 1, ro.color.data());
	GLuint loc_light_color = glGetUniformLocation(solid_program, "lightColor");
	glUniform3fv(loc_light_color, 1, ro.lightColor.data());
	GLuint loc_light_intensity = glGetUniformLocation(solid_program, "lightIntensity");
	glUniform1f(loc_light_intensity, ro.light_intensity);
	
	GLuint loc_use_color_attribute = glGetUniformLocation(solid_program, "useColorAttribute");
	glUniform1i(loc_use_color_attribute, mesh.use_color_attribute ? 1 : 0);

	GLuint loc_model_mat = glGetUniformLocation(solid_program, "modelMatrix");
	GLuint loc_view_mat = glGetUniformLocation(solid_program, "viewMatrix");
	GLuint loc_proj_mat = glGetUniformLocation(solid_program, "projectionMatrix");
	glUniformMatrix4fv(loc_model_mat, 1, GL_FALSE, control.matrix.data());
	glUniformMatrix4fv(loc_view_mat, 1, GL_FALSE, control.camera.view.data());
	glUniformMatrix4fv(loc_proj_mat, 1, GL_FALSE, control.camera.projection.data());

	GLuint loc_camera_pos = glGetUniformLocation(solid_program, "cameraPos");
	glUniform3fv(loc_camera_pos, 1, control.camera.eye.data());
	
	GLuint loc_light_dir = glGetUniformLocation(solid_program, "lightDir");
	Vector3f light_dir(0, 0, 1);
	glUniform3fv(loc_light_dir, 1, light_dir.data());

	check_gl_error();

	glBindVertexArray(mesh.vao_solid);
	glDrawArrays(GL_TRIANGLES, 0, mesh.n_solid);

	glBindVertexArray(0);
	glUseProgram(0);

	if (ro.wire && !force_no_wireframe) {
		glDrawBuffers(1, render_target.draw_buffers);
		glUseProgram(wire_program);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		GLuint loc_color1 = glGetUniformLocation(wire_program, "color1");
		glUniform4fv(loc_color1, 1, ro.wire_color.data());

		GLuint loc_model_mat = glGetUniformLocation(wire_program, "modelMatrix");
		GLuint loc_view_mat = glGetUniformLocation(wire_program, "viewMatrix");
		GLuint loc_proj_mat = glGetUniformLocation(wire_program, "projectionMatrix");
		glUniformMatrix4fv(loc_model_mat, 1, GL_FALSE, control.matrix.data());
		glUniformMatrix4fv(loc_view_mat, 1, GL_FALSE, control.camera.view.data());
		glUniformMatrix4fv(loc_proj_mat, 1, GL_FALSE, control.camera.projection.data());

		check_gl_error();
		glBindVertexArray(mesh.vao_wire);
		glDepthFunc(GL_LEQUAL);
		glEnable(GL_BLEND);
		glEnable(GL_LINE_SMOOTH);
		
	
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glLineWidth(ro.wire_line_width);
		glDrawArrays(GL_TRIANGLES, 0, mesh.n_wire);
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_BLEND);
		glDepthFunc(GL_LESS);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		glBindVertexArray(0);
		glUseProgram(0);
	}
}

void GUIApplication::_draw_mesh(const GLSubdivisionMeshInfo& umesh)
{
	glDrawBuffers(1, render_target.draw_buffers);

	glUseProgram(solid_displacement_program);
	GLuint loc_metallic = glGetUniformLocation(solid_displacement_program, "metallic");
	glUniform1f(loc_metallic, ro.metallic);
	GLuint loc_roughness = glGetUniformLocation(solid_displacement_program, "roughness");
	glUniform1f(loc_roughness, ro.roughness);
	GLuint loc_ao = glGetUniformLocation(solid_displacement_program, "ao");
	glUniform1f(loc_ao, ro.ambient);
	GLuint loc_shading_weight = glGetUniformLocation(solid_displacement_program, "shadingWeight");
	glUniform1f(loc_shading_weight, ro.shading_weight);
	GLuint loc_flat = glGetUniformLocation(solid_displacement_program, "flatShading");
	glUniform1i(loc_flat, ro.flat ? 1 : 0);
	GLuint loc_mesh_color = glGetUniformLocation(solid_displacement_program, "meshColor");
	glUniform3fv(loc_mesh_color, 1, ro.color.data());
	GLuint loc_light_color = glGetUniformLocation(solid_displacement_program, "lightColor");
	glUniform3fv(loc_light_color, 1, ro.lightColor.data());
	GLuint loc_light_intensity = glGetUniformLocation(solid_displacement_program, "lightIntensity");
	glUniform1f(loc_light_intensity, ro.light_intensity);
	
	GLuint loc_use_color_attribute = glGetUniformLocation(solid_displacement_program, "useColorAttribute");
	glUniform1i(loc_use_color_attribute, umesh.use_color_attribute ? 1 : 0);

	GLuint loc_model_mat = glGetUniformLocation(solid_displacement_program, "modelMatrix");
	GLuint loc_view_mat = glGetUniformLocation(solid_displacement_program, "viewMatrix");
	GLuint loc_proj_mat = glGetUniformLocation(solid_displacement_program, "projectionMatrix");
	glUniformMatrix4fv(loc_model_mat, 1, GL_FALSE, control.matrix.data());
	glUniformMatrix4fv(loc_view_mat, 1, GL_FALSE, control.camera.view.data());
	glUniformMatrix4fv(loc_proj_mat, 1, GL_FALSE, control.camera.projection.data());

	GLuint loc_camera_pos = glGetUniformLocation(solid_displacement_program, "cameraPos");
	glUniform3fv(loc_camera_pos, 1, control.camera.eye.data());
	
	GLuint loc_light_dir = glGetUniformLocation(solid_displacement_program, "lightDir");
	Vector3f light_dir(0, 0, 1);
	glUniform3fv(loc_light_dir, 1, light_dir.data());

	GLuint loc_displacement_scale = glGetUniformLocation(solid_displacement_program, "displacementScale");
	glUniform1f(loc_displacement_scale, ro.displacement_scale);

	check_gl_error();
	glBindVertexArray(umesh.vao_solid);
	check_gl_error();
	glDrawArrays(GL_TRIANGLES, 0, umesh.n_solid);
	check_gl_error();

	glBindVertexArray(0);
	glUseProgram(0);

	if (ro.wire) {
		glUseProgram(wire_displacement_program);

		check_gl_error();
		GLuint loc_model_mat = glGetUniformLocation(wire_displacement_program, "modelMatrix");
		GLuint loc_view_mat = glGetUniformLocation(wire_displacement_program, "viewMatrix");
		GLuint loc_proj_mat = glGetUniformLocation(wire_displacement_program, "projectionMatrix");
		check_gl_error();
		glUniformMatrix4fv(loc_model_mat, 1, GL_FALSE, control.matrix.data());
		glUniformMatrix4fv(loc_view_mat, 1, GL_FALSE, control.camera.view.data());
		glUniformMatrix4fv(loc_proj_mat, 1, GL_FALSE, control.camera.projection.data());
		check_gl_error();

		GLuint loc_displacement_scale = glGetUniformLocation(wire_displacement_program, "displacementScale");
		glUniform1f(loc_displacement_scale, ro.displacement_scale);

		// draw border edges

		check_gl_error();
		glBindVertexArray(umesh.vao_wire_border);

		check_gl_error();
		GLuint loc_border_color1 = glGetUniformLocation(wire_displacement_program, "color1");
		check_gl_error();
		glUniform4fv(loc_border_color1, 1, ro.wire_color.data());

		check_gl_error();
		glDepthFunc(GL_LEQUAL);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_LINE_SMOOTH);
		glLineWidth(ro.wire_line_width * 2);
		check_gl_error();
		glDrawArrays(GL_LINES, 0, umesh.n_wire_border);
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_BLEND);
		glDepthFunc(GL_LESS);

		// draw inner edges

		if (ro.draw_umesh_wire) {

			check_gl_error();
			glBindVertexArray(umesh.vao_wire_inner);

			GLuint loc_inner_color1 = glGetUniformLocation(wire_displacement_program, "color1");
			glUniform4fv(loc_inner_color1, 1, ro.wire_color2.data());

			glDepthFunc(GL_LEQUAL);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_LINE_SMOOTH);
			glLineWidth(ro.wire_line_width * 0.5f);
			glDrawArrays(GL_LINES, 0, umesh.n_wire_inner);
			glDisable(GL_LINE_SMOOTH);
			glDisable(GL_BLEND);
			glDepthFunc(GL_LESS);

			check_gl_error();
		}

		glBindVertexArray(0);
		glUseProgram(0);
	}
}

void GUIApplication::_draw_lines(const GLLineInfo& lines, const Vector4f& color)
{
	glDrawBuffers(1, render_target.draw_buffers);

	glUseProgram(wire_program);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GLuint loc_color1 = glGetUniformLocation(wire_program, "color1");
	glUniform4fv(loc_color1, 1, color.data());

	GLuint loc_model_mat = glGetUniformLocation(wire_program, "modelMatrix");
	GLuint loc_view_mat = glGetUniformLocation(wire_program, "viewMatrix");
	GLuint loc_proj_mat = glGetUniformLocation(wire_program, "projectionMatrix");
	glUniformMatrix4fv(loc_model_mat, 1, GL_FALSE, control.matrix.data());
	glUniformMatrix4fv(loc_view_mat, 1, GL_FALSE, control.camera.view.data());
	glUniformMatrix4fv(loc_proj_mat, 1, GL_FALSE, control.camera.projection.data());

	check_gl_error();
	glBindVertexArray(lines.vao_line);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LINE_SMOOTH);
	glLineWidth(ro.wire_line_width * 2);
	glDrawArrays(GL_LINES, 0, 2 * lines.n_lines);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_BLEND);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindVertexArray(0);
	glUseProgram(0);
	
	check_gl_error();
}

void GUIApplication::_draw_onscreen()
{
	glDrawBuffer(GL_BACK);
	glViewport(0, 0, window.width, window.height);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	check_gl_error();
	glUseProgram(quad_program);
	glBindVertexArray(render_target.quad_vao);
	check_gl_error();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, render_target.color_buffer);
	GLuint loc_offscreen = glGetUniformLocation(quad_program, "offscreen");
	glUniform1i(loc_offscreen, 0);

	check_gl_error();

	glDisable(GL_DEPTH_TEST);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glEnable(GL_DEPTH_TEST);

	glBindVertexArray(0);
	check_gl_error();
}

void GUIApplication::_draw_background()
{
	glClearBufferfv(GL_COLOR, 0, window.background0.data());

	int c_x = window.width / 8;
	int c_y = window.height / 6;
	glEnable(GL_SCISSOR_TEST);
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 8; ++j) {
			if ((i * 4 + j) % 2 == (i % 2)) {
				glScissor(j * c_x, i * c_y, c_x, c_y);
				glClearBufferfv(GL_COLOR, 0, window.background1.data());
			}
		}
	}
	glDisable(GL_SCISSOR_TEST);
}

void GUIApplication::_get_mouse_ray(Vector3* o, Vector3* dir) const
{
	int m_x = window.mx;
	int m_y = window.height - window.my;

	Vector2f screen_origin(window.width / 2.0f, window.height / 2.0f);
	Vector2f screen_coord = Vector2f(window.mx, window.height - window.my) - screen_origin;
	screen_coord.x() /= (window.width / 2.0f);
	screen_coord.y() /= (window.height / 2.0f);
	Vector3f image_plane_pos = screen_to_image_plane(control.camera, screen_coord);

	*o = control.camera.eye.cast<Scalar>();
	*dir = (image_plane_pos - control.camera.eye).normalized().cast<Scalar>();
}

void GUIApplication::select_edge_near_cursor()
{
	//if (ro.current_layer != RenderingOptions::RenderLayer_InputMesh)
	if (ro.current_layer != RenderingOptions::RenderLayer_BaseMesh)
		return;

	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);
	
	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		Vector3 p = o + isect.t * dir;

		Vector3 v0 = s.base.V.row(s.base.F(isect.fi, 0));
		Vector3 v1 = s.base.V.row(s.base.F(isect.fi, 1));
		Vector3 v2 = s.base.V.row(s.base.F(isect.fi, 2));

		int e = nearest_edge(p, v0, v1, v2);

		gui.selection.fi = isect.fi;
		gui.selection.e = e;

		Edge edge(s.base.F(gui.selection.fi, e), s.base.F(gui.selection.fi, (e + 1) % 3));

		Vector3 e0 = s.base.V.row(edge.first);
		Vector3 e1 = s.base.V.row(edge.second);

		glBindBuffer(GL_ARRAY_BUFFER, gl_selection.buffer);
		Vector3f* buf = (Vector3f*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		buf[0] = e0.cast<float>();
		buf[1] = e1.cast<float>();
		buf[2] = e0.cast<float>();
		buf[3] = e1.cast<float>();
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else {
		gui.selection.fi = -1;
		gui.selection.e = 0;
		gui.selection.status = 0;

		Vector3 z(0, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, gl_selection.buffer);
		Vector3f* buf = (Vector3f*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		buf[0] = z.cast<float>();
		buf[1] = z.cast<float>();
		buf[2] = z.cast<float>();
		buf[3] = z.cast<float>();
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void GUIApplication::select_vertex_near_cursor()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);
	
	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		Vector3 p = o + isect.t * dir;

		Vector3 v0 = s.base.V.row(s.base.F(isect.fi, 0));
		Vector3 v1 = s.base.V.row(s.base.F(isect.fi, 1));
		Vector3 v2 = s.base.V.row(s.base.F(isect.fi, 2));

		int e = nearest_vertex(p, v0, v1, v2);

		gui.selection.fi = isect.fi;
		gui.selection.e = e;

		int deg = s.base.F.cols();

		Vector3 e0 = s.base.V.row(s.base.F(isect.fi, e));
		Vector3 e1 = s.base.V.row(s.base.F(isect.fi, (e + 1) % deg));

		glBindBuffer(GL_ARRAY_BUFFER, gl_selection.buffer);
		Vector3f* buf = (Vector3f*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		buf[0] = e0.cast<float>();
		buf[1] = (e0 + 0.2 * (e1 - e0)).cast<float>();
		buf[2] = e0.cast<float>();
		buf[3] = (e0 + 0.2 * (e1 - e0)).cast<float>();
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else {
		gui.selection.fi = -1;
		gui.selection.e = 0;
		gui.selection.status = 0;

		Vector3 z(0, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, gl_selection.buffer);
		Vector3f* buf = (Vector3f*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		buf[0] = z.cast<float>();
		buf[1] = z.cast<float>();
		buf[2] = z.cast<float>();
		buf[3] = z.cast<float>();
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void GUIApplication::center_on_mouse_cursor()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);
	
	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		control.offset = (o + isect.t * dir).cast<float>();
	}
}

void GUIApplication::set_view_from_quality_rank()
{
	int i = quality.goto_rank - 1;
	if (quality.computed_mode != Quality::None && i >= 0 && i < (int)quality.points.size()) {
		std::cout << "Centering view on rank " << i << ", value = " << quality.points[i].second << ")" << std::endl;
		int layer;
		if (quality.computed_mode == Quality::DistanceInputToMicro)
			layer = RenderingOptions::RenderLayer_InputMesh;
		else if (quality.computed_mode == Quality::BaseVertexVisibility)
			layer = RenderingOptions::RenderLayer_BaseMesh;
		else
			layer = RenderingOptions::RenderLayer_MicroMesh;
		if (!gui.lock_view)
			ro.current_layer = layer;

		quality.current_mode = quality.computed_mode;
		control.offset = quality.points[i].first.cast<float>();
		control.r = 0.005 * s.hi.box.diagonal().norm();

		_require_full_draw();
	}
}

void GUIApplication::log_collapse_info_near_cursor() const
{
	if (s.micromesh.micro_fn > 0) {
		Vector3 o, dir;
		_get_mouse_ray(&o, &dir);

		IntersectionInfo isect;
		if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
			Vector3 p = o + isect.t * dir;

			Vector3 v0 = s.base.V.row(s.base.F(isect.fi, 0));
			Vector3 v1 = s.base.V.row(s.base.F(isect.fi, 1));
			Vector3 v2 = s.base.V.row(s.base.F(isect.fi, 2));

			int e = nearest_edge(p, v0, v1, v2);

			std::pair<Decimation::EdgeEntry, int> cdata = s.algo.handle->_get_collapse_data(isect.fi, e);
			s.algo.handle->log_collapse_data(cdata.first, cdata.second);
		}
	}
}

void GUIApplication::tweak_subdivision_level_under_cursor(int val)
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);

	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		_log_tweak_tessellation(isect.fi, val);
	}
}

void GUIApplication::flip_edge_near_cursor()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);

	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		Vector3 p = o + isect.t * dir;

		Vector3 v0 = s.base.V.row(s.base.F(isect.fi, 0));
		Vector3 v1 = s.base.V.row(s.base.F(isect.fi, 1));
		Vector3 v2 = s.base.V.row(s.base.F(isect.fi, 2));

		int e = nearest_edge(p, v0, v1, v2);

		_log_flip_edge(isect.fi, e);
	}
}

void GUIApplication::split_edge_near_cursor()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);

	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		Vector3 p = o + isect.t * dir;

		Vector3 v0 = s.base.V.row(s.base.F(isect.fi, 0));
		Vector3 v1 = s.base.V.row(s.base.F(isect.fi, 1));
		Vector3 v2 = s.base.V.row(s.base.F(isect.fi, 2));

		int e = nearest_edge(p, v0, v1, v2);

		_log_split_edge(isect.fi, e);
	}
}

void GUIApplication::split_vertex_near_cursor()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);
	
	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		Vector3 p = o + isect.t * dir;

		Vector3 v0 = s.base.V.row(s.base.F(isect.fi, 0));
		Vector3 v1 = s.base.V.row(s.base.F(isect.fi, 1));
		Vector3 v2 = s.base.V.row(s.base.F(isect.fi, 2));

		int fvi = nearest_vertex(p, v0, v1, v2);

		_log_split_vertex(s.base.F(isect.fi, fvi));
	}
}

void GUIApplication::_edit_init()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);

	IntersectionInfo isect;
	if (gui.base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		Vector3 p = o + isect.t * dir;

		Vector3 v0 = s.base.V.row(s.base.F(isect.fi, 0));
		Vector3 v1 = s.base.V.row(s.base.F(isect.fi, 1));
		Vector3 v2 = s.base.V.row(s.base.F(isect.fi, 2));

		gui.edit.active = true;
		gui.edit.offset = Vector3(0, 0, 0);
		gui.edit.vi = s.base.F(isect.fi, nearest_vertex(p, v0, v1, v2));

		gl_edit.n = 3 * s.base.VF[gui.edit.vi].size();

		glBindBuffer(GL_ARRAY_BUFFER, gl_edit.buffer);
		glBufferData(GL_ARRAY_BUFFER, gl_edit.n * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		_edit_update(); // init buffer
	}
}

// moves along a plane parallel to the view plane and passing through the vertex
void GUIApplication::_edit_update()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);

	Vector3 p = s.base.V.row(gui.edit.vi);
	Vector3 view_dir = (control.camera.target - control.camera.eye).cast<Scalar>();

	Scalar t = ray_plane_intersection(o, dir, p, view_dir);
	gui.edit.offset = (o + t * dir) - p;

	glBindBuffer(GL_ARRAY_BUFFER, gl_edit.buffer);
	Vector3f* buf = (Vector3f*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	Vector3f edited_pos = (s.base.V.row(gui.edit.vi) + gui.edit.offset.transpose()).cast<float>();

	for (const VFEntry& vfe : s.base.VF[gui.edit.vi]) {
		for (int i = 0; i < 3; ++i) {
			*buf++ = s.base.F(vfe.first, i) == gui.edit.vi
				? edited_pos
				: s.base.V.row(s.base.F(vfe.first, i)).cast<float>();
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GUIApplication::_edit_finalize()
{
	_log_move_vertex();
	gui.edit.active = false;
}

void GUIApplication::_base_mesh_changed(bool reset_directions, bool tessellate)
{
	int deg = s.base.F.cols();
	Assert(deg == 3);

	// update mesh buffer
	{
		std::vector<float> buffer;
		buffer.reserve(18 * s.base.F.rows());
		for (int fi = 0; fi < s.base.F.rows(); ++fi) {
			for (int j = 0; j < 3; ++j) {
				buffer.push_back(s.base.V.row(s.base.F(fi, j))(0));
				buffer.push_back(s.base.V.row(s.base.F(fi, j))(1));
				buffer.push_back(s.base.V.row(s.base.F(fi, j))(2));
				buffer.push_back(s.base.VN.row(s.base.F(fi, j))(0));
				buffer.push_back(s.base.VN.row(s.base.F(fi, j))(1));
				buffer.push_back(s.base.VN.row(s.base.F(fi, j))(2));
			}
		}

		glBindVertexArray(gl_mesh.vao_solid);
		glBindBuffer(GL_ARRAY_BUFFER, gl_mesh.buffer);
		glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float) + (3 * s.base.F.rows() * sizeof(uint32_t)), nullptr, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, buffer.size() * sizeof(float), buffer.data());
	
		gl_mesh.color_offset = buffer.size() * sizeof(float);
		gl_mesh.use_color_attribute = false;
		gl_mesh.n_solid = gl_mesh.n_wire = 3 * s.base.valid_fn;

		GLint color_location = glGetAttribLocation(solid_program, "color");
		glVertexAttribPointer(color_location, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, (void*)(gl_mesh.color_offset));
		check_gl_error();
		
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// update debug view (colored edges according to constraint that blocked the edge collapses)
	{
		// ABGR colors
		operations.colors[Decimation::OpStatus_Feasible] = 0xff000000;
		operations.colors[Decimation::OpStatus_Unknown] = 0xffc0c0c0;
		operations.colors[Decimation::OpStatus_FailTopology] = 0xff3c14dc;
		operations.colors[Decimation::OpStatus_FailNormals] = 0xff71b33c;
		operations.colors[Decimation::OpStatus_FailAspectRatio] = 0xff507fff;
		operations.colors[Decimation::OpStatus_FailGeometricError] = 0xffe16941;
		operations.colors[Decimation::OpStatus_FailVertexRingNormals] = 0xffd355ba;

		std::map<int, std::vector<Edge>> edges;
		for (const std::pair<Edge, Decimation::OpStatus>& entry : s.algo.handle->opstatus)
			edges[entry.second].push_back(entry.first);

		for (auto& entry : operations.gl_edges) {
			GLLineInfo& gl_edge_ops = entry.second;

			if (gl_edge_ops.vao_line) {
				glDeleteVertexArrays(1, &gl_edge_ops.vao_line);
				gl_edge_ops.vao_line = 0;
			}

			if (gl_edge_ops.buffer_line) {
				glDeleteBuffers(1, &gl_edge_ops.buffer_line);
				gl_edge_ops.buffer_line = 0;
			}

			gl_edge_ops.n_lines = 0;
		}

		for (const auto& entry : edges) {
			int status = entry.first;

			GLLineInfo& gl_edge_ops = operations.gl_edges[status];

			int vertex_size = 6 * sizeof(float);

			std::vector<float> lines;

			for (const Edge& e : entry.second) {
				Vector3 e0 = s.base.V.row(e.first);
				Vector3 e1 = s.base.V.row(e.second);
				lines.push_back(e0.x()); lines.push_back(e0.y()); lines.push_back(e0.z());
				lines.push_back(e1.x()); lines.push_back(e1.y()); lines.push_back(e1.z());
				gl_edge_ops.n_lines++;
			}

			glGenVertexArrays(1, &gl_edge_ops.vao_line);
			glBindVertexArray(gl_edge_ops.vao_line);

			glGenBuffers(1, &gl_edge_ops.buffer_line);
			glBindBuffer(GL_ARRAY_BUFFER, gl_edge_ops.buffer_line);
			glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);

			GLint wire_pos_location = glGetAttribLocation(wire_program, "position");
			glVertexAttribPointer(wire_pos_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(wire_pos_location);
			check_gl_error();

			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
	}

	_init_gl_direction_field(gl_direction_field.base, s.base.V, s.base.D);

	gui.base.bvh = BVHTree();
	gui.base.bvh.build_tree(&s.base.V, &s.base.F, &s.base.VN, 64);

	if (reset_directions)
		_log_set_displacement_dirs();

	if (tessellate)
		_log_tessellate();
}

void GUIApplication::_proxy_mesh_changed()
{
	//glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_proxy.buffer);
	//Vector3f *ptr = (Vector3f *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	//for (int fi = 0; fi < s.proxy.F.rows(); ++fi) {
	//	for (int j = 0; j < 3; ++j) {
	//		*ptr++ = s.proxy.V.row(s.proxy.F(fi, j)).cast<float>();
	//		*ptr++ = s.proxy.VN.row(s.proxy.F(fi, j)).cast<float>();
	//	}
	//}
	//Assert(glUnmapBuffer(GL_ARRAY_BUFFER));
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Apparently using mapped memory leads to slow rendering times when wireframe is enabled...
	{
		std::vector<float> buffer;
		buffer.reserve(18 * s.proxy.F.rows());
		for (int fi = 0; fi < s.proxy.F.rows(); ++fi) {
			for (int j = 0; j < 3; ++j) {
				buffer.push_back(s.proxy.V.row(s.proxy.F(fi, j))(0));
				buffer.push_back(s.proxy.V.row(s.proxy.F(fi, j))(1));
				buffer.push_back(s.proxy.V.row(s.proxy.F(fi, j))(2));
				buffer.push_back(s.proxy.VN.row(s.proxy.F(fi, j))(0));
				buffer.push_back(s.proxy.VN.row(s.proxy.F(fi, j))(1));
				buffer.push_back(s.proxy.VN.row(s.proxy.F(fi, j))(2));
			}
		}
		glBindBuffer(GL_ARRAY_BUFFER, gl_mesh_proxy.buffer);
		glBufferSubData(GL_ARRAY_BUFFER, 0, buffer.size() * sizeof(float), buffer.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		gl_mesh_proxy.n_solid = gl_mesh_proxy.n_wire = 3 * s.proxy.F.rows();
	}

	_init_gl_direction_field(gl_direction_field.proxy, s.proxy.V, s.proxy.D);
	//_init_gl_proxy_colors_from_vertex_area();
}

void GUIApplication::_set_base_directions(const MatrixX& VD)
{
	s.base.VD = VD;
	_init_gl_directions();
}

void GUIApplication::_update_umesh_subdivision()
{
	_init_gl_umesh();
}

void GUIApplication::_set_selected_dir_toward_eye()
{
	// TODO find the vertex directly here instead of calling select_vertex_near_cursor() first

	if (gui.selection.fi != -1) {
		int v_selection = s.base.F(gui.selection.fi, gui.selection.e);
		s.base.VD.row(v_selection) = (control.camera.eye.cast<Scalar>() - s.base.V.row(v_selection).transpose()).normalized();

		_log_tweak_displacement_dir(v_selection, s.base.VD.row(v_selection));
	}
}

void GUIApplication::_compute_quality()
{
	if (!(s.micromesh.micro_fn > 0)) {
		quality.current_mode = Quality::None;
	}

	quality.points.clear();
	quality.points.reserve(s.hi.V.rows());

	MatrixX V;
	MatrixXi F;
	MatrixX VN;

	gl_mesh.use_color_attribute = false;
	gl_mesh_in.use_color_attribute = false;
	gl_umesh.use_color_attribute = false;

	VectorX base_visibility;

	if (quality.current_mode != Quality::None) {
		// compute quality values
		switch (quality.current_mode) {
		case Quality::DistanceInputToMicro:
			s.micromesh.extract_mesh(V, F);
			VN = compute_vertex_normals(V, F);
			compute_vertex_quality_hausdorff_distance(s.hi.V, s.hi.VN, s.hi.F, V, VN, F, s.hi.VQ);
			_init_gl_hi_colors_from_vertex_quality();
			gl_mesh_in.use_color_attribute = true;
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_InputMesh;
			break;
		case Quality::MicrofaceAspect:
			s.micromesh.compute_face_quality_aspect_ratio();
			_init_gl_umesh_colors_from_face_quality();
			gl_umesh.use_color_attribute = true;
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
			break;
		case Quality::MicrofaceStretch:
			s.micromesh.compute_face_quality_stretch();
			_init_gl_umesh_colors_from_face_quality();
			gl_umesh.use_color_attribute = true;
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
			break;
		case Quality::MicrodisplacementDistance:
			s.micromesh.compute_vertex_quality_displacement_distance();
			_init_gl_umesh_colors_from_vertex_quality();
			gl_umesh.use_color_attribute = true;
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
			break;
		case Quality::BaseVertexVisibility:
			std::tie(std::ignore, base_visibility) = compute_optimal_visibility_directions(s.base.V, s.base.F, s.base.VF);
			_init_gl_base_colors_from_vertex_scalar_field(base_visibility, 0, 1);
			gl_mesh.use_color_attribute = true;
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_BaseMesh;
			break;
		default:
			Assert(0 && "Unreachable");
		}

		// fill points vector
		switch (quality.current_mode) {
		case Quality::DistanceInputToMicro:
			for (int i = 0; i < (int)s.hi.V.rows(); ++i)
				quality.points.push_back(std::make_pair(s.hi.V.row(i), s.hi.VQ(i)));
			break;
		case Quality::MicrofaceAspect:
		case Quality::MicrofaceStretch:
			for (const SubdivisionTri& st : s.micromesh.faces) {
				for (int i = 0; i < (int)st.F.rows(); ++i) {
					Vector3 p = barycenter(st.V.row(st.F(i, 0)), st.V.row(st.F(i, 1)), st.V.row(st.F(i, 2)));
					quality.points.push_back(std::make_pair(p, st.FQ(i)));
				}
			}
			break;
		case Quality::MicrodisplacementDistance:
			for (const SubdivisionTri& st : s.micromesh.faces) {
				for (int i = 0; i < (int)st.V.rows(); ++i) {
					quality.points.push_back(std::make_pair(st.V.row(i) + st.VD.row(i), st.VQ(i)));
				}
			}
			break;
		case Quality::BaseVertexVisibility:
			for (int i = 0; i < (int)s.base.V.rows(); ++i) {
				quality.points.push_back(std::make_pair(s.base.V.row(i), base_visibility(i)));
			}
			break;
		default:
			Assert(0 && "Unreachable");
		}

		typedef std::pair<Vector3, Scalar> QP;

		std::sort(quality.points.begin(), quality.points.end(), [](const QP& qp1, const QP& qp2) -> bool {
			return qp1.second < qp2.second;
			});
	}

	quality.computed_mode = quality.current_mode;

	_require_full_draw();
}

void GUIApplication::_screenshot()
{
	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	std::vector<uint8_t> pixels(window.width * window.height * 4, 255);

	glBindFramebuffer(GL_FRAMEBUFFER, render_target.fbo);
	glReadBuffer(render_target.draw_buffers[0]);
	glReadPixels(0, 0, window.width, window.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	uint32_t* p = reinterpret_cast<uint32_t*>(pixels.data());

	// set transparent background
	for (unsigned i = 0; i < pixels.size() / 4; ++i) {
		if (pixels[i * 4 + 3] < 255) {
			// de-muliply alpha when blending with background
			float alpha = pixels[i * 4 + 3] / 255.0f;
			for (int k = 0; k < 3; ++k) {
				float c = pixels[i * 4 + k] / 255.0f;
				pixels[i * 4 + k] = uint8_t((c / alpha) * 255);
			}
			//pixels[i * 4 + 3] = 255;
		}
	}

	std::string name = std::string(gui.screenshot.string) + std::to_string(gui.screenshot.counter++) + ".png";

	stbi_flip_vertically_on_write(1);

	int status = stbi_write_png(name.c_str(), window.width, window.height, 4, pixels.data(), 0);
	if (status != 0) {
		std::cerr << "Written " << std::quoted(name) << std::endl;
	}
	else {
		std::cerr << "ERROR writing " << std::quoted(name) << std::endl;
	}
}

void GUIApplication::_log_init_proxy()
{
	namespace cap = cmdargs::init_proxy;

	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::InitProxy];

	const DecimationParameters& dparams = gui.proxy.dparams;

	if (gui.proxy.fn_flag)
		cmd.args.push_back(CMDARG_INTEGER(cap::min_fn, gui.proxy.fn_target));

	if (gui.proxy.smoothing_iterations > 0)
		cmd.args.push_back(CMDARG_INTEGER(cap::smoothing_iterations, gui.proxy.smoothing_iterations));

	if (gui.proxy.anisotropic_smoothing_iterations > 0) {
		cmd.args.push_back(CMDARG_INTEGER(cap::anisotropic_smoothing_iterations, gui.proxy.anisotropic_smoothing_iterations));
		cmd.args.push_back(CMDARG_SCALAR(cap::anisotropic_smoothing_weight, gui.proxy.anisotropic_smoothing_weight));
	}

	if (dparams.use_vertex_smoothing)
		cmd.args.push_back(CMDARG_SCALAR(cap::vertex_smoothing, dparams.smoothing_coefficient));

	cmd.args.push_back(CMDARG_SCALAR(cap::border_multiplier, dparams.border_error_scaling));
	cmd.args.push_back(CMDARG_SCALAR(cap::aspect_multiplier, dparams.ar_scaling));
	cmd.args.push_back(CMDARG_SCALAR(cap::visibility_multiplier, dparams.visibility_scaling));
	cmd.args.push_back(CMDARG_SCALAR(cap::normals_multiplier, dparams.normals_scaling));

	if (dparams.bound_geometric_error)
		cmd.args.push_back(CMDARG_SCALAR(cap::max_error, dparams.max_relative_error));

	if (dparams.bound_aspect_ratio)
		cmd.args.push_back(CMDARG_SCALAR(cap::min_aspect_ratio, dparams.min_aspect_ratio));

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_load_base_mesh(const std::string& file_path)
{
	namespace cal = cmdargs::load_base_mesh;

	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::LoadBaseMesh];

	CommandArgument path_arg;
	path_arg.name = cal::path;
	std::snprintf(path_arg.value.string_val, CommandArgument::MAX_STRING_SIZE, "%s", file_path.c_str());

	cmd.args.push_back(path_arg);

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_decimate()
{
	namespace cad = cmdargs::decimate;

	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::Decimate];
	
	const DecimationParameters& dparams = gui.base.decimation.dparams;

	if (gui.proxy.enabled)
		cmd.args.push_back(CMDARG_BOOL(cad::proxy, true));

	if (dparams.use_vertex_smoothing)
		cmd.args.push_back(CMDARG_SCALAR(cad::vertex_smoothing, dparams.smoothing_coefficient));

	cmd.args.push_back(CMDARG_SCALAR(cad::border_multiplier, dparams.border_error_scaling));
	cmd.args.push_back(CMDARG_SCALAR(cad::aspect_multiplier, dparams.ar_scaling));
	cmd.args.push_back(CMDARG_SCALAR(cad::visibility_multiplier, dparams.visibility_scaling));
	cmd.args.push_back(CMDARG_SCALAR(cad::normals_multiplier, dparams.normals_scaling));

	if (gui.base.decimation.fn_flag)
		cmd.args.push_back(CMDARG_INTEGER(cad::min_fn, gui.base.decimation.fn_target));

	if (dparams.bound_geometric_error)
		cmd.args.push_back(CMDARG_SCALAR(cad::max_error, dparams.max_relative_error));

	if (dparams.bound_aspect_ratio)
		cmd.args.push_back(CMDARG_SCALAR(cad::min_aspect_ratio, dparams.min_aspect_ratio));

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_tessellate()
{
	namespace cat = cmdargs::tessellate;

	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::Tessellate];

	CommandArgument mode_arg;
	mode_arg.name = cat::mode;

	switch (gui.tessellate.current_mode) {
	case 0: // constant
		std::snprintf(mode_arg.value.string_val, CommandArgument::MAX_STRING_SIZE, "%s", cat::modes::constant.c_str());
		cmd.args.push_back(mode_arg);
		cmd.args.push_back(CMDARG_INTEGER(cat::level, gui.tessellate.level));
		break;
	case 1: // uniform size
		std::snprintf(mode_arg.value.string_val, CommandArgument::MAX_STRING_SIZE, "%s", cat::modes::uniform.c_str());
		cmd.args.push_back(mode_arg);
		cmd.args.push_back(CMDARG_SCALAR(cat::microexpansion, gui.tessellate.microexpansion));
		cmd.args.push_back(CMDARG_INTEGER(cat::max_level, gui.tessellate.max_level));
		break;
	case 2: // adaptive
		std::snprintf(mode_arg.value.string_val, CommandArgument::MAX_STRING_SIZE, "%s", cat::modes::adaptive.c_str());
		cmd.args.push_back(mode_arg);
		if (gui.tessellate.adaptive.target == gui.tessellate.TARGET_PRIMITIVE_COUNT) {
			cmd.args.push_back(CMDARG_SCALAR(cat::microexpansion, gui.tessellate.microexpansion));
		}
		else {
			Assert(gui.tessellate.adaptive.target == gui.tessellate.TARGET_ERROR);
			cmd.args.push_back(CMDARG_SCALAR(cat::max_error, gui.tessellate.max_error));
		}
		cmd.args.push_back(CMDARG_INTEGER(cat::max_level, gui.tessellate.max_level));
		break;
	default:
		break;
	}

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_set_displacement_dirs()
{
	std::string disp_type;

	switch (gui.directions.current_type) {
	case DirectionsType::MaximalVisibility:
		disp_type = cmdargs::set_displacement_dirs::types::max_visibility;
		break;
	case DirectionsType::BaseVertexNormals:
		disp_type = cmdargs::set_displacement_dirs::types::normals;
		break;
	case DirectionsType::BaseVertexNormalsWithTangentsOnBorder:
		disp_type = cmdargs::set_displacement_dirs::types::tangent;
		break;
	default:
		Assert(0 && "Unreachable");
	}

	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::SetDisplacementDirs];
	CommandArgument arg;
	arg.name = cmdargs::set_displacement_dirs::type;
	std::snprintf(arg.value.string_val, CommandArgument::MAX_STRING_SIZE, "%s", disp_type.c_str());
	cmd.args.push_back(arg);

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_displace()
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::Displace];

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_tweak_tessellation(int fi, int delta)
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::TweakTessellation];
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::tweak_tessellation::base_fi, fi));
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::tweak_tessellation::delta, delta));

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_tweak_displacement_dir(int vi, Vector3 dir)
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::TweakDisplacementDir];
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::tweak_displacement_dir::base_vi, vi));

	CommandArgument arg;
	arg.name = cmdargs::tweak_displacement_dir::direction;
	for (int i = 0; i < 3; ++i)
		arg.value.vector3_val[i] = double(dir(i));
	cmd.args.push_back(arg);

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_optimize_base_topology()
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::OptimizeBaseTopology];

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_optimize_base_positions(const std::string& optim_mode)
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::OptimizeBasePositions];
	CommandArgument arg;
	arg.name = cmdargs::optimize_base_positions::mode;
	std::snprintf(arg.value.string_val, CommandArgument::MAX_STRING_SIZE, "%s", optim_mode.c_str());
	cmd.args.push_back(arg);

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_minimize_prismoids()
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::MinimizePrismoids];

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_reset_tessellation_offsets()
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::ResetTessellationOffsets];
	CommandArgument arg;
	arg.name = cmdargs::optimize_base_positions::mode;

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_flip_edge(int fi, int e)
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::FlipEdge];
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::flip_edge::base_fi, fi));
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::flip_edge::edge, e));

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_split_edge(int fi, int e)
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::SplitEdge];
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::split_edge::base_fi, fi));
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::split_edge::edge, e));

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_move_vertex()
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::MoveVertex];
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::move_vertex::base_vi, gui.edit.vi));

	CommandArgument arg;
	arg.name = cmdargs::move_vertex::position;
	Vector3 pos = s.base.V.row(gui.edit.vi) + gui.edit.offset.transpose();
	for (int i = 0; i < 3; ++i)
		arg.value.vector3_val[i] = double(pos(i));
	cmd.args.push_back(arg);

	cmd_queue.push_back(cmd);
}

void GUIApplication::_log_split_vertex(int vi)
{
	SessionCommand cmd;
	cmd.name = CommandStrings[CommandType::SplitVertex];
	cmd.args.push_back(CMDARG_INTEGER(cmdargs::split_vertex::base_vi, vi));

	cmd_queue.push_back(cmd);
}

void GUIApplication::_execute_pending_command()
{
	while (cmd_queue.size() > 0) {
		SessionCommand cmd = cmd_queue.front();
		cmd_queue.pop_front();

		s.execute(cmd);
		sequence.add_command(cmd);
		cmd_strings.push_back(cmd.to_string());

		Assert(Commands.find(cmd.name) != Commands.end());
		CommandType cmd_type = Commands[cmd.name];
		
		switch (cmd_type) {
		case InitProxy:
			_proxy_mesh_changed();
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_ProxyMesh;
			break;
		case LoadBaseMesh:
			_base_mesh_changed();
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_ProxyMesh;
			break;
		case Decimate:
			_base_mesh_changed();
			break;
		case Tessellate:
			_init_gl_umesh();
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
			gui.micromesh_exists = true;
			break;
		case SetDisplacementDirs:
			_init_gl_directions();
			break;
		case Displace:
			_init_gl_umesh();
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
			break;
		case TweakTessellation:
			_init_gl_umesh();
			if (!gui.lock_view)
				ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
			break;
		case TweakDisplacementDir:
			_init_gl_directions();
			if (s.micromesh.is_displaced()) {
				_init_gl_umesh();
				if (!gui.lock_view)
					ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
			}
			break;
		case OptimizeBaseTopology:
			_base_mesh_changed();
			break;
		case OptimizeBasePositions:
			_base_mesh_changed();
			break;
		case MinimizePrismoids:
			_init_gl_directions();
			_base_mesh_changed(false, false);
			_init_gl_umesh();
			break;
		case ResetTessellationOffsets:
			//_init_gl_umesh();
			break;
		case FlipEdge:
			_base_mesh_changed();
			break;
		case SplitEdge:
			_base_mesh_changed();
			break;
		case MoveVertex:
			_base_mesh_changed();
			break;
		case SplitVertex:
			_base_mesh_changed();
			break;
		case Save:
			break;
		default:
			std::cerr << "Unrecognized command " << cmd.name << std::endl; // should never happen
		}

		_require_full_draw();

	}
}

