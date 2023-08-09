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
#include "gui_view.h"
#include "camera.h"
#include "gl_utils.h"
#include "utils.h"
#include "mesh_io.h"
#include "mesh_io_gltf.h"
#include "tangent.h"
#include "intersection.h"
#include "mesh_utils.h"

#include "quality.h"
#include "color.h"

#include "micro.h"
#include "bvh.h"

#include "shader_strings.h"

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

	bool show_demo_window = true;
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

		//ImGui::ShowDemoWindow(&show_demo_window);

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
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	
	glfwDestroyWindow(window.handle);

	glfwTerminate();
}

void GUIApplication::load_mesh(const std::string& meshfile, bool reset_controls)
{
	GLTFReadInfo read_micromesh;
	if (!read_gltf(meshfile, read_micromesh)) {
		std::cerr << "Error reading gltf file " << meshfile << std::endl;
		return;
	}

	if (!read_micromesh.has_subdivision_mesh()) {
		std::cerr << "gltf file does not contain micromesh data" << std::endl;
		return;
	}

	base.V = read_micromesh.get_vertices();
	base.F = read_micromesh.get_faces();

	umesh = read_micromesh.get_subdivision_mesh();

	base.box = Box3();
	for (int i = 0; i < base.V.rows(); ++i)
		base.box.add(base.V.row(i));

	SubdivisionMesh umesh = read_micromesh.get_subdivision_mesh();

	if (reset_controls)
		_init_transforms();

	//gl_mesh.use_color_attribute = false;
	//gl_mesh.color_offset = 0;
	//gl_mesh_in.use_color_attribute = false;
	//gl_mesh_in.color_offset = 0;
	_gl_umesh.use_color_attribute = false;
	_gl_umesh.color_offset = 0;


	//quality.current_mode = Quality::None;
	//quality.computed_mode = Quality::None;
	//quality.goto_rank = 1;
	//quality.distribution = Distribution();

	ro.current_layer = RenderingOptions::RenderLayer_MicroMesh;
	ro.flat = true;

	_init_gl_buffers(_gl_umesh);
	
	file_loaded = true;

	base.bvh = BVHTree();
	base.bvh.build_tree(&base.V, &base.F, nullptr);

	_require_full_draw();
}

void GUIApplication::_init_transforms()
{
	control.offset = base.box.center().cast<float>();

	//control.theta = 0.0f;
	//control.phi = M_PI_2;
	control.r = base.box.diagonal().norm();

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

	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontFromMemoryCompressedTTF(font_droidsans_compressed_data, font_droidsans_compressed_size, 15);

	ImGui_ImplGlfw_InitForOpenGL(window.handle, true);
	ImGui_ImplOpenGL3_Init();

	ImGuiStyle& style = ImGui::GetStyle();
	style.FrameRounding = 2.0f;
	style.GrabRounding = style.FrameRounding;

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

	std::string title = std::string("Micro-Mesh Previewer");

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

void GUIApplication::_init_gl_buffers(const MatrixX& V, const MatrixXi& F, GLMeshInfo& gl_mesh) const
{
	MatrixX FN = compute_face_normals(V, F);
	// generate buffer data
	int vertex_size = 6 * sizeof(float); // position and normal
	std::vector<float> buffer_data;
	buffer_data.reserve(3 * F.rows() * 6);
	for (int i = 0; i < F.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 p = V.row(F(i, j));
			Vector3 n = FN.row(i);
			buffer_data.push_back(p.x());
			buffer_data.push_back(p.y());
			buffer_data.push_back(p.z());
			buffer_data.push_back(n.x());
			buffer_data.push_back(n.y());
			buffer_data.push_back(n.z());
		}
	}

	// setup mesh buffers
	if (!gl_mesh.buffer)
		glGenBuffers(1, &gl_mesh.buffer);

	// pre-allocate color storage
	glBindBuffer(GL_ARRAY_BUFFER, gl_mesh.buffer);
	glBufferData(GL_ARRAY_BUFFER, buffer_data.size() * sizeof(float) + (3 * F.rows() * sizeof(uint32_t)), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_data.size() * sizeof(float), buffer_data.data());
	gl_mesh.color_offset = buffer_data.size() * sizeof(float);

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
	
	gl_mesh.n_solid = gl_mesh.n_wire = (int)3 * F.rows();

	check_gl_error();
}

void GUIApplication::_init_gl_buffers(GLSubdivisionMeshInfo& gl_umesh) const
{
	if (gl_umesh.vao_solid)
		glDeleteVertexArrays(1, &gl_umesh.vao_solid);
	if (gl_umesh.vao_wire_border)
		glDeleteVertexArrays(1, &gl_umesh.vao_wire_border);
	if (gl_umesh.vao_wire_inner)
		glDeleteVertexArrays(1, &gl_umesh.vao_wire_inner);

	constexpr int vertex_size = 9 * sizeof(float);
	constexpr int wire_vertex_size = 6 * sizeof(float);

	std::vector<float> solid_buffer;
	std::vector<float> wire_buffer_border;
	std::vector<float> wire_buffer_inner;

	solid_buffer.reserve(uint64_t(3) * umesh.micro_fn * 9);
	wire_buffer_border.reserve(uint64_t(3) * umesh.micro_fn * 6);
	wire_buffer_inner.reserve(umesh.micro_fn);

	gl_umesh.fn = 0;
	for (const SubdivisionTri& st : umesh.faces) {
		BarycentricGrid bary_grid(st.subdivision_level());

		MatrixX FN = compute_face_normals(st.V + st.VD, st.F);

		int deg = st.F.cols();
		Assert(deg == 3);
		for (int fi = 0; fi < st.F.rows(); ++fi) {
			gl_umesh.fn++;
			for (int j = 0; j < deg; ++j) {
				Vector3 p = st.V.row(st.F(fi, j));
				Vector3 n = FN.row(fi);
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

	gl_umesh.use_color_attribute = false;

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
	static const float color_clear[] = { 0.29f, 0.29f, 0.29f, 1.0f };
	static const float color_clear_screenshot[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	static const int id_clear = -1;

	check_gl_error();

	glBindFramebuffer(GL_FRAMEBUFFER, render_target.fbo);
	check_gl_error();

	glDrawBuffers(2, render_target.draw_buffers);
	check_gl_error();

	glViewport(0, 0, window.width, window.height);
	check_gl_error();

	bool draw_mesh = _full_draw;

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
				_draw_mesh(_gl_mesh_base);
			}

			if (ro.current_layer == RenderingOptions::RenderLayer_InputMesh) {
				_draw_mesh(_gl_mesh_input);
			}

			if (ro.current_layer == RenderingOptions::RenderLayer_MicroMesh) {
				_draw_mesh(_gl_umesh);
			}

		}

		check_gl_error();

		glBindVertexArray(0);
		glUseProgram(0);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	check_gl_error();

	_full_draw = false;
}

void GUIApplication::_draw_mesh(const GLMeshInfo& mesh)
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

	if (ro.wire) {
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

		glBindVertexArray(umesh.vao_wire_border);

		GLuint loc_border_color1 = glGetUniformLocation(wire_displacement_program, "color1");
		glUniform4fv(loc_border_color1, 1, ro.wire_color.data());

		check_gl_error();

		glDepthFunc(GL_LEQUAL);
		glEnable(GL_BLEND);
		glEnable(GL_LINE_SMOOTH);

		glLineWidth(ro.wire_line_width);
		glDrawArrays(GL_LINES, 0, umesh.n_wire_border);

		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_BLEND);
		glDepthFunc(GL_LESS);

		// draw inner edges

		check_gl_error();
		glBindVertexArray(umesh.vao_wire_inner);

		GLuint loc_inner_color1 = glGetUniformLocation(wire_displacement_program, "color1");
		glUniform4fv(loc_inner_color1, 1, ro.wire_color2.data());

		glDepthFunc(GL_LESS);
		glEnable(GL_BLEND);
		glEnable(GL_LINE_SMOOTH);

		glLineWidth(ro.wire_line_width * 0.5f);
		glDrawArrays(GL_LINES, 0, umesh.n_wire_inner);

		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_BLEND);
		glDepthFunc(GL_LESS);

		check_gl_error();

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

void GUIApplication::center_on_mouse_cursor()
{
	Vector3 o, dir;
	_get_mouse_ray(&o, &dir);
	
	IntersectionInfo isect;
	if (base.bvh.ray_intersection(o, dir, &isect, FailOnBackwardRayIntersection)) {
		control.offset = (o + isect.t * dir).cast<float>();
	}
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

