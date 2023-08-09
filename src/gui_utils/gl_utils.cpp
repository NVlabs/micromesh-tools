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

#include "gl_utils.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

static void glfw_error_callback(int err, const char* message)
{
	std::cerr << "GLFW Error " << err << ": " << message << std::endl;
}

void gl_init()
{
}

void gl_terminate()
{
	glfwTerminate();
}

void check_gl_error()
{
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		std::cerr << "OpenGL error " << error << " ";
		if (error == GL_INVALID_VALUE)
			std::cerr << "GL_INVALID_VALUE";
		if (error == GL_INVALID_OPERATION)
			std::cerr << "GL_INVALID_OPERATION";
		std::cerr << std::endl;
	}
}

uint32_t compile_shader_program(const char** vs_text, const char** fs_text)
{
	GLint status;
	char infolog[4096] = { 0 };

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, vs_text, NULL);
	glCompileShader(vs);
	glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		glGetShaderInfoLog(vs, 4096, NULL, infolog);
		std::cerr << infolog << std::endl;
		std::cerr << "Vertex shader compilation failed" << std::endl;
		memset(infolog, 0, 4096);	
	}

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, fs_text, NULL);
	glCompileShader(fs);
	glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		glGetShaderInfoLog(fs, 4096, NULL, infolog);
		std::cerr << infolog << std::endl;
		std::cerr << "Fragment shader compilation failed" << std::endl;
		memset(infolog, 0, 4096);	
	}
	
	GLuint program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramInfoLog(program, 4096, NULL, infolog);
	if (*infolog) {
		std::cerr << infolog << std::endl;
	}
	glValidateProgram(program);
	glGetProgramInfoLog(program, 4096, NULL, infolog);
	if (*infolog) {
		std::cerr << infolog << std::endl;
	}
	glGetProgramInfoLog(program, 4096, NULL, infolog);
	if (*infolog) {
		std::cerr << infolog << std::endl;
	}
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		std::cerr << "Shader program link failed" << std::endl;
	}
	glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
	if (status == GL_FALSE) {
		std::cerr << "Shader program validation failed" << std::endl;
	}
	
	glDeleteShader(vs);
	glDeleteShader(fs);

	check_gl_error();

	return program;
}




