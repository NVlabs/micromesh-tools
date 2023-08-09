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
#include "gui_controls.h"

#include <iostream>
#include <fstream>
#include <iomanip>


OrbitControl::OrbitControl()
{
	camera.eye = Vector3f(0, 0, 1);
	camera.target = Vector3f(0, 0, 0);
	camera.up = Vector3f(0, 1, 0);

	theta = 0;
	phi = 0;
}

void OrbitControl::update_polar_angles(float dx, float dy)
{
	theta -= dx * 0.005;
	phi -= dy * 0.005;
}

void OrbitControl::update_polar_radius(float step)
{
	if (step > 0)
		r *= 0.95;
	else
		r /= 0.95;
}

void OrbitControl::pan(float dx, float dy)
{
	Vector3f forward = (camera.target - camera.eye).normalized();
	Vector3f right = forward.cross(camera.up).normalized();

	offset -= camera.aspect * r * dx * right;
	offset += r * dy * camera.up;
}

void OrbitControl::update_camera_transform()
{
	Vector3f forward = (camera.target - camera.eye).normalized();
	Vector3f right = forward.cross(camera.up).normalized();

	camera.eye = - forward;

	// rotate by theta around up
	float cu = std::cos(theta);
	float su = std::sin(theta);
	camera.eye = camera.eye * cu + (camera.up.cross(camera.eye)) * su + camera.up * (camera.up.dot(camera.eye)) * (1 - cu);

	// rotate by phi around right
	float cr = std::cos(phi);
	float sr = std::sin(phi);
	camera.eye = camera.eye * cr + (right.cross(camera.eye)) * sr + right * (right.dot(camera.eye)) * (1 - cr);

	theta = 0;
	phi = 0;

	// normalize
	camera.eye.normalize();

	// update up vector
	forward = - camera.eye;
	right = forward.cross(camera.up);
	camera.up = right.cross(forward).normalized();

	// scale
	camera.eye *= r;

	camera.target = offset;
	camera.eye += camera.target;

	camera.view = look_at(camera.eye, camera.target, camera.up);
	camera.projection = perspective(radians(camera.fov), camera.aspect, camera.near_clip * r, camera.far_clip * r);

	//matrix(0, 0) = matrix(1, 1) = matrix(2, 2) = 1 / r;
}

void OrbitControl::write(const std::string& filename) const
{
	std::ofstream ofs(filename);
	if (ofs) {
		ofs << file_header << std::endl;
		ofs << r << std::endl;
		for (int i = 0; i < 3; ++i)
			ofs << offset[i] << std::endl;
		for (int i = 0; i < 3; ++i)
			ofs << camera.eye[i] << std::endl;
		for (int i = 0; i < 3; ++i)
			ofs << camera.target[i] << std::endl;
		for (int i = 0; i < 3; ++i)
			ofs << camera.up[i] << std::endl;
	}
	else {
		std::cerr << "Error writing " << std::quoted(filename) << std::endl;
	}
}

void OrbitControl::read(const std::string& filename)
{
	std::ifstream ifs(filename);
	if (ifs) {
		std::string header;
		ifs >> header;
		if (header == file_header) {
			ifs >> r;
			for (int i = 0; i < 3; ++i)
				ifs >> offset[i];
			for (int i = 0; i < 3; ++i)
				ifs >> camera.eye[i];
			for (int i = 0; i < 3; ++i)
				ifs >> camera.target[i];
			for (int i = 0; i < 3; ++i)
				ifs >> camera.up[i];
		}
		else {
			std::cerr << "Error reading " << std::quoted(filename) << " (invalid header)" << std::endl;
		}
	}
	else {
		std::cerr << "Error reading " << std::quoted(filename) << std::endl;
	}
}

