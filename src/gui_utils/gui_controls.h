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
#include "camera.h"

struct OrbitControl {

	const std::string file_header = "umeshtools_orbit_control_data";

	Camera camera;

	float theta = 0.0f;
	float phi = M_PI_2;
	float r = 1.0f;

	OrbitControl();

	Vector3f offset = Vector3f(0, 0, 0);

	Matrix4f matrix = Matrix4f::Identity();

	void update_polar_angles(float dx, float dy);
	void update_polar_radius(float step);
	void pan(float dx, float dy);
	void update_camera_transform();

	void write(const std::string& filename) const;
	void read(const std::string& filename);
};



