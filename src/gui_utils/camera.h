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
#include "utils.h"

struct Camera {
	Vector3f eye;
	Vector3f target;
	Vector3f up;

	float fov = 60.0f;
	float aspect = 1.0f;
	float near_clip = 0.001f;
	float far_clip = 1000.0f;
	
	Matrix4f view;
	Matrix4f projection;
};

inline Matrix4f look_at(const Vector3f& eye, const Vector3f& target, const Vector3f& up)
{
	Vector3f z = (eye - target).normalized();
	Vector3f x = up.cross(z).normalized();
	Vector3f y = z.cross(x).normalized();
	Matrix4f M = Matrix4f::Identity();
	M(0, 0) = x(0);
	M(0, 1) = x(1);
	M(0, 2) = x(2);
	M(1, 0) = y(0);
	M(1, 1) = y(1);
	M(1, 2) = y(2);
	M(2, 0) = z(0);
	M(2, 1) = z(1);
	M(2, 2) = z(2);
	M(0, 3) = -x.dot(eye);
	M(1, 3) = -y.dot(eye);
	M(2, 3) = -z.dot(eye);

	return M;
}

inline Matrix4f perspective(float fov, float aspect, float near_plane, float far_plane)
{
	float tan_half_y = std::tan(fov / 2.0f);
	Matrix4f M = Matrix4f::Zero();
	M(0, 0) = 1.0f / (tan_half_y * aspect);
	M(1, 1) = 1.0f / tan_half_y;
	M(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
	M(2, 3) = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
	M(3, 2) = -1.0f;
	return M;
}

// This function converts a normalized screen
// coordinate point p in [-1, 1] x [1, 1] into the corresponding 3D point
// on the image plane in world space
inline Vector3f screen_to_image_plane(const Camera& camera, Vector2f p)
{
	Assert(p.x() >= -1 && p.x() <= 1);
	Assert(p.y() >= -1 && p.y() <= 1);

	Vector3f cx = camera.view.block(0, 0, 1, 3).transpose();
	Vector3f cy = camera.view.block(1, 0, 1, 3).transpose();
	Vector3f cz = camera.view.block(2, 0, 1, 3).transpose();

	float tan_half_y = std::tan(radians(camera.fov / 2.0f));
	Vector3f fo = camera.eye + (-cz * camera.near_clip);
	Vector3f fy = cy * tan_half_y * camera.near_clip;
	Vector3f fx = cx * tan_half_y * camera.near_clip * camera.aspect;

	return fo + p.x() * fx + p.y() * fy;
}

