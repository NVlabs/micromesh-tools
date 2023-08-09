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

enum ColorMap {
	Viridis = 0,
	Plasma = 1,
	Cividis = 2,
	Turbo = 3,
	RdPu = 4,
	PuRd = 5,
	GnBu = 6,
	PuBu = 7,
	YlGnBu = 8,
	GistGray = 9,
	Hot = 10,
	GistHeat = 11,
	Copper = 12
};

Vector4u8 get_color_mapping(Scalar v, Scalar minv, Scalar maxv, ColorMap cmap);

inline Vector4f color_from_u32_abgr(uint32_t c)
{
	return Vector4f(
		(c & 0xff) / float(255),
		((c >> 8) & 0xff) / float(255),
		((c >> 16) & 0xff) / float(255),
		((c >> 24) & 0xff) / float(255));
}

inline uint32_t u32_rgba_from_vec3(const Vector3& v)
{
	Vector3 rgb = clamp(v, Vector3(0, 0, 0), Vector3(1, 1, 1));
	return (int(rgb(0) * 255)) | (int(rgb(1) * 255) << 8) | (int(rgb(2) * 255) << 16) | (0xff << 24);
}

MatrixXf scatter_colors(int n);

