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
#include "aabb.h"

#include <utility>

// Solves o + td = a (v1 - v0) + b (v2 - v0) in t, a, b
// Returns the parametric ray length for the plane intersection, and the
// barycentric coordinates of v0 and v1 in res[0], res[1] and res[2]
// respectively
// if res[1] and res[2] do not satisfy the constraints for a convex combination
// the intersection falls outside the triangle
inline Vector3 ray_triangle_intersection(const Vector3& o, const Vector3& d,
	const Vector3& v0, const Vector3& v1, const Vector3& v2)
{
	Matrix3 A;
	A.col(0) = d;
	A.col(1) = v2 - v0;
	A.col(2) = v2 - v1;
	Vector3 b = v2 - o;
	return A.inverse() * b;
}

inline bool ray_triangle_intersection(
	const Vector3& o, const Vector3& d,
	const Vector3& v0, const Vector3& v1, const Vector3& v2,
	Scalar *t_hit, Scalar *bary_0, Scalar *bary_1, Scalar *bary_2)
{
	constexpr Scalar tol = 6 * std::numeric_limits<Scalar>::epsilon();

	Vector3 params = ray_triangle_intersection(o, d, v0, v1, v2);
	Scalar t = params[0];
	Scalar b0 = params[1];
	Scalar b1 = params[2];
	Scalar b2 = 1 - b0 - b1;
	if ((b0 + tol) >= 0 && (b1 + tol) >= 0 && (b2 + tol) >= 0 && (b0 + b1 + b2) <= (1 + tol)) {
		*t_hit = t;
		*bary_0 = b0;
		*bary_1 = b1;
		*bary_2 = b2;
		return true;
	}
	else {
		return false;
	}
}

// This function uses slab intersections
inline bool ray_box_intersection(const Vector3& o, const Vector3& d, const Box3& box, Scalar *t_min, Scalar *t_max)
{
	Scalar t0 = std::numeric_limits<Scalar>::lowest();
	Scalar t1 = std::numeric_limits<Scalar>::max();

	for (int i = 0; i < 3; ++i) {
		Scalar inv_di = Scalar(1) / d[i];
		Scalar t0i = (box.cmin[i] - o[i]) * inv_di;
		Scalar t1i = (box.cmax[i] - o[i]) * inv_di;
		t1i *= 1 + 1e-12;
		if (t0i > t1i)
			std::swap(t0i, t1i);
		t0 = std::max(t0, t0i);
		t1 = std::min(t1, t1i);
		if (t0 > t1)
			return false;
	}

	*t_min = t0;
	*t_max = t1;

	return true;
}

// ray-plane intersection, plane is encoded as (point, normal)
inline Scalar ray_plane_intersection(const Vector3& o, const Vector3& d, const Vector3& p, const Vector3& n)
{
	return (p - o).dot(n) / d.dot(n);
}

