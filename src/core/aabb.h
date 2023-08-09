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

#include <iostream>
#include <fstream>

// aabb.h
// 
// This file implements a simple axis aligned bounding box
//

struct Box3 {
	Vector3 cmin;
	Vector3 cmax;

	Box3()
	{
		constexpr Scalar minval = std::numeric_limits<Scalar>::lowest();
		constexpr Scalar maxval = std::numeric_limits<Scalar>::max();
		cmin = Vector3(maxval, maxval, maxval);
		cmax = Vector3(minval, minval, minval);
	}

	void add(const Vector3& p)
	{
		cmin.x() = std::min(cmin.x(), p.x());
		cmin.y() = std::min(cmin.y(), p.y());
		cmin.z() = std::min(cmin.z(), p.z());

		cmax.x() = std::max(cmax.x(), p.x());
		cmax.y() = std::max(cmax.y(), p.y());
		cmax.z() = std::max(cmax.z(), p.z());
	}

	void add(const Box3& b)
	{
		cmin.x() = std::min(cmin.x(), b.cmin.x());
		cmin.y() = std::min(cmin.y(), b.cmin.y());
		cmin.z() = std::min(cmin.z(), b.cmin.z());

		cmax.x() = std::max(cmax.x(), b.cmax.x());
		cmax.y() = std::max(cmax.y(), b.cmax.y());
		cmax.z() = std::max(cmax.z(), b.cmax.z());
	}

	Vector3 diagonal() const
	{
		return cmax - cmin;
	}

	// Returns the dimension of maximum extent
	int max_extent() const
	{
		Vector3 d = diagonal();
		if (d.x() > d.y() && d.x() > d.z())
			return 0;
		else if (d.y() > d.z())
			return 1;
		else
			return 2;
	}

	// Returns the dimension of minimum extent
	int min_extent() const
	{
		Vector3 d = diagonal();
		if (d.x() < d.y() && d.x() < d.z())
			return 0;
		else if (d.y() < d.z())
			return 1;
		else
			return 2;
	}

	Vector3 center() const
	{
		return Scalar(0.5) * (cmin + cmax);
	}

	// returns the projection of p onto the bounding box
	Vector3 project(const Vector3& p) const
	{
		return clamp(p, cmin, cmax);
	}

	// returns the distance of p from the bounding box
	Scalar distance(const Vector3& p) const
	{
		return (project(p) - p).norm();
	}
};

struct Box2 {
	Vector2 cmin;
	Vector2 cmax;

	Box2()
	{
		constexpr Scalar minval = std::numeric_limits<Scalar>::lowest();
		constexpr Scalar maxval = std::numeric_limits<Scalar>::max();
		cmin = Vector2(maxval, maxval);
		cmax = Vector2(minval, minval);
	}

	void add(const Vector2& p)
	{
		cmin.x() = std::min(cmin.x(), p.x());
		cmin.y() = std::min(cmin.y(), p.y());

		cmax.x() = std::max(cmax.x(), p.x());
		cmax.y() = std::max(cmax.y(), p.y());
	}

	void add(const Box2& b)
	{
		cmin.x() = std::min(cmin.x(), b.cmin.x());
		cmin.y() = std::min(cmin.y(), b.cmin.y());

		cmax.x() = std::max(cmax.x(), b.cmax.x());
		cmax.y() = std::max(cmax.y(), b.cmax.y());
	}

	Vector2 diagonal() const
	{
		return cmax - cmin;
	}

	// Returns the dimension of maximum extent
	int max_extent() const
	{
		Vector2 d = diagonal();
		if (d.x() > d.y())
			return 0;
		else
			return 1;
	}

	// Returns the dimension of minimum extent
	int min_extent() const
	{
		Vector2 d = diagonal();
		if (d.x() < d.y())
			return 0;
		else
			return 1;
	}

	Vector2 center() const
	{
		return Scalar(0.5) * (cmin + cmax);
	}

	// returns the projection of p onto the bounding box
	Vector2 project(const Vector2& p) const
	{
		return clamp(p, cmin, cmax);
	}

	// returns the distance of p from the bounding box
	Scalar distance(const Vector2& p) const
	{
		return (project(p) - p).norm();
	}
};

