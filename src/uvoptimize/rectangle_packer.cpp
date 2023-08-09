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

#include "rectangle_packer.h"
#include "utils.h"

#include <stack>

static bool place_rectangle(const VectorXi& H, Vector2i rect, int grid_size, Vector2i* best_placement, int* height)
{
	*best_placement = Vector2i(-1, -1); // not placed
	*height = H.size() + 1; // just above the grid

	for (int i = 0; i < grid_size; ++i) {
		if (i + rect.x() < grid_size) { // make sure the placement is within the X of the grid
			// find the Y
			int y = 0;
			for (int j = 0; j < rect.x(); ++j) {
				y = std::max(y, H(i + j));
			}
			int curr_height = y + rect.y();
			if (curr_height < *height) {
				*height = curr_height;
				*best_placement = Vector2i(i, y);
			}
		}
	}

	return *best_placement != Vector2i(-1, -1);
}

static bool pack(const std::vector<Vector2>& rectangles, int grid_size, Scalar scale, PackingInfo* pi)
{
	Scalar grid_area = grid_size * grid_size;
	Scalar object_area = std::accumulate(rectangles.begin(), rectangles.end(), Scalar(0), [](Scalar val, const Vector2& r) { return val + r.x() * r.y(); });

	// start packing with a scale such that the object area covers ~ 80% of the packing area
	pi->object_scale = std::sqrt(scale * grid_area / object_area);

	std::vector<Vector2i> scaled_rectangles;
	for (const Vector2& r : rectangles) {
		int w = std::ceil(r.x() * pi->object_scale);
		int h = std::ceil(r.y() * pi->object_scale);
		scaled_rectangles.push_back(Vector2i(w, h));
	}

	// sort rectangles by decreasing area
	std::vector<int> permutation(scaled_rectangles.size());
	std::iota(permutation.begin(), permutation.end(), 0);
	std::sort(permutation.begin(), permutation.end(),
		[&scaled_rectangles](int i, int j) -> bool {
			Vector2i r1 = scaled_rectangles[i];
			Vector2i r2 = scaled_rectangles[j];
			return r1.x() * r1.y() > r2.x() * r2.y();
		}
	);

	// initialize the packing horizon
	VectorXi H = VectorXi::Constant(grid_size, 0);

	// initialize rectangle placements
	pi->placements = std::vector<Vector2i>(scaled_rectangles.size(), Vector2i(-1, -1));

	for (int i = 0; i < scaled_rectangles.size(); ++i) {
		int ri = permutation[i];

		int placement_height;
		bool placed = place_rectangle(H, scaled_rectangles[ri], grid_size, &(pi->placements[ri]), &placement_height);

		if (!placed)
			return false;

		// update horizon
		int x_0 = pi->placements[ri].x();
		for (int x = 0; x < scaled_rectangles[ri].x(); ++x)
			H(x_0 + x) = placement_height;

		int a = 0;
	}

	return true;
}

std::vector<Chart> extract_charts(const MatrixX& UV, const MatrixXi& F, const VFAdjacency& VF)
{
	std::vector<Chart> charts;

	std::set<int> visited;
	for (int vi = 0; vi < UV.rows(); ++vi) {
		if (!visited.count(vi)) {
			// extract chart (connected component)
			Chart chart;

			// get faces by visiting using VFAdjacency
			std::stack<int> s;
			s.push(vi);
			while (!s.empty()) {
				int vj = s.top();
				s.pop();
				if (!visited.count(vj)) { // skip already visited vertices
					for (const VFEntry& vfe : VF[vj]) {
						chart.faces.insert(vfe.first); // add adjacent face
						for (int i = 0; i < 3; ++i) {
							s.push(F(vfe.first, i)); // add all face vertices
						}
					}
					visited.insert(vj);
				}
			}

			// get chart UV box
			for (int fi : chart.faces) {
				for (int i = 0; i < 3; ++i) {
					chart.box.add(UV.row(F(fi, i)));
				}
			}
			charts.push_back(chart);
		}
	}

	return charts;
}

MatrixX pack_charts(const std::vector<Chart>& charts, int grid_size, const MatrixX& UV, const MatrixXi& F, const VFAdjacency& VF)
{
	// extract rectangles;
	std::vector<Vector2> rectangles;
	for (const Chart& c : charts) {
		rectangles.push_back(c.box.diagonal());
	}

	// pack them
	Scalar scale = 1;
	PackingInfo pi;
	bool packed = false;

	while (!(packed = pack(rectangles, grid_size, scale, &pi)))
		scale *= 0.95;

	Assert(packed);

	// now transform the UV coordinates

	// first map vertices to charts
	VectorXi vert_to_chart = VectorXi::Constant(UV.rows(), -1);
	for (int i = 0; i < charts.size(); ++i) {
		for (int fi : charts[i].faces) {
			for (int j = 0; j < 3; ++j)
				vert_to_chart(F(fi, j)) = i;
		}
	}

	// then transform UVs
	MatrixX PUV = UV;
	for (int i = 0; i < UV.rows(); ++i) {
		int chart_id = vert_to_chart[i];
		Vector2 uv = PUV.row(i);
		// translate to origin
		uv -= charts[chart_id].box.cmin;
		// scale to packing area
		uv *= pi.object_scale;
		// offset
		uv += pi.placements[chart_id].cast<Scalar>();
		// scale to parameter space
		uv /= Scalar(grid_size);
		PUV.row(i) = uv;
	}

	return PUV;
}

