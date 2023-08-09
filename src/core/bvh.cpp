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

#include "bvh.h"
#include "intersection.h"
#include "tangent.h"
#include "utils.h"

#include <stack>

bool BVHTree::ray_intersection(const Vector3& o, const Vector3& d, IntersectionInfo* ii, const std::function<bool(IntersectionInfo)>& test) const
{
	int num_test_interior = 0;
	int num_test_primitives = 0;

	*ii = {};
	ii->fi = -1;
	ii->b = Vector3::Zero();
	ii->t = std::numeric_limits<Scalar>::max();
	ii->o = o;
	ii->d = d;

	if (nodes.empty())
		return false;

	std::stack<int> s;
	s.push(0);

	while (!s.empty()) {
		int current_node = s.top();
		s.pop();

		Scalar node_t_min, node_t_max;
		if (ray_box_intersection(o, d, nodes[current_node].box, &node_t_min, &node_t_max)) {
			Assert(node_t_min <= node_t_max);
			bool ray_origin_inside_volume = node_t_min * node_t_max <= 0;
			if (ii->fi == -1 || ray_origin_inside_volume || std::min(std::abs(node_t_min), std::abs(node_t_max)) < std::abs(ii->t)) {
				// if no hit found yet or found hit closer than current best
				if (nodes[current_node].left == -1) {
					Assert(nodes[current_node].left == nodes[current_node].right);
					// leaf nodes, test primitives
					for (int fi : nodes[current_node].primitives) {
						IntersectionInfo f_isect;
						f_isect.fi = fi;
						f_isect.o = o;
						f_isect.d = d;
						bool hit = ray_triangle_intersection(o, d, V->row((*F)(fi, 0)), V->row((*F)(fi, 1)), V->row((*F)(fi, 2)), &f_isect.t, &f_isect.b[0], &f_isect.b[1], &f_isect.b[2]);
						if (hit && std::abs(f_isect.t) < std::abs(ii->t) && test(f_isect)) {
							*ii = f_isect;
						}
						num_test_primitives++;
					}
				}
				else {
					// interior node, push children onto stack
					s.push(nodes[current_node].right);
					s.push(nodes[current_node].left);
				}
			}
		}
		num_test_interior++;
	}

	return ii->fi != -1;
}

bool BVHTree::ray_intersection(const Vector3& o, const Vector3& d, IntersectionInfo* ii) const
{
	return ray_intersection(o, d, ii, [](const IntersectionInfo& ii) { return true; });
}

bool BVHTree::nearest_point(const Vector3& p, NearestInfo* ni, const std::function<bool(int)>& test) const
{
	int num_test_interior = 0;
	int num_test_primitives = 0;

	constexpr Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
	
	*ni = {};
	ni->fi = -1;
	ni->p = Vector3(nan, nan, nan);
	ni->d = std::numeric_limits<Scalar>::max();

	if (nodes.empty())
		return false;

	std::stack<int> s;
	s.push(0);

	while (!s.empty()) {
		int current_node = s.top();
		s.pop();

		Scalar box_distance = nodes[current_node].box.distance(p);
		if (box_distance < ni->d) {
			//std::cout << "At box distance " << box_distance << std::endl;
			if (nodes[current_node].left == -1) {
				Assert(nodes[current_node].left == nodes[current_node].right);
				for (int fi : nodes[current_node].primitives) {
					if (test(fi)) {
						Vector3 f_nearest_point = nearest_triangle_point(p,
							V->row((*F)(fi, 0)), V->row((*F)(fi, 1)), V->row((*F)(fi, 2)));

						Scalar f_distance = (f_nearest_point - p).norm();
						if (f_distance < ni->d) {
							ni->fi = fi;
							ni->p = f_nearest_point;
							ni->d = f_distance;
						}
						num_test_primitives++;
					}
				}
			}
			else {
				int left = nodes[current_node].left;
				int right = nodes[current_node].right;
				if (nodes[left].box.distance(p) < nodes[right].box.distance(p)) {
					s.push(right);
					s.push(left);
				}
				else {
					s.push(left);
					s.push(right);
				}
			}
		}
		num_test_interior++;
	}

	return ni->fi != -1;
}

bool BVHTree::nearest_point(const Vector3& p, NearestInfo* ni) const
{
	return nearest_point(p, ni, [](int fi) { return true; });
}

void BVHTree::build_tree(const MatrixX* Vptr, const MatrixXi* Fptr, const MatrixX* VNptr, int max_primitive_count)
{
	Timer t;
	std::cerr << "Building BVH..." << std::endl;

	h = 0;

	V = Vptr;
	F = Fptr;
	VN = VNptr;

	if (F->rows() > 0) {
		std::vector<int> face_indices;
		std::vector<Vector3> centroids;
		for (int fi = 0; fi < F->rows(); ++fi) {
			if ((*F)(fi, 0) != INVALID_INDEX) {
				face_indices.push_back(fi);
				centroids.push_back((V->row((*F)(fi, 0)) + V->row((*F)(fi, 1)) + V->row((*F)(fi, 2))) / Scalar(3));
			}
			else {
				centroids.push_back(Vector3::Zero());
			}
		}

		nodes.push_back(BVHNode());
		_build_node(face_indices.begin(), face_indices.end(), 0, 0, max_primitive_count, centroids);

		_update_bounding_volume(0);
	}

	std::size_t sz = nodes.size() * sizeof(BVHNode);
	for (const auto& node: nodes) {
		sz += sizeof(int) * node.primitives.size();
	}

	std::cerr << "BVH build took " << t.time_elapsed() << " seconds" << std::endl;
	std::cerr << "Tree depth is " << h << std::endl;
	std::cerr << "Tree size is " << sz / (1024 * 1024) << " MBs" << std::endl;
}

void BVHTree::_build_node(_VecIntIterator fbegin, _VecIntIterator fend, int node_index, int depth, int max_primitive_count, const std::vector<Vector3>& centroids)
{
	// Fail if the depth is too large
	// This can happen if there are duplicate faces in the mesh
	Assert(depth < 200 && "BVH build: tree size too large, possible infinite recursion");

	h = std::max(h, depth);

	int deg = F->cols();
	Assert(deg == 3);

	for (auto itfi = fbegin; itfi != fend; ++itfi) {
		nodes[node_index].box.add(centroids[*itfi]);
	}

	int fn = std::distance(fbegin, fend);

	if (fn <= max_primitive_count) {
		// store indices
		for (auto itfi = fbegin; itfi != fend; ++itfi)
			nodes[node_index].primitives.push_back(*itfi);
	}
	else {
		// split node and recurse
		int children_index = nodes.size();
		nodes.push_back(BVHNode());
		nodes.push_back(BVHNode());

		nodes[node_index].left = children_index;
		nodes[node_index].right = children_index + 1;

		int splitting_axis = nodes[node_index].box.max_extent();
		Scalar midpoint = nodes[node_index].box.center()[splitting_axis];

		_VecIntIterator split = std::partition(fbegin, fend, [&](int fi) {
			return centroids[fi][splitting_axis] < midpoint;
		});

		_build_node(fbegin, split, nodes[node_index].left, depth + 1, max_primitive_count, centroids);
		_build_node(split, fend, nodes[node_index].right, depth + 1, max_primitive_count, centroids);
	}
}

void BVHTree::_update_bounding_volume(int node_index)
{
	int deg = F->cols();

	if (nodes[node_index].left == -1 || nodes[node_index].right == -1) {
		// if leaf node, update the bounding box with the extents of all the primitives
		Assert(nodes[node_index].left == nodes[node_index].right);
		for (int fi : nodes[node_index].primitives) {
			for (int j = 0; j < deg; ++j) {
				nodes[node_index].box.add(V->row((*F)(fi, j)));
			}
		}
	}
	else {
		// update bounding volumes of children and pull up
		int left = nodes[node_index].left;
		int right = nodes[node_index].right;
		_update_bounding_volume(left);
		_update_bounding_volume(right);
		nodes[node_index].box.add(nodes[left].box);
		nodes[node_index].box.add(nodes[right].box);
	}
}


