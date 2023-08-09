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

#include <vector>
#include <functional>

// bvh.h
// 
// Simple implementation of a BVH using axis aligned bounding boxes
// as bounding volumes.
// Binary tree, splits are applied at the midpoint of the largest
// dimension of the bounding volume of a node.
// 
// The interface supports ray intersection and nearest point queries,
// and accepts a function object to filter invalid intersections based
// on user-defined predicates.
//

// Intersection point data
struct IntersectionInfo {
	// Index of the intersecting face 
	int fi;
	// Barycentric coordinates of the intersection point
	Vector3 b;
	// Parametric ray length
	Scalar t;

	// Ray origin
	Vector3 o;
	// Ray direction
	Vector3 d;
};

// Nearest point data
struct NearestInfo {
	// Index of the face where the nearest point lies
	int fi;
	// Nearest point coordinates
	Vector3 p;
	// Distance
	Scalar d;
};

// BVH node
struct BVHNode {
	// Bounding volume of the node
	Box3 box;

	// Index of the left child of the node
	int left = -1;
	// Index of the right child of the node
	int right = -1;

	// List of face indices stored at a node (only for leaf nodes)
	std::vector<int> primitives;
};

// Predefined predicate, fails if the intersection is backward relative to the ray origin
inline bool FailOnBackwardRayIntersection(const IntersectionInfo& ii) { return ii.t > 0; }

// Binary tree of AABBs
struct BVHTree {
	typedef std::function<bool(IntersectionInfo)> IntersectionFilter;
	typedef std::function<bool(int)> NearestFilter;

	typedef std::vector<int>::iterator _VecIntIterator;

	// Tree nodes
	std::vector<BVHNode> nodes;

	// Vertex positions of the bounded mesh
	const MatrixX* V = nullptr;

	// Face indices of the bounded mesh
	const MatrixXi* F = nullptr;

	// Vertex normals of the bounded mesh
	const MatrixX* VN = nullptr;

	// Height of the tree
	int h = 0;

	// Ray intersection query
	// Returns true if an intersection is found
	// Arguments
	//   o IN The ray origin
	//   d IN The ray direction
	//   ii OUT Pointer to the IntersectionInfo object filled with the intersection data (if found)
	//   test IN Test used to filter invalid intersections (default: always pass)
	bool ray_intersection(const Vector3& o, const Vector3& d, IntersectionInfo* ii, const std::function<bool(IntersectionInfo)>& test) const;
	bool ray_intersection(const Vector3& o, const Vector3& d, IntersectionInfo* ii) const;

	// Nearest point query (projection)
	// Returns true if a nearest point is found (not always the case if a filter test is used)
	// Arguments
	//   p IN The query point
	//   ni OUT Pointer to the NearestInfo object filled with the nearest query data (if found)
	//   test IN Test used to filter invalid projections (default: always pass)
	bool nearest_point(const Vector3& p, NearestInfo* ni, const std::function<bool(int)>& test) const;
	bool nearest_point(const Vector3& p, NearestInfo* ni) const;

	// Builds a tree from the given mesh buffers
	//   Vptr IN Vertex positions
	//   Fptr IN Face indices
	//   VNptr IN Vertex normals
	//   max_primitive_count IN Maximum number of faces per node, otherwise node is split
	void build_tree(const MatrixX* Vptr, const MatrixXi* Fptr, const MatrixX* VNptr, int max_primitive_count = 4);

	// Builds a BVH node from a list of face indices
	// Arguments
	//   fbegin IN Sequence of face indices to insert in the node (begin)
	//   fend IN Sequence of face indices to insert in the node (end)
	//   node_index IN Index of the node to build
	//   depth IN Depth of the current node
	//   max_primitive_count IN Maximum number of faces per node, otherwise node is split
	//   centroids IN Full vector of face centroids, used to control the splitting
	void _build_node(_VecIntIterator fbegin, _VecIntIterator fend, int node_index, int depth, int max_primitive_count, const std::vector<Vector3>& centroids);

	// Recursive update of the bounding volumes, propagates from leaf nodes
	void _update_bounding_volume(int node_index);
};

