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
#include "adjacency.h"
#include "local_operations.h"
#include "quadric.h"
#include "utils.h"
#include "tangent.h"

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <memory>

//
// decimation.h
// 
// This class implement the mesh decimation algorithm. The implementation works by iteratively
// collapsing mesh edges following a given ordering. The decimation logic is implemented in
// the Decimation struct, and relies on callback objects to compute the vertex placement following
// a collapse operation, and the related cost.
// 
// Callback objects follow the DecimationCallbackObject interface, which defines two methods

//   compute_position_and_cost() computes the optimal vertex placement and the cost of the collapse
//       operation (operations are sorted from lowest to highest cost)
// 
//   update_on_collapse() is called after every collapse operation and is used to update auxiliary
//       data that the decimation callback object may store (e.g. additional vertex attributes)
// 
// Collapse operations are encoded as pair of vertices. The decimation process assumes the topology
// is initially valid, and updated throughout the execution.
//
// Currently, there are three callback objects implemented

//   - DecimationCallbackMidpoint implements a simple midpoint edge collapse, cost is the edge length 
// 
//   - DecimationCallbackQuadric implements the Quadric edge collapse algorith of Garland&Heckbert (QSLIM),
//         vertex position is the quadric minimizer, cost is the minimizer quadric value (geometric error)
// 
//   - DecimationCallbackSmoothedQuadric implements QSLIM with an additional vertex smoothing blended into
//         the final optimal vertex placement. This is needed to get out of locked vertex cluster when
//         aspect ratio constraints are included in the decimation process.
//

// Edge collapse attributes (position, geometric error and cost) returned
// by DecimationCallbackSmoothedQuadric::evaluate_collapse()
struct CollapseData {
	// the optimal vertex position
	Vector3 position = Vector3(Infinity, Infinity, Infinity);

	// the geometric error of the vertex position
	Scalar geometric_error = Infinity;

	// the cost function value
	Scalar cost = Infinity;

	// flags of zero-valued denominator terms for the cost function
	uint8_t _infinite_cost_flags = 0;

	static constexpr uint8_t _AspectRatio_Flag = 1 << 1;
	static constexpr uint8_t _Visibility_Flag  = 1 << 2;
	static constexpr uint8_t _Normals_Flag     = 1 << 3;
};

struct DecimationParameters {
	// Penalty function parameters
	Scalar ar_scaling = 0.5;
	Scalar visibility_scaling = 0.5;
	Scalar normals_scaling = 0.1;
	Scalar border_error_scaling = 1;
	
	// Feasibility test switches and thresholds
	bool preserve_topology = true;
	bool bound_normals_correlation = false;
	bool bound_aspect_ratio = false;
	bool bound_geometric_error = false;
	
	Scalar min_normal_correlation = 0.3;
	Scalar min_border_normal_correlation = 0.9;
	Scalar min_aspect_ratio = 0.2;
	Scalar aspect_ratio_adaptive_tolerance = 0.1;
	Scalar max_relative_error = std::numeric_limits<Scalar>::max();

	// Smoothed edge collapse controls
	bool use_vertex_smoothing = false;
	Scalar smoothing_coefficient = 0;

	// If true, after every collapse tries to flip edges to improve aspect ratio
	bool local_flips_after_collapse = false;

	// If true, periodically updates the vertex quadrics by projecting them from
	// the original input mesh
	bool reproject_quadrics = false;
};

// DecimationCallbackSmoothedQuadric
// Helper class to manage quadrics and evaluate collapses during simplification
struct DecimationCallbackSmoothedQuadric {
	const MatrixX& V;
	const MatrixXi& F;
	const VFAdjacency& VF;

	VectorXu8 VB;

	std::vector<Quadric> Q;
	std::vector<Scalar> QW;

	MatrixX VD; // Optimal visibility directions
	VectorX VIS; // Vertex visibility

	DecimationCallbackSmoothedQuadric(const MatrixX& VV, const MatrixXi& FF, const VFAdjacency& VFF);

	bool has_quadrics() const;

	bool init_quadrics(Scalar border_error_scaling);

	Quadric get_edge_quadric(Edge e) const;

	void _compute_position_and_error(Edge e, Vector3& p, Scalar& error) const;

	CollapseData evaluate_collapse(Edge e, const MatrixX& FN, const DecimationParameters& decimation_parameters) const;

	void update_on_collapse(Edge e);
	void update_on_split(Edge e, int new_vertex, const Vector3& new_vertex_pos);
	void update_on_vertex_split(int old_vertex, int new_vertex);
};

struct Decimation {

	// Status bits for the feasiblity of edge collapses
	static constexpr int OpStatus_Feasible = 0;
	static constexpr int OpStatus_Unknown = 1 << 0;
	static constexpr int OpStatus_FailTopology = 1 << 1;
	static constexpr int OpStatus_FailNormals = 1 << 2;
	static constexpr int OpStatus_FailAspectRatio = 1 << 3;
	static constexpr int OpStatus_FailGeometricError = 1 << 4;
	static constexpr int OpStatus_FailVertexRingNormals = 1 << 5;

	typedef int Timestamp;
	typedef int OpStatus;

	// An EdgeEntry object encodes a collapse operation
	struct EdgeEntry {
		Edge edge;

		// Collapse data
		CollapseData collapse_data;
		
		// Timestamp to detect obsolete collapses in the queue of operations
		Timestamp time;

		EdgeEntry()
			: edge(INVALID_INDEX, INVALID_INDEX), collapse_data(), time(-1)
		{
		}

		EdgeEntry(const Edge& e, const CollapseData& cd, Timestamp t)
			: edge(e), collapse_data(cd), time(t)
		{
		}

		bool operator<(const EdgeEntry& other) const
		{
			return collapse_data.cost < other.collapse_data.cost;
		}

		bool operator>(const EdgeEntry& other) const
		{
			return collapse_data.cost > other.collapse_data.cost;
		}
	};

	DecimationParameters _parameters;

	// Object-scale max error threshold
	Scalar _max_error = std::numeric_limits<Scalar>::max();

	// Decimation mesh buffers
	MatrixX& V;
	MatrixXi& F;
	VFAdjacency& VF;

	MatrixX& D;
	VectorX& DW;

	VectorX& FR;
	MatrixX& FN;

	// Input mesh buffers
	const MatrixX& hi_V;
	const MatrixXi& hi_F;
	VectorXu8 hi_VB;

	// Handle to the quadric callback object
	std::shared_ptr<DecimationCallbackSmoothedQuadric> dcb;

	// TODO unused, to remove
	std::vector<std::set<int>> decimated_to_input_verts;

	// Heap of collapse operations
	std::vector<EdgeEntry> _heap;
	// Timestamp map, the EdgeEntry ee operation is valid if tmap[ee] == ee.timestamp
	std::unordered_map<Edge, Timestamp> tmap;
	// Map of edge statuses, used for debugging purposes
	std::map<Edge, OpStatus> opstatus;

	int fn_curr;
	int fn_start;

	Timer timer;

	// log and debug data
	double t_init = 0;
	double t_pop = 0;
	double t_collapse = 0;
	double t_collapse_wasted = 0;
	double t_update = 0;

	int collapse_not_ok = 0;
	int collapse_ok = 0;
	int update = 0;
	int nflips = 0;

	int nreject_topology = 0;
	int nreject_geometry = 0;
	int nreject_aspect_ratio = 0;
	int nreject_error = 0;
	int nreject_cone = 0;

	struct {
		int n_obsolete = 0;
		int n_eval = 0;
	} _debug;

	Decimation(const MatrixX& hi_V, const MatrixXi& hi_F, MatrixX& VV, MatrixXi& FF, VFAdjacency& VFF,
		MatrixX& DD, VectorX& DWW, VectorX& FR, MatrixX& FN);

	// loads the provided mesh buffers into the decimation mesh
	void _import_decimation_mesh(const MatrixX& iV, const MatrixXi& iF, const VFAdjacency& iVF,
		const std::vector<Quadric>& iQ, const std::vector<Scalar>& iQW,
		const MatrixX& D, const VectorX& DW, const VectorX& iFR, const MatrixX& iFN);

	// Clears the algorithm state and initializes the queue of operations
	void init(const DecimationParameters& parameters);
	
	//bool collapse_allowed(const Flap& flap, const Vector3& vpos);

	// this function returns true if a next valid collapse is in the
	// queue ( *** WHITOUT *** popping the valid entry) and the corresponding
	// flap, otherwise it returns false (which implies that q.empty() == true)
	bool next_valid(Flap& flap);

	// Executes the iterative edge-collapse decimation, using the provided fn_target
	// value as lower bound on the number of faces
	int execute(int fn_target);

	// Returns true if the edges to update after each collapse are up to distance two
	// according to the current decimation parameters
	bool _needs_large_update() const;

	// Recomputes quadrics by projecting them from the input mesh
	void _refresh_quadrics();

	// moves each vertex in its quadric minimzer (useful after reprojection, or to
	// relocate vertices at the end of the simplification when smoothed edge collapses are used
	void _move_vertices_in_quadrics_min();
	// rebuild the queue of operations
	void _rebuild_heap();
	// performs heap compaction to get rid of obsolete entries
	void _compact_heap();

	// Inserts collapse operations in the queue
	void _push_operation(const Edge& e);
	void _push_operations(const std::set<Edge>& edges);

	// Pops the next operation from the queue
	void _pop_operation();

	// Tests the feasiblity of the operation, returning the corresponding status
	int _test_operation(const EdgeEntry& ee, const Flap& flap) const;

	std::pair<EdgeEntry, int> _get_collapse_data(int fi, int e) const;
	void log_collapse_data(const EdgeEntry& ee, int status)const ;

	// Returns true if the queue is empty
	bool queue_empty();

	// Compacts the decimation mesh buffers after simplification.
	// TODO FIXME This is here because face compaction invalidates the list of operations, but it's not a particularly good choice
	void compact_faces();

	// Update vertex and face data after an edge split
	void manage_split(const SplitInfo& si);

	// Update vertex and face data after a vertex split
	void manage_vertex_split(const VertexSplitInfo& vsi);
};

// Mesh decimation without a priority queue, Q and QW are return parameters
void decimate_mesh_fast(MatrixX& V, MatrixXi& F, VFAdjacency& VF, int fn_target,
	std::vector<Quadric>& Q, std::vector<Scalar>& QW,
	MatrixX& D, VectorX& DW, VectorX& FR, MatrixX& FN, const DecimationParameters& decimation_parameters);

void decimate_mesh_parallel(MatrixX& V, MatrixXi& F, VFAdjacency& VF, int fn_target,
	std::vector<Quadric>& Q, std::vector<Scalar>& QW,
	MatrixX& D, VectorX& DW, VectorX& FR, MatrixX& FN, const DecimationParameters& decimation_parameters, int nw = 4);

struct LocalEdgeFlip {

	enum FlipStatus : uint32_t {
		Feasible = 0,
		Unknown,
		Fail_Topology,
		Fail_AspectRatio,
		Fail_DirectionFieldAlignment,
		Fail_Split,
		Fail_Planarity
	};

	typedef int Timestamp;

	struct Flip {
		Edge edge;
		Scalar planarity;
		Timestamp time;

		Flip(const Edge& e, Scalar p, Timestamp t)
			: edge(e), planarity(p), time(t)
		{
		}

		bool operator<(const Flip& other) const
		{
			return planarity < other.planarity;
		}

		bool operator>(const Flip& other) const
		{
			return planarity > other.planarity;
		}
	};

	static constexpr int EXPIRED = -1;

	MatrixX& V;
	MatrixXi& F;
	VFAdjacency& VF;

	const MatrixX& D;
	const VectorX& DW;

	std::shared_ptr<const DecimationCallbackSmoothedQuadric> dcb;

	Scalar planarity_threshold = 1e-6;

	// if false, after an edge flip the 4 other edges of the quad are added to the list of candidate operations
	bool forbid_new_flips = false;

	std::priority_queue<Flip, std::vector<Flip>, std::greater<Flip>> q;
	std::map<Edge, Timestamp> tmap;

	std::set<Edge> obsolete;
	std::set<Edge> fresh;

	LocalEdgeFlip(MatrixX& VV, MatrixXi& FF, VFAdjacency& VFF, std::shared_ptr<const DecimationCallbackSmoothedQuadric> dcbb,
		const MatrixX& DD, const VectorX& DWW)
		: V(VV), F(FF), VF(VFF), dcb(dcbb), D(DD), DW(DWW)
	{
	}

	void init();
	void init(const std::set<Edge>& edges);

	int execute();

	bool next_valid(Flap& flap);

	FlipStatus _test_operation(const Flip& f, const Flap& flap);
};

