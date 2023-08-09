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

#include "decimation.h"
#include "local_operations.h"
#include "direction_field.h"
#include "mesh_utils.h"
#include "aabb.h"

#include <thread>
#include <atomic>
#include <stack>
#include <random>

using EdgeEntry = Decimation::EdgeEntry;

struct DecimationData {
	MatrixX& V;
	MatrixXi& F;
	VFAdjacency& VF;

	std::shared_ptr<DecimationCallbackSmoothedQuadric> dcb;

	VectorX& FR;
	MatrixX& FN;

	std::vector<int>& faces;
	std::vector<std::atomic_flag>& flags;

	std::atomic_int fn_curr;

	DecimationData(MatrixX& VV, MatrixXi& FF, VFAdjacency& VFF, std::shared_ptr<DecimationCallbackSmoothedQuadric> dcb_,
		VectorX& FFR, MatrixX& FFN,
		std::vector<int>& faces_, std::vector<std::atomic_flag>& flags_)
		: V(VV), F(FF), VF(VFF), dcb(dcb_), FR(FFR), FN(FFN), faces(faces_), flags(flags_), fn_curr(FF.rows())
	{
	}
};

struct DecimationWorker {

	using EdgeEntry = Decimation::EdgeEntry;

	int wid;

	std::mt19937 generator;

	Scalar current_cost_threshold;

	const DecimationParameters& dparams;

	// Algorithm parameters
	const int step_increase_count = 20; // How many samples above the current error threshold before increasing it
	const int interrupt_count = 500; // How many failed iterations before interrupting execution (~ target error threshold reached)
	const uint8_t num_collapse_samples = 3; // How many collapses are sampled at each iteration
	Scalar increase_factor = 1.3;

	std::vector<uint8_t> _marks;
	static constexpr uint8_t MarkFree = 0xff;

	struct {
		int n_iter = 0;
		int n_collapse = 0;
		int n_fail_expand = 0;
		int n_fail_error = 0;

		Scalar t_lock = 0;
		Scalar t_total = 0;
	} stats;

	DecimationWorker(int id, Scalar initial_cost_guess, const DecimationParameters& decimation_parameters)
		: wid(id), generator(id), current_cost_threshold(initial_cost_guess), dparams(decimation_parameters)
	{
	}

	//void operator()(DecimationData& data, int fn_target)
	//{
	//	run(data, fn_target);
	//}

	// deicimation loop
	void run(DecimationData& data, int fn_target, Scalar max_error)
	{
		_marks = std::vector<uint8_t>(data.flags.size(), MarkFree);

		const int MAX_LOCK_FAILURES = 20;

		int n_consecutive_lock_failures = 0;

		Timer t;

		int n_failures_after_cost_increase = 0;
		int n_consecutive_failures = 0;

		while (data.fn_curr > fn_target && n_consecutive_lock_failures < MAX_LOCK_FAILURES) {
			if (n_consecutive_failures > interrupt_count) { // stop if too many unsuccessful iterations
				//std::cout << "Too many iterations without collapses, stopping" << std::endl;
				break;
			}

			//Timer tt;
			stats.n_iter++;

			// gradually increase error tolerance when decimating
			if (n_failures_after_cost_increase > step_increase_count /* && current_error_threshold < max_error*/) {
				n_failures_after_cost_increase = 0;
				current_cost_threshold = current_cost_threshold * increase_factor;
			}
		
			std::vector<std::vector<int>> locked_face_sets;
			std::vector<EdgeEntry> collapses = _sample_collapses(data, locked_face_sets);

			// if we don't get any edge, it's because of a collision when trying to lock the local regions
			if (collapses.size() == 0) {
				n_consecutive_lock_failures++;
				stats.n_fail_expand++;
			}
			else {
				n_consecutive_lock_failures = 0;
			}

			//stats.t_lock += tt.time_since_last_check(); // too slow? test whether it's slow to use atomics (i.e. bools with nw == 1 are faster) or
			// it's the way we are finding the faces that is slow (using vfadj leads to a lot of redundant visits)

			// sort indices by increasing collapse cost
			std::vector<int> sorted_collapses = vector_of_indices(int(collapses.size()));
			std::sort(sorted_collapses.begin(), sorted_collapses.end(), [&](int i, int j) { return collapses[i] < collapses[j]; });

			bool collapsed = false; // becomes true if the iteration managed to collapse an edge
			for (int i : sorted_collapses) {
				const EdgeEntry& ee = collapses[i];
				if (std::isfinite(ee.collapse_data.cost) && ee.collapse_data.cost <= current_cost_threshold) {
					// attempt collapse
					Flap flap = compute_flap(ee.edge, data.F, data.VF);
					bool feasible = (!dparams.bound_geometric_error) || (ee.collapse_data.geometric_error <= max_error);
					if (feasible)
						feasible = collapse_preserves_topology(flap, data.F, data.VF, data.dcb->VB);
					if (feasible && dparams.bound_aspect_ratio)
						feasible = collapse_preserves_aspect_ratio(flap, data.V, data.F, data.VF, ee.collapse_data.position, dparams.min_aspect_ratio, data.FR, dparams.aspect_ratio_adaptive_tolerance);

					if (feasible) {
						// collapse edge
						CollapseInfo collapse = collapse_edge(flap, data.V, data.F, data.VF, data.dcb->VB, ee.collapse_data.position);
						Assert(collapse.ok);
						data.dcb->update_on_collapse(ee.edge);
						//if (data.D.size() > 0)
						//	update_direction_field_on_collapse(ee.edge, data.D, data.DW);
						update_face_aspect_ratios_around_vertex(data.V, data.F, data.VF, collapse.vertex, data.FR);

						// decrement counter and erase the collapsed faces from the set of locks, so invalid faces remain locked
						for (int dfi : flap.f) {
							data.fn_curr--;
							//data.faces[dfi] = INVALID_INDEX;
							//locked_faces.erase(dfi);
							//Assert(locked_face_sets[i].erase(dfi) > 0);
							auto itdfi = std::find(locked_face_sets[i].begin(), locked_face_sets[i].end(), dfi);
							Assert(itdfi != locked_face_sets[i].end());
							locked_face_sets[i].erase(itdfi);
						}

						collapsed = true;
						break;
					}
				}
			}

			if (!collapsed) {
				n_consecutive_failures++;
				n_failures_after_cost_increase++;
			}
			else {
				stats.n_collapse++;

				n_consecutive_failures = 0;
				n_failures_after_cost_increase = 0;
			}

			//_unlock(locked_faces, data);
			_unlock(locked_face_sets, data);
		}

		stats.t_total += t.time_elapsed();

	}

	// samples multiple collapses, tracking for each collapse the corresponding face set so that the faces can be unlocked later
	std::vector<EdgeEntry> _sample_collapses(DecimationData& data, std::vector<std::vector<int>>& locked_face_sets)
	{
		Timer t;
		locked_face_sets.clear();

		std::vector<int> faces;
		std::vector<EdgeEntry> collapses;

		uint8_t current_mark = 0;

		for (int i = 0; i < num_collapse_samples; ++i) {
			int fi = _sample_face(data);
			std::vector<int> locked_faces = _lock_region(fi, data, current_mark++);
			if (locked_faces.empty()) { // unlock, clear and return an empty set
				_unlock(locked_face_sets, data);
				locked_face_sets.clear();
				return {};
			}
			else {
				faces.push_back(fi);
				locked_face_sets.push_back(locked_faces);
			}
		}

		stats.t_lock += t.time_elapsed();

		for (int fi : faces) {
			collapses.push_back(_select_random_collapse(fi, data));
		}

		return collapses;
	}

	// samples a face (the edge collapse will be the edge of this face with smaller cost
	int _sample_face(DecimationData& data)
	{
		std::uniform_int_distribution<int> face_distrib(0, int(data.faces.size()) - 1);
		int fi = INVALID_INDEX;

		do {
			//fi = face_distrib(generator);
			fi = data.faces[face_distrib(generator)];
		//} while (data.flags[fi].test_and_set());
		} while (data.flags[fi].test_and_set());

		return fi;
	}

	bool _expand_frontier(const std::set<int>& frontier, std::vector<int>& acquired, DecimationData& data, uint8_t current_mark)
	{
		bool locked = false;
		for (int vi : frontier) {
			Assert(vi < data.V.rows());
			Assert(vi >= 0);
			for (const VFEntry& vfe : data.VF[vi]) {
				locked = data.flags[vfe.first].test_and_set();
				if (locked && _marks[vfe.first] != current_mark) {
					// internal or external collision
					return false;
				}
				else if (!locked) {
					acquired.push_back(vfe.first);
					_marks[vfe.first] = current_mark;
				}
			}
		}

		return true;
	}

	// returns the set of locked faces (if empty, then locking has failed)
	std::vector<int> _lock_region(int fi, DecimationData& data, uint8_t current_mark)
	{
		// set of acquired faces
		std::vector<int> acquired = { fi };

		Assert(_marks[fi] == MarkFree);
		_marks[fi] = current_mark;

		// frontier of vertices to visit with VFAdj
		std::set<int> frontier_0 = { data.F(fi, 0), data.F(fi, 1), data.F(fi, 2) };

		// expand from the starting face
		bool success = _expand_frontier(frontier_0, acquired, data, current_mark);

		// expand one step further
		if (success) {
			std::set<int> frontier_1;

			for (int next_fi : acquired) {
				if (fi != next_fi) {
					frontier_1.insert(data.F(fi, 0));
					frontier_1.insert(data.F(fi, 1));
					frontier_1.insert(data.F(fi, 2));
				}
			}

			success = _expand_frontier(frontier_1, acquired, data, current_mark);
		}

		// return
		if (success) {
			return acquired;
		}
		else {
			_unlock(acquired, data);
			return {};
		}
	}

	void _unlock(const std::vector<int>& faces, DecimationData& data)
	{
		for (int i : faces) {
			data.flags[i].clear();
			_marks[i] = MarkFree;
		}
	}

	void _unlock(const std::vector<std::vector<int>>& facelists, DecimationData& data)
	{
		for (const std::vector<int>& faces : facelists)
			_unlock(faces, data);
	}

	EdgeEntry _select_random_collapse(int fi, DecimationData& data)
	{
		std::uniform_int_distribution<int> edge_distrib(0, 2);
		int e = edge_distrib(generator);
		Assert(data.F(fi, 0) != INVALID_INDEX);
		Edge edge(data.F(fi, e), data.F(fi, (e + 1) % 3));
		CollapseData cd = data.dcb->evaluate_collapse(edge, data.FN, dparams);
		return EdgeEntry(edge, cd, 0);
	}

	EdgeEntry _select_best_collapse(int fi, DecimationData& data)
	{
		EdgeEntry ee;

		for (int i = 0; i < 3; ++i) {
			Edge e(data.F(fi, i), data.F(fi, (i + 1) % 3));
			CollapseData cd = data.dcb->evaluate_collapse(e, data.FN, dparams);
			if (cd.cost < ee.collapse_data.cost) {
				ee = EdgeEntry(e, cd, 0);
			}
		}

		return ee;
	}

};

void decimate_mesh_parallel(MatrixX& V, MatrixXi& F, VFAdjacency& VF, int fn_target,
	std::vector<Quadric>& Q, std::vector<Scalar>& QW,
	MatrixX& D, VectorX& DW, VectorX& FR, MatrixX& FN, const DecimationParameters& dparams, int nw)
{
	using EdgeEntry = Decimation::EdgeEntry;

	Timer t;

	Scalar max_error = std::numeric_limits<Scalar>::max();
	if (dparams.bound_geometric_error) {
		Assert(dparams.max_relative_error >= 0);
		//Scalar hi_avg_edge_length = average_edge(V, F);
		Box3 box;
		for (int i = 0; i < V.rows(); ++i)
			box.add(V.row(i));
		max_error = dparams.max_relative_error * (1e-3 * box.diagonal()).squaredNorm();
	}

	// Init quadrics
	std::shared_ptr<DecimationCallbackSmoothedQuadric> dcb = std::make_shared<DecimationCallbackSmoothedQuadric>(V, F, VF);
	Assert(!dcb->has_quadrics());
	dcb->init_quadrics(dparams.border_error_scaling);

	// Track valid face indices
	std::vector<int> faces(F.rows());
	std::iota(faces.begin(), faces.end(), 0); // [0, ..., F.rows() - 1]

	//std::vector<std::atomic_flag> flags;
	std::vector<std::atomic_flag> flags(faces.size());

	DecimationData data(V, F, VF, dcb, FR, FN, faces, flags);

	// Guess initial error threshold (very low)
	std::mt19937 generator(0);
	std::uniform_int_distribution<int> edge_distrib(0, 2);

	auto sample_collapses = [&](int n) -> std::vector<EdgeEntry> {
		// sample a number of face edges
		std::set<Edge> edges;
		{
			int fn = F.rows();
			std::uniform_int_distribution<int> face_distrib(0, fn - 1);

			while (edges.size() < n) {
				int fi = faces[face_distrib(generator)];
				int e = edge_distrib(generator);
				Assert(F(fi, 0) != INVALID_INDEX);
				Edge edge(F(fi, e), F(fi, (e + 1) % 3));
				edges.insert(edge);
			}
		}

		std::vector<EdgeEntry> collapses;
		for (Edge e : edges) {
			CollapseData cd = dcb->evaluate_collapse(e, FN, dparams);
			collapses.push_back(EdgeEntry(e, cd, 0));
		}

		std::sort(collapses.begin(), collapses.end());

		return collapses;
	};

	std::vector<EdgeEntry> initial_sample = sample_collapses(20);
	Scalar error_value = initial_sample[0].collapse_data.cost; // use the minimum error as initial guess

	std::vector<DecimationWorker> workers;
	for (int i = 0; i < nw; ++i)
		workers.push_back(DecimationWorker(i, error_value, dparams));

	//workers.push_back(DecimationWorker(0, error_value));

	auto launch_worker = [&](int id, int thread_fn_target) -> void {
		workers[id].run(data, thread_fn_target, max_error);
	};

	Timer t_loop;

	while (data.fn_curr > fn_target) {
		// compact face indices
		std::cout << "Iterating... " << data.fn_curr << " / " << fn_target << std::endl;
		data.faces.erase(std::remove_if(data.faces.begin(), data.faces.end(), [](int i) { return i < 0; }), data.faces.end());

		// no need to resize the flags vector
		//data.flags = std::vector<std::atomic_flag>(data.faces.size());
		// these functions are not allowed with atomics, so we need to reallocate the vector
		//data.flags.clear();
		//data.flags.resize(data.faces.size());

		// stop at 40% decimation w.r.t current face count and compact the indices list to speed-up sampling
		int thread_fn_target = std::max(fn_target, int(0.4 * data.fn_curr));

		std::vector<std::thread> threads;

		int fn_iter_start = data.fn_curr;

		for (int i = 0; i < nw; ++i)
			threads.push_back(std::thread(launch_worker, i, thread_fn_target));

		for (int i = 0; i < nw; ++i)
			threads[i].join();

		if (data.fn_curr == fn_iter_start) {
			std::cerr << "Warning: workers iteration did not simplify mesh, interrputing" << std::endl;
			break;
		}
	}

	for (const DecimationWorker& dw : workers) {
		std::cout << "LOG Worker " << dw.wid << std::endl;
		std::cout << "    cost_threshold = " << dw.current_cost_threshold << std::endl;
		std::cout << "    n_iter         = " << dw.stats.n_iter << std::endl;
		std::cout << "    n_collapse     = " << dw.stats.n_collapse << " (" << dw.stats.n_collapse / Scalar(dw.stats.n_iter) << ")" << std::endl;
		std::cout << "    n_fail_error   = " << dw.stats.n_fail_error << " (" << dw.stats.n_fail_error / Scalar(dw.stats.n_iter) << ")" << std::endl;
		std::cout << "    n_fail_expand  = " << dw.stats.n_fail_expand << " (" << dw.stats.n_fail_expand / Scalar(dw.stats.n_iter) << ")" << std::endl;
		std::cout << "    t_total        = " << dw.stats.t_total << " secs" << std::endl;
		std::cout << "    t_lock         = " << dw.stats.t_lock << " secs" << std::endl << std::endl;
	}

	std::cerr << "decimate_mesh_parallel() took " << t.time_elapsed() << " seconds" << std::endl;
	std::cerr << "  t_loop = " << t_loop.time_elapsed() << " seconds" << std::endl;

	Q = dcb->Q;
	QW = dcb->QW;
}

