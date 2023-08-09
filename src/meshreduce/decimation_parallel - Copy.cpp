#include "decimation.h"
#include "local_operations.h"
#include "direction_field.h"
#include "mesh_utils.h"

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

	Scalar current_error_threshold;
	Scalar increase_factor = 2;

	const DecimationParameters& dparams;

	std::vector<bool> visited;

	// Algorithm parameters
	const int step_increase_count = 10; // How many samples above the current error threshold before increasing it
	const int interrupt_count = 500; // How many failed iterations before interrupting execution (~ target error threshold reached)
	const int num_collapse_samples = 3; // How many collapses are sampled at each iteration

	struct {
		int n_iter = 0;
		int n_collapse = 0;
		int n_fail_expand = 0;
		int n_fail_error = 0;

		Scalar t_lock = 0;
		Scalar t_sampling = 0;
		Scalar t_total = 0;
	} stats;

	DecimationWorker(int id, Scalar etol, const DecimationParameters& decimation_parameters)
		: wid(id), generator(id), current_error_threshold(etol), dparams(decimation_parameters)
	{
	}

	void operator()(DecimationData& data, int fn_target, int face_chunk_size)
	{
		run(data, fn_target);
	}

	// deicimation loop
	void run(DecimationData& data, int fn_target)
	{
		const int MAX_LOCK_FAILURES = 20;

		int n_consecutive_lock_failures = 0;

		Timer t;

		int last_collapse = 0;

		visited.clear();
		visited.resize(data.F.rows(), false);

		while (data.fn_curr > fn_target && n_consecutive_lock_failures < MAX_LOCK_FAILURES) {
			Timer tt;
			stats.n_iter++;

			// gradually increase error tolerance when decimating
			if (last_collapse > step_increase_count /* && current_error_threshold < max_error*/) {
				last_collapse = 0;
				//current_error_threshold = std::min(max_error, current_error_threshold * increase_factor);
				current_error_threshold = current_error_threshold * increase_factor;
			}
			else if (last_collapse > interrupt_count) { // stop if too many unsuccessful iterations
				//std::cout << "Too many iterations without collapses, stopping" << std::endl;
				break;
			}
			
			bool collapsed = false; // becomes true if the iteration managed to collapse an edge

			std::vector<std::set<int>> locked_face_sets;
			std::vector<EdgeEntry> collapses = _sample_collapses(data, locked_face_sets);

			// if we don't get any edge, it's because of a collision when trying to lock the local regions
			if (collapses.size() == 0) {
				n_consecutive_lock_failures++;
				stats.n_fail_expand++;
			}

			// sample a face index
			//int fi = _sample_face(data);

			//Assert(fi >= 0);
			//Assert(fi < data.F.rows());
			//Assert(data.F(fi, 0) != INVALID_INDEX);

			//stats.t_sampling += tt.time_since_last_check();

			//// lock_region() or continue
			//std::set<int> locked_faces = _lock_region(fi, data);
			//if (locked_faces.empty()) {
			//	n_consecutive_lock_failures++;
			//	stats.n_fail_expand++;
			//	continue;
			//}

			n_consecutive_lock_failures = 0;

			stats.t_lock += tt.time_since_last_check(); // too slow? test whether it's slow to use atomics (i.e. bools with nw == 1 are faster) or
			// it's the way we are finding the faces that is slow (using vfadj leads to a lot of redundant visits)

			// choose edge with lowest cost
			//EdgeEntry ee = _select_best_collapse(fi, data);
			
			// test collapse

			//for (const EdgeEntry& ee : collapses) {
			for (int i = 0; i < collapses.size(); ++i) {
				const EdgeEntry& ee = collapses[i];

				if (std::isfinite(ee.collapse_data.cost) && ee.collapse_data.geometric_error <= current_error_threshold) {
					// attempt collapse
					Flap flap = compute_flap(ee.edge, data.F, data.VF);
					bool feasible = collapse_preserves_topology(flap, data.F, data.VF, data.dcb->VB);
					if (feasible && dparams.bound_aspect_ratio)
						feasible = collapse_preserves_aspect_ratio(flap, data.V, data.F, data.VF, ee.collapse_data.position, dparams.min_aspect_ratio, data.FR, dparams.aspect_ratio_adaptive_tolerance);

					if (feasible) {
						// collapse edge
						CollapseInfo collapse = collapse_edge(flap, data.V, data.F, data.VF, data.dcb->VB, ee.collapse_data.position);
						Assert(collapse.ok);

						// update data
						data.dcb->update_on_collapse(ee.edge);
						//if (data.D.size() > 0)
						//	update_direction_field_on_collapse(ee.edge, data.D, data.DW);
						update_face_aspect_ratios_around_vertex(data.V, data.F, data.VF, collapse.vertex, data.FR);

						// decrement counter and erase the collapsed faces from the set of locks, so invalid faces remain locked
						for (int dfi : flap.f) {
							data.fn_curr--;
							data.faces[dfi] = INVALID_INDEX;
							//locked_faces.erase(dfi);
							Assert(locked_face_sets[i].erase(dfi) > 0);
						}

						collapsed = true;
						break;
					}
				}
			}

			if (!collapsed) {
				last_collapse++;
			}
			else {
				stats.n_collapse++;
				last_collapse = 0;
			}

			//_unlock(locked_faces, data);
			_unlock(locked_face_sets, data);
		}

		stats.t_total += t.time_elapsed();

	}

	// samples multiple collapses, tracking for each collapse the corresponding face set so that the faces can be unlocked later
	std::vector<EdgeEntry> _sample_collapses(DecimationData& data, std::vector<std::set<int>>& locked_face_sets)
	{
		locked_face_sets.clear();

		std::vector<int> faces;
		std::vector<EdgeEntry> collapses;

		for (int i = 0; i < num_collapse_samples; ++i) {
			int fi = _sample_face(data);
			std::set<int> locked_faces = _lock_region(fi, data);
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

		for (int fi : faces) {
			collapses.push_back(_select_random_collapse(fi, data));
		}

		std::sort(collapses.begin(), collapses.end());
		return collapses;
	}

	// samples a face (the edge collapse will be the edge of this face with smaller cost
	int _sample_face(DecimationData& data)
	{
		std::uniform_int_distribution<int> face_distrib(0, int(data.faces.size()) - 1);
		int fi = INVALID_INDEX;

		do {
			fi = face_distrib(generator);
		} while (data.flags[fi].test_and_set());

		return fi;
	}

	bool _expand_frontier(const std::set<int>& frontier, std::set<int>& acquired, DecimationData& data)
	{
		bool locked = false;
		for (int vi : frontier) {
			Assert(vi < data.V.rows());
			Assert(vi >= 0);
			for (const VFEntry& vfe : data.VF[vi]) {
				if (acquired.count(vfe.first) == 0) {
					locked = data.flags[vfe.first].test_and_set();
					// if the face was already locked, we failed
					if (!locked)
						acquired.insert(vfe.first);
					else
						return false;
				}
			}
		}

		return true;
	}

	// returns the set of locked faces (if empty, then locking has failed)
	std::set<int> _lock_region(int fi, DecimationData& data)
	{
		// set of acquired faces
		std::set<int> acquired = { fi };

		// frontier of vertices to visit with VFAdj
		std::set<int> frontier_0 = { data.F(fi, 0), data.F(fi, 1), data.F(fi, 2) };

		// expand from the starting face
		bool success = _expand_frontier(frontier_0, acquired, data);

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

			success = _expand_frontier(frontier_1, acquired, data);
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

	void _unlock(const std::set<int>& faces, DecimationData& data)
	{
		for (int i : faces)
			data.flags[i].clear();
	}

	void _unlock(const std::vector<std::set<int>>& facesets, DecimationData& data)
	{
		for (const std::set<int>& faces : facesets)
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

	std::cout << "TODO FIXME ADD max_error" << std::endl;

	Timer t;

	// Init quadrics
	std::shared_ptr<DecimationCallbackSmoothedQuadric> dcb = std::make_shared<DecimationCallbackSmoothedQuadric>(V, F, VF);
	Assert(!dcb->has_quadrics());
	dcb->init_quadrics(dparams.border_error_scaling);

	// Track valid face indices
	std::vector<int> faces(F.rows());
	std::iota(faces.begin(), faces.end(), 0); // [0, ..., F.rows() - 1]

	std::vector<std::atomic_flag> flags;

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
		workers[id].run(data, thread_fn_target);
	};

	Timer t_loop;

	int thread_fn_target = data.fn_curr;
	int fn_decimation_step = 1000000; // every 1M decimated faces, compact the face data for faster sampling

	while (data.fn_curr > fn_target) {
		//if (data.fn_curr != int(data.faces.size())) {
			std::cout << "Compacting data" << std::endl;
			//Assert(0 && "broken, need to have the inverted index when indices are compacted and no-longerm mapped 1:1 with array positions");
			data.faces.erase(std::remove_if(data.faces.begin(), data.faces.end(), [](int i) { return i < 0; }), data.faces.end());

			data.flags = std::vector<std::atomic_flag>(data.faces.size());
			// these functions are not allowed with atomics, so we need to reallocate the vector
			//data.flags.clear();
			//data.flags.resize(data.faces.size());
		//}

		std::vector<std::thread> threads;

		thread_fn_target = std::max(thread_fn_target - fn_decimation_step, fn_target);

		for (int i = 0; i < nw; ++i)
			threads.push_back(std::thread(launch_worker, i, thread_fn_target));

		for (int i = 0; i < nw; ++i)
			threads[i].join();
	}

	for (const DecimationWorker& dw : workers) {
		std::cout << "LOG Worker " << dw.wid << std::endl;
		std::cout << "    err_threshold = " << dw.current_error_threshold << std::endl;
		std::cout << "    n_iter        = " << dw.stats.n_iter << std::endl;
		std::cout << "    n_collapse    = " << dw.stats.n_collapse << " (" << dw.stats.n_collapse / Scalar(dw.stats.n_iter) << ")" << std::endl;
		std::cout << "    n_fail_error  = " << dw.stats.n_fail_error << " (" << dw.stats.n_fail_error / Scalar(dw.stats.n_iter) << ")" << std::endl;
		std::cout << "    n_fail_expand = " << dw.stats.n_fail_expand << " (" << dw.stats.n_fail_expand / Scalar(dw.stats.n_iter) << ")" << std::endl;
		std::cout << "    t_total       = " << dw.stats.t_total << " secs" << std::endl;
		std::cout << "    t_lock        = " << dw.stats.t_lock << " secs" << std::endl << std::endl;
		std::cout << "    t_sampling    = " << dw.stats.t_sampling << " secs" << std::endl << std::endl;
	}

	//write_obj("before_flip.obj", V, F);
	//int nflips = align_to_direction_field(V, F, VF, VB, D, DW);
	//write_obj("after_flip.obj", V, F);
	//std::cerr << "align_to_direction_field() flipped " << nflips << " edges" << std::endl;

	std::cerr << "decimate_mesh_parallel() took " << t.time_elapsed() << " seconds" << std::endl;
	std::cerr << "  t_loop = " << t_loop.time_elapsed() << " seconds" << std::endl;

	Q = dcb->Q;
	QW = dcb->QW;
}
