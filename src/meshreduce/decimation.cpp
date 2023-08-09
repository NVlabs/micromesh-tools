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
#include "space.h"
#include "adjacency.h"
#include "local_operations.h"
#include "mesh_utils.h"
#include "utils.h"
#include "flip.h"
#include "clean.h"
#include "aabb.h"
#include "direction_field.h"

#include "visibility.h"

#include "flip.h"

#include <queue>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <random>


// -- DecimationCallbackSmoothedQuadric class -------------------

DecimationCallbackSmoothedQuadric::DecimationCallbackSmoothedQuadric(const MatrixX& VV, const MatrixXi& FF, const VFAdjacency& VFF)
	: V(VV), F(FF), VF(VFF), Q(), QW()
{
	per_vertex_border_flag(V, F, VF, VB);
	std::tie(VD, VIS) = compute_optimal_visibility_directions(V, F, VF);
}

bool DecimationCallbackSmoothedQuadric::has_quadrics() const
{
	return Q.size() > 0;
}

bool DecimationCallbackSmoothedQuadric::init_quadrics(Scalar border_error_scaling)
{
	if (!has_quadrics()) {
		Assert(border_error_scaling >= 0);
		compute_quadrics_per_vertex(V, F, VF, VB, border_error_scaling, Q, QW);
		return true;
	}

	return false;
}

Quadric DecimationCallbackSmoothedQuadric::get_edge_quadric(Edge e) const
{
	return (Q[e.first] + Q[e.second]) / (QW[e.first] + QW[e.second]);
}

void DecimationCallbackSmoothedQuadric::_compute_position_and_error(Edge e, Vector3& p, Scalar& error) const
{
	Quadric quadric = get_edge_quadric(e);
	p = quadric.minimizer();
	error = quadric.error(p);
}

CollapseData DecimationCallbackSmoothedQuadric::evaluate_collapse(Edge e, const MatrixX& FN, const DecimationParameters& dparams) const
{
	CollapseData cd;

	// first compute the optimal vertex position and its geometric error
	if (!dparams.use_vertex_smoothing || VB(e.first) || VB(e.second)) {
		// if vertex smoothing is not enabled or a border vertex is involved,
		// fall back to the simple vertex quadrics
		_compute_position_and_error(e, cd.position, cd.geometric_error);
	}
	else {
		// if vertex smoothing is enabled, compute the optimal vertex position (QSLIM),
		// compute the smoothed position and its squared distance quadric
		// blend the two quadrics to find the final position
		int deg = F.cols();

		// compute tangent planes at edge endpoints
		Plane p1(V.row(e.first), compute_vertex_normal(e.first, V, F, VF));
		Plane p2(V.row(e.second), compute_vertex_normal(e.second, V, F, VF));

		// compute barycenter q of 1-ring neighborhood of edge
		std::unordered_set<int> vset(16);
		for (const VFEntry& vfe : VF[e.first]) {
			int i = vfe.first;
			for (int j = 0; j < deg; ++j)
				vset.insert(F(i, j));
		}

		for (const VFEntry& vfe : VF[e.second]) {
			int i = vfe.first;
			for (int j = 0; j < deg; ++j)
				vset.insert(F(i, j));
		}
		vset.erase(e.first);
		vset.erase(e.second);

		Vector3 q(0, 0, 0);
		for (int vi : vset) {
			q += V.row(vi);
		}
		q /= vset.size();

		// project q onto tangent planes and choose the projection that yields
		// the smallest error relative to the edge quadric as 'pulling' point
		Quadric edge_quadric = get_edge_quadric(e);

		Vector3 l1 = project(q, p1);
		Vector3 l2 = project(q, p2);

		Vector3 l = edge_quadric.error(l1) < edge_quadric.error(l2) ? l1 : l2;
		Quadric pull = compute_point_quadric(l);

		// sum edge quadric and pull
		Quadric quadric = edge_quadric + (dparams.smoothing_coefficient * pull);

		// the new position is the minimizer of the summed quadric
		cd.position = quadric.minimizer();

		// *** the geometric error is measured with the EDGE QUADRIC ONLY ***
		cd.geometric_error = edge_quadric.error(cd.position);
	}

	// next, set the cost
	cd.cost = cd.geometric_error;

	Flap flap = compute_flap(e, F, VF);

	Scalar normal_correlation = 1;
	Scalar visibility = 1;
	Scalar ar = 1;

	// compute the penalties only if the faces are not `almost flipped`, otherwise it is
	// not possible to `unfold` the surface during simplification (it creates vertices with
	// negative visibility in doing so)

	if (dparams.normals_scaling > 0) {
		// Using the original cached face normals to determine the normal correlation
		normal_correlation = compute_collapse_normal_correlation(flap, V, F, VF, cd.position, FN);
		// normalize in the range [0,1]
		//normal_correlation = (normal_correlation + 1) / Scalar(2);
		normal_correlation = clamp<Scalar>(normal_correlation, 0, 1);
	}

	// if one of the vertices has negative visibility, go ahead with the collapse anyway (risky...)
	if (dparams.visibility_scaling > 0) {
		if (VIS(e.first) > 0 && VIS(e.second) > 0)
			visibility = compute_collapse_visibility_approx2(flap, V, F, VF, cd.position, VD, VIS);
		else
			visibility = 0.01; // just use a very small threshold value
	}

	if (dparams.ar_scaling > 0) {
		ar = compute_collapse_aspect_ratio(flap, V, F, VF, cd.position);
	}

	if (visibility == 0 || ar == 0 || normal_correlation == 0) {
		cd.cost = Infinity;
		if (ar == 0)
			cd._infinite_cost_flags |= CollapseData::_AspectRatio_Flag;
		if (visibility == 0)
			cd._infinite_cost_flags |= CollapseData::_Visibility_Flag;
		if (normal_correlation == 0)
			cd._infinite_cost_flags |= CollapseData::_Normals_Flag;
	}
	else {
		cd.cost = cd.geometric_error / (std::pow(visibility, dparams.visibility_scaling) * std::pow(ar, dparams.ar_scaling) * std::pow(normal_correlation, dparams.normals_scaling));
	}
	
	return cd;
}

void DecimationCallbackSmoothedQuadric::update_on_collapse(Edge e)
{
	// sum quadrics
	Q[e.first].add(Q[e.second]);
	QW[e.first] += QW[e.second];

	// update visibility data (not really being efficient here...)
	std::set<int> verts;
	for (const VFEntry vfe : VF[e.first]) {
		for (int i = 0; i < 3; ++i) {
			verts.insert(F(vfe.first, i));
		}
	}

	for (int vi : verts) {
		Scalar vis_pre = VIS(vi);
		Vector3 dir_pre = VD.row(vi);
		std::vector<Vector3> area_vectors;
		for (const VFEntry vfe : VF[vi])
			area_vectors.push_back(compute_area_vector(V.row(F(vfe.first, 0)), V.row(F(vfe.first, 1)), V.row(F(vfe.first, 2))).normalized());
		std::pair<Vector3, Scalar> visibility_data = compute_positive_visibility_from_directions(area_vectors);
		VD.row(vi) = visibility_data.first;
		VIS(vi) = visibility_data.second;
	}
}

void DecimationCallbackSmoothedQuadric::update_on_split(Edge e, int new_vertex, const Vector3& new_vertex_pos)
{
	Assert(new_vertex == int(Q.size()));
	Assert(new_vertex == int(QW.size()));

	Quadric qp = compute_point_quadric(new_vertex_pos);
	Scalar wp = QW[e.first] + QW[e.second];

	Q.push_back(Q[e.first] + Q[e.second] + wp * qp);
	QW.push_back(QW[e.first] + QW[e.second] + wp);

	// compute the visibility of the new vertex
	std::vector<Vector3> area_vectors;
	for (const VFEntry vfe : VF[new_vertex])
		area_vectors.push_back(compute_area_vector(V.row(F(vfe.first, 0)), V.row(F(vfe.first, 1)), V.row(F(vfe.first, 2))).normalized());
	
	VIS.conservativeResize(Q.size());
	VD.conservativeResize(Q.size(), VD.cols());
	std::pair<Vector3, Scalar> visibility_data = compute_positive_visibility_from_directions(area_vectors);
	VD.row(new_vertex) = visibility_data.first;
	VIS(new_vertex) = visibility_data.second;
}

void DecimationCallbackSmoothedQuadric::update_on_vertex_split(int old_vertex, int new_vertex)
{
	Assert(new_vertex == int(Q.size()));
	Assert(new_vertex == int(QW.size()));

	Quadric qp = Q[old_vertex];
	Scalar wp = QW[old_vertex];

	Q.push_back(qp);
	QW.push_back(wp);

	Assert(0 && "TODO update visibility stuff");
}


// -- Decimation class -------------------

Decimation::Decimation(const MatrixX& hi_VV, const MatrixXi& hi_FF, MatrixX& VV, MatrixXi& FF, VFAdjacency& VFF, MatrixX& DD, VectorX& DWW, VectorX& FFR, MatrixX& FFN)
	: _parameters(), hi_V(hi_VV), hi_F(hi_FF), V(VV), F(FF), VF(VFF), D(DD), DW(DWW), FR(FFR), FN(FFN)
{
	//per_vertex_border_flag(V, F, VB);
	dcb = std::make_unique<DecimationCallbackSmoothedQuadric>(V, F, VF);
	hi_VB = dcb->VB;
}

void Decimation::_import_decimation_mesh(const MatrixX& iV, const MatrixXi& iF, const VFAdjacency& iVF,
	const std::vector<Quadric>& iQ, const std::vector<Scalar>& iQW,
	const MatrixX& iD, const VectorX& iDW, const VectorX& iFR, const MatrixX& iFN)
{
	V = iV;
	F = iF;
	VF = iVF;

	dcb->Q = iQ;
	dcb->QW = iQW;
	per_vertex_border_flag(V, F, dcb->VB);
	std::tie(dcb->VD, dcb->VIS) = compute_optimal_visibility_directions(V, F, VF);

	D = iD;
	DW = iDW;

	FR = iFR;
	FN = iFN;
}

void Decimation::init(const DecimationParameters& parameters)
{
	_debug = {};
	_parameters = parameters;

	if (!dcb->has_quadrics())
		dcb->init_quadrics(_parameters.border_error_scaling);

	_max_error = std::numeric_limits<Scalar>::max();
	if (_parameters.bound_geometric_error) {
		Assert(_parameters.max_relative_error >= 0);
		//Scalar hi_avg_edge_length = average_edge(hi_V, hi_F);
		Box3 box;
		for (int i = 0; i < hi_V.rows(); ++i)
			box.add(hi_V.row(i));
		_max_error = _parameters.max_relative_error * (1e-3 * box.diagonal()).squaredNorm();
	}

	timer.reset();

	t_init = 0;
	t_pop = 0;
	t_collapse = 0;
	t_collapse_wasted = 0;
	t_update = 0;

	collapse_not_ok = 0;
	collapse_ok = 0;
	update = 0;
	nflips = 0;

	nreject_topology = 0;
	nreject_geometry = 0;
	nreject_aspect_ratio = 0;
	nreject_error = 0;
	nreject_cone = 0;
	
	_rebuild_heap();
	
	decimated_to_input_verts.clear();
	decimated_to_input_verts.resize(V.rows());
	for (int i = 0; i < (int)V.rows(); ++i)
		decimated_to_input_verts[i] = { i };

	fn_curr = 0;
	for (int i = 0; i < F.rows(); ++i)
		if (F(i, 0) != INVALID_INDEX)
			fn_curr++;
	
	fn_start = fn_curr;

	t_init += timer.time_elapsed();
}

// this function returns true if a next valid collapse is in the
// queue ( *** WHITOUT *** popping the valid entry) and the corresponding
// flap, otherwise it returns false (which implies that q.empty() == true)
bool Decimation::next_valid(Flap& flap)
{
	while (!queue_empty()) {
		//EdgeEntry entry = q.top();
		EdgeEntry entry = _heap.front();

		auto ittime = tmap.find(entry.edge);

		if (ittime != tmap.end() && ittime->second == entry.time) {

			_debug.n_eval++;

			flap = compute_flap(entry.edge, F, VF);
			int status = _test_operation(entry, flap);

			opstatus[entry.edge] = status;

			if (status == OpStatus_Feasible) {
				return true;
			}
			else {
				collapse_not_ok++;
				if (status & OpStatus_FailTopology)
					nreject_topology++;
				if (status & OpStatus_FailNormals)
					nreject_geometry++;
				if (status & OpStatus_FailAspectRatio)
					nreject_aspect_ratio++;
				if (status & OpStatus_FailGeometricError)
					nreject_error++;
				if (status & OpStatus_FailVertexRingNormals)
					nreject_cone++;
				_pop_operation();
			}
		}
		else {
			_debug.n_obsolete++;
			_pop_operation();
		}
	}
	Assert(queue_empty());
	return false;
}

int Decimation::execute(int fn_target)
{
	timer.reset();

	int deg = F.cols();

	int fn_step = (fn_curr - fn_target) / 10;
	int next_step = fn_curr - fn_step;

	while (true) {

		if (fn_curr <= fn_target)
			break;

		Flap flap;
		bool valid = next_valid(flap);
		if (!valid) {
			break;
		}
		else {
			EdgeEntry entry = _heap.front();

			_pop_operation();

			CollapseInfo collapse = collapse_edge(flap, V, F, VF, dcb->VB, entry.collapse_data.position);
			collapse_ok++;

			{
				std::set<int>& s1 = decimated_to_input_verts[collapse.vertex];
				std::set<int>& s2 = decimated_to_input_verts[collapse.collapsed_vertex];

				s1.insert(s2.begin(), s2.end());

				decimated_to_input_verts[collapse.collapsed_vertex].clear();
			}

			V.row(collapse.vertex) = entry.collapse_data.position;
			dcb->update_on_collapse(entry.edge);
			if (D.size() > 0)
				update_direction_field_on_collapse(entry.edge, D, DW);

			//tmap[entry.edge]++;
			tmap.erase(entry.edge);
			for (const Edge& e : collapse.expired) {
				tmap.erase(e); //tmap[e]++;
				opstatus.erase(e);
			}

			// Update aspect ratio record
			update_face_aspect_ratios_around_vertex(V, F, VF, collapse.vertex, FR);

			// vertices affected by the operation
			// i.e. whose incident edges must be re-evaluated
			std::set<int> verts; 
			
			// Local edge flips
			if (_parameters.local_flips_after_collapse) {
				std::set<Edge> star;
				for (const VFEntry& vfe : VF[collapse.vertex]) {
					star.insert(Edge(F(vfe.first, vfe.second), F(vfe.first, (vfe.second + 1) % deg)));
				}

				LocalEdgeFlip local_flips(V, F, VF, dcb, D, DW);
				local_flips.init(star);
				local_flips.forbid_new_flips = true; // only flip edges from the vertex star
				nflips += local_flips.execute();

				for (const Edge& e : local_flips.obsolete) {
					//tmap[e]++;
					tmap.erase(e);
					opstatus.erase(e);
				}

				for (const Edge& e : local_flips.fresh) {
					// for each flipped edge, add the 4 quad vertices to the list of updates
					std::set<int> shared = shared_faces(VF, e.first, e.second);
					for (int fi : shared)
						for (int j = 0; j < deg; ++j)
							verts.insert(F(fi, j));
				}
			}

			// reinsert all edges from faces adjacent to the surviving vertex

			//if (test_aspect_ratio || test_vertex_ring_normals) {
			if (_needs_large_update()) {
				// since the change in AR of the faces around the collapsed edge
				// may affect the feasibility of *all* the edges incident to the 1RR
				// of the surviving vertex, readding only the edges of the 1RR is not
				// sufficient
				for (const VFEntry& vfe : VF[collapse.vertex])
					for (int j = 0; j < deg; ++j)
						verts.insert(F(vfe.first, j));

				//std::set<int> verts2;
				//for (int vi : verts) {
				//	for (const VFEntry& vfe : VF[vi])
				//		for (int j = 0; j < deg; ++j)
				//			verts2.insert(F(vfe.first, j));
				//}
				//verts = verts2;
			}
			else {
				verts.insert(collapse.vertex);
			}

			std::set<Edge> updated;
			for (int vi : verts) {
				for (const VFEntry& vfe : VF[vi]) {
					int fi = vfe.first;
					for (int j = 0; j < deg; ++j) {
						Edge e(F(fi, j), F(fi, (j + 1) % deg));
						if (verts.find(e.first) != verts.end() || verts.find(e.second) != verts.end())
							updated.insert(e);
					}
				}
			}

			_push_operations(updated);

			fn_curr -= flap.size();

			if (fn_curr < next_step) {
				std::cout << "  Face count " << fn_curr << " (" << timer.time_elapsed() << " secs since decimation start)"
					<< std::endl;
				//std::cout << " ------- tmap.size() == " << tmap.size() << "   q.size() == " << q.size() << std::endl;
				std::cout << " ------- tmap.size() == " << tmap.size() << "   q.size() == " << _heap.size() << std::endl;
				//_compact_heap();
				//std::cout << "    ---- tmap.size() == " << tmap.size() << "   q.size() == " << _heap.size() << std::endl;
				//if (reproject_quadrics) {
				//	_refresh_quadrics();
				//	_rebuild_heap();
				//}
				next_step -= fn_step;
			}
		}
	}

	if (_parameters.reproject_quadrics) {
		_refresh_quadrics();
		_rebuild_heap();
	}

	double t_loop = timer.time_elapsed();

	std::cout << "Performed " << collapse_ok << " edge collapses" << std::endl;
	std::cout << "     Average quadric updates per collapse: " << update / float(collapse_ok) << std::endl;
	std::cout << "     Average local flips per collapse: " << nflips / float(collapse_ok) << std::endl;
	std::cout << "     Number of evaluated ops: " << _debug.n_eval << std::endl;
	std::cout << "     Number of obsolete ops discarded from queue: " << _debug.n_obsolete << std::endl;
	std::cout << "Rejected " << collapse_not_ok << " collapses due to infeasibility" << std::endl;
	std::cout << "     nreject_topology " << nreject_topology << std::endl;
	std::cout << "     nreject_geometry " << nreject_geometry << std::endl;
	std::cout << "     nreject_aspect_ratio " << nreject_aspect_ratio << std::endl;
	std::cout << "     nreject_error " << nreject_error << std::endl;
	std::cout << "     nreject_cone " << nreject_cone << std::endl;
	std::cout << "Decimation took " << t_init + t_loop << " secs" << std::endl;
	std::cout << "     t_init " << t_init << " secs" << std::endl;
	std::cout << "     t_loop " << t_loop << " secs" << std::endl;
	std::cout << "       t_pop " << t_pop << " secs" << std::endl;
	std::cout << "       t_collapse " << t_collapse << " secs" << std::endl;
	std::cout << "       t_update " << t_update << " secs" << std::endl;
	std::cout << "       t_collapse_wasted " << t_collapse_wasted << " secs" << std::endl;

	// integrity check
	for (int i = 0; i < F.rows(); ++i) {
		int ni = 0;
		for (int j = 0; j < deg; ++j)
			if (F(i, j) == INVALID_INDEX)
				ni++;
		Assert(ni == 0 || ni == deg);
	}

	return fn_curr;
}

bool Decimation::_needs_large_update() const
{
	return _parameters.ar_scaling > 0 || _parameters.visibility_scaling > 0 || _parameters.normals_scaling > 0
		|| _parameters.bound_aspect_ratio || _parameters.bound_normals_correlation;
}

void Decimation::_refresh_quadrics()
{
	map_quadrics_onto_mesh(hi_V, hi_F, hi_VB, _parameters.border_error_scaling, V, F, dcb->VB, dcb->Q, dcb->QW);
	_move_vertices_in_quadrics_min();
}

void Decimation::_move_vertices_in_quadrics_min()
{
	for (unsigned i = 0; i < V.rows(); ++i) {
		V.row(i) = dcb->Q[i].minimizer();
	}
}

void Decimation::_rebuild_heap()
{
	_heap = std::vector<EdgeEntry>();
	tmap = std::unordered_map<Edge, Timestamp>();
	opstatus = std::map<Edge, int>();

	int deg = F.cols();
	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			for (int j = 0; j < deg; ++j) {
				Edge e(F(i, j), F(i, (j + 1) % deg));
				Assert(VF[e.first].size() > 0);
				Assert(VF[e.second].size() > 0);
				if (tmap.find(e) == tmap.end()) {
					_push_operation(e);
				}
			}
		}
	}
}

void Decimation::_compact_heap()
{
	std::vector<EdgeEntry> heap_new;
	for (const EdgeEntry& ee : _heap) {
		auto it = tmap.find(ee.edge);
		if (it != tmap.end() && it->second == ee.time)
			heap_new.push_back(ee);
	}
	std::make_heap(heap_new.begin(), heap_new.end(), std::greater<EdgeEntry>());
	_heap = heap_new;
}

void Decimation::_push_operation(const Edge& e)
{
	auto itmap = tmap.find(e);
	CollapseData cd = dcb->evaluate_collapse(e, FN, _parameters);
	Timestamp et = (itmap == tmap.end()) ? 0 : (itmap->second + 1);

	_heap.push_back(EdgeEntry(e, cd, et));
	std::push_heap(_heap.begin(), _heap.end(), std::greater<EdgeEntry>());

	tmap[e] = et;
	opstatus[e] = OpStatus_Unknown;
}

void Decimation::_push_operations(const std::set<Edge>& edges)
{
	for (int i = 0; i < (int)edges.size(); ++i) {
		auto itedge = edges.begin();
		std::advance(itedge, i);
		const Edge& e = *itedge;

		CollapseData cd = dcb->evaluate_collapse(e, FN, _parameters);

		auto itmap = tmap.find(e);
		Timestamp et = (itmap == tmap.end()) ? 0 : (itmap->second + 1);

		_heap.push_back(EdgeEntry(e, cd, et));
		std::push_heap(_heap.begin(), _heap.end(), std::greater<EdgeEntry>());

		tmap[e] = et;
		opstatus[e] = OpStatus_Unknown;
	}
}

void Decimation::_pop_operation()
{
	std::pop_heap(_heap.begin(), _heap.end(), std::greater<EdgeEntry>());
	_heap.pop_back();
}

int Decimation::_test_operation(const EdgeEntry& ee, const Flap& flap) const
{
	int status = OpStatus_Feasible;

	if (ee.collapse_data._infinite_cost_flags) {
		if (ee.collapse_data._infinite_cost_flags & CollapseData::_Visibility_Flag)
			status |= OpStatus_FailVertexRingNormals;
		else if (ee.collapse_data._infinite_cost_flags & CollapseData::_AspectRatio_Flag)
			status |= OpStatus_FailAspectRatio;
		else if (ee.collapse_data._infinite_cost_flags & CollapseData::_Normals_Flag)
			status |= OpStatus_FailNormals;
	}
	else {
		if (_parameters.bound_geometric_error && ee.collapse_data.geometric_error >= _max_error)
			status |= OpStatus_FailGeometricError;
		else if (_parameters.preserve_topology && !collapse_preserves_topology(flap, F, VF, dcb->VB))
			status |= OpStatus_FailTopology;
		else if (_parameters.bound_normals_correlation
			&& !(collapse_preserves_geometry(flap, V, F, VF, ee.collapse_data.position, _parameters.min_normal_correlation, _parameters.min_border_normal_correlation, dcb->VB, FN)
				&& collapse_preserves_orientation_topology(flap, F, VF, dcb->VB)))
			status |= OpStatus_FailNormals;
		else if (_parameters.bound_aspect_ratio && !collapse_preserves_aspect_ratio(flap, V, F, VF, ee.collapse_data.position, _parameters.min_aspect_ratio, FR, _parameters.aspect_ratio_adaptive_tolerance))
			status |= OpStatus_FailAspectRatio;
	}

	return status;
}

std::pair<Decimation::EdgeEntry, int> Decimation::_get_collapse_data(int fi, int e) const
{
	int deg = F.cols();
	Assert(F(fi, 0) != INVALID_INDEX);
	Assert(e >= 0);
	Assert(e < deg);

	Edge edge(F(fi, e), F(fi, (e + 1) % deg));
	CollapseData cd = dcb->evaluate_collapse(edge, FN, _parameters);

	Flap flap = compute_flap(edge, F, VF);

	EdgeEntry ee(edge, cd, -1);
	int status = _test_operation(ee, flap);

	return std::make_pair(ee, status);
}

void Decimation::log_collapse_data(const EdgeEntry& ee, int status) const
{
	std::cout << "== collapse_edge(" << ee.edge.first << ", " << ee.edge.second << ") ====================" << std::endl;
	std::cout << "VIS(" << ee.edge.first << ") = " << dcb->VIS(ee.edge.first) << std::endl;
	std::cout << "VIS(" << ee.edge.second << ") = " << dcb->VIS(ee.edge.second) << std::endl;
	std::cout << "\tcost = " << ee.collapse_data.cost << std::endl;
	std::cout << "\terror = " << ee.collapse_data.geometric_error << std::endl;
	std::cout << "\tAR_FLAG  = " << bool(ee.collapse_data._infinite_cost_flags & CollapseData::_AspectRatio_Flag) << std::endl;
	std::cout << "\tVIS_FLAG = " << bool(ee.collapse_data._infinite_cost_flags & CollapseData::_Visibility_Flag) << std::endl;
	std::cout << "\tNRM_FLAG = " << bool(ee.collapse_data._infinite_cost_flags & CollapseData::_Normals_Flag) << std::endl;
	std::cout << "\tFeasibility = " << (status == OpStatus_Feasible) << std::endl;
	if (status != OpStatus_Feasible) {
		std::cout << "\t\tERROR = " << bool(status & OpStatus_FailGeometricError) << std::endl;
		std::cout << "\t\tTOPOLOGY = " << bool(status & OpStatus_FailTopology) << std::endl;
		std::cout << "\t\tASPECTRATIO= " << bool(status & OpStatus_FailAspectRatio) << std::endl;
		std::cout << "\t\tVISIBILITY = " << bool(status & OpStatus_FailVertexRingNormals) << std::endl;
	}
}

bool Decimation::queue_empty()
{
	return _heap.size() == 0;
}

void Decimation::compact_faces()
{
	remove_degenerate_faces_inplace(F, FR, FN);
	std::vector<int> remap_vi = compact_vertex_data(F, V, dcb->VB, dcb->Q, dcb->QW, dcb->VD, dcb->VIS, D, DW);

	std::unordered_map<Edge, Timestamp> tmap_;
	std::map<Edge, int> opstatus_;
	for (const std::pair<Edge, Timestamp>& entry : tmap) {
		Edge e = entry.first;
		Edge re = Edge(remap_vi[e.first], remap_vi[e.second]);
		tmap_[re] = tmap[e];
		opstatus_[re] = opstatus[e];
	}
	tmap = tmap_;
	opstatus = opstatus_;
}

void Decimation::manage_split(const SplitInfo& si)
{
	dcb->update_on_split(si.e, si.u, si.split_position);

	int new_faces = si.f2 == INVALID_INDEX ? 1 : 2;
	
	FR.conservativeResize(FR.rows() + new_faces);
	FR(si.f12) = FR(si.f1);
	if (new_faces == 2)
		FR(si.f22) = FR(si.f2);
	
	FN.conservativeResize(FN.rows() + new_faces, FN.cols());
	FN.row(si.f12) = FN.row(si.f1);
	if (new_faces == 2)
		FN.row(si.f22) = FN.row(si.f2);
}

void Decimation::manage_vertex_split(const VertexSplitInfo& vsi)
{
	dcb->update_on_vertex_split(vsi.old_vertex, vsi.new_vertex);

	int new_faces = 2;
	
	FR.conservativeResize(FR.rows() + new_faces);
	FR(vsi.f1) = 0;
	FR(vsi.f2) = 0;
	
	FN.conservativeResize(FN.rows() + new_faces, FN.cols());
	FN.row(vsi.f1) = FN.row(vsi.f1_old);
	FN.row(vsi.f2) = FN.row(vsi.f2_old);
}

void decimate_mesh_fast(MatrixX& V, MatrixXi& F, VFAdjacency& VF, int fn_target,
	std::vector<Quadric>& Q, std::vector<Scalar>& QW,
	MatrixX& D, VectorX& DW, VectorX& FR, MatrixX& FN, const DecimationParameters& dparams)
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

	//std::cout << "Starting mesh decimation" << std::endl;

	// Init quadrics
	std::shared_ptr<DecimationCallbackSmoothedQuadric> dcb = std::make_shared<DecimationCallbackSmoothedQuadric>(V, F, VF);
	Assert(!dcb->has_quadrics());
	
	dcb->init_quadrics(dparams.border_error_scaling);

	// Track valid face indices
	int fn = F.rows();

	std::map<int, int> remap;
	std::vector<int> valid_faces;
	valid_faces.reserve(fn);
	for (int i = 0; i < fn; ++i) {
		Assert(F(i, 0) != INVALID_INDEX);
		valid_faces.push_back(i);
		remap[i] = i;
	}

	// Algorithm parameters
	const int step_increase_count = 20; // How many samples above the current cost threshold before increasing it
	const int interrupt_count = 500; // How many failed iterations before interrupting execution (~ target error threshold reached)
	const int num_collapse_samples = 3; // How many collapses are sampled at each iteration

	const Scalar increase_factor = 1.3;

	// Prepare data for random sampling of edge collapses
	std::mt19937 generator(0);
	std::uniform_int_distribution<int> edge_distrib(0, 2);

	auto sample_collapses = [&](int n) -> std::vector<EdgeEntry> {
		// sample a number of face edges
		std::set<Edge> edges;
		{
			std::uniform_int_distribution<int> face_distrib(0, fn - 1);

			while (edges.size() < n) {
				int fi = valid_faces[face_distrib(generator)];
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
	Scalar current_cost_threshold = 0;// initial_sample[0].collapse_data.cost; // use the minimum error as initial guess
	for (const EdgeEntry& ee : initial_sample) {
		if (ee.collapse_data.cost > 0) {
			current_cost_threshold = ee.collapse_data.cost;
			break;
		}
	}
	if (current_cost_threshold == 0) {
		std::cout << "Warning: using small init threshold because initial guess was 0" << std::endl;
		current_cost_threshold = 10 * std::numeric_limits<Scalar>::min();
	}

	std::cout << "Initial cost guess is " << current_cost_threshold << std::endl;

	int n_collapsed = 0;
	int n_failures_after_cost_increase = 0;

	long n_iters = 0;

	int fn_start = fn;
	int n_to_decimate = std::max(fn - fn_target , 1);

	Timer t_loop;

	Scalar max_collapsed_cost = 0;
	int n_consecutive_failures = 0;

	while (fn > fn_target) {

		if (n_consecutive_failures > interrupt_count && fn < fn_start) { // stop if too many unsuccessful iterations AND some decimation happened
			//std::cout << "Too many iterations without collapses, stopping" << std::endl;
			break;
		}

		// gradually increase error tolerance when decimating
		if (n_failures_after_cost_increase > step_increase_count) {
			n_failures_after_cost_increase = 0;
			current_cost_threshold = current_cost_threshold * increase_factor;
			//std::cout << "Increasing cost threshold to " << current_cost_threshold << " after " << n_collapsed << " collapses" << std::endl;
		}

		std::vector<EdgeEntry> collapses = sample_collapses(num_collapse_samples);

		bool collapsed = false;
		for (const EdgeEntry& ee : collapses) {
			if (std::isfinite(ee.collapse_data.cost) && ee.collapse_data.cost <= current_cost_threshold) { // for each feasible operation
				// attempt collapse
				Flap flap = compute_flap(ee.edge, F, VF);
				bool feasible = (!dparams.bound_geometric_error) || (ee.collapse_data.geometric_error <= max_error);
				if (feasible)
					feasible = collapse_preserves_topology(flap, F, VF, dcb->VB);
				if (feasible && dparams.bound_aspect_ratio)
					feasible = collapse_preserves_aspect_ratio(flap, V, F, VF, ee.collapse_data.position, dparams.min_aspect_ratio, FR, dparams.aspect_ratio_adaptive_tolerance);

				// if success, update data and break
				if (feasible) {
					CollapseInfo collapse = collapse_edge(flap, V, F, VF, dcb->VB, ee.collapse_data.position);
					Assert(collapse.ok);
					dcb->update_on_collapse(ee.edge);
					if (D.size() > 0)
						update_direction_field_on_collapse(ee.edge, D, DW);
					update_face_aspect_ratios_around_vertex(V, F, VF, collapse.vertex, FR);

					for (int dfi : flap.f) {
						--fn;
						remap[valid_faces[fn]] = remap[dfi];
						std::swap(valid_faces[remap[dfi]], valid_faces[fn]);
					}

					collapsed = true;

					max_collapsed_cost = std::max(max_collapsed_cost, ee.collapse_data.cost);

					break;
				}
			}
		}

		if (!collapsed) {
			n_failures_after_cost_increase++;
			n_consecutive_failures++;
		}
		else {
			n_collapsed++;

			n_failures_after_cost_increase = 0;
			n_consecutive_failures = 0;
		}

		n_iters++;
		if (n_iters % 10000 == 0) {
			int n_decimated = fn_start - fn;
			Scalar perc = n_decimated / (Scalar)n_to_decimate;
			std::cout << "Proxy decimation at " << int(perc * 100) << "%\r";
		}
	}
	std::cout << "Proxy decimation at 100%" << std::endl;
	std::cout << "Final cost threshold: " << current_cost_threshold << " (max collapsed cost = " << max_collapsed_cost << ")" << std::endl;

	std::cerr << "decimate_mesh_fast() took " << t.time_elapsed() << " seconds, " << n_collapsed << " collapses" << std::endl;
	std::cerr << "    t_loop = " << t_loop.time_elapsed() << " seconds" << std::endl;

	Q = dcb->Q;
	QW = dcb->QW;
}

void LocalEdgeFlip::init()
{
	std::set<Edge> edges;
	
	// populate queue and timestamp map
	for (int i = 0; i < F.rows(); ++i)
		if (F(i, 0) != INVALID_INDEX)
			for (int j = 0; j < 3; ++j)
				edges.insert(Edge(F(i, j), F(i, (j + 1) % 3)));

	init(edges);
}

void LocalEdgeFlip::init(const std::set<Edge>& edges)
{
	for (const Edge& e : edges) {
		Flap flap = compute_flap(e, F, VF);
		if (flap.size() == 2) {
			Scalar ep = flap_planarity(flap, V, F);
			//Scalar ep = flap_cost(flap, V, F);
			//if (ep < planarity_threshold) {
				Timestamp et = 0;
				q.push(Flip(e, ep, et));
				tmap[e] = et;
			//}
		}
	}
}

bool LocalEdgeFlip::next_valid(Flap& flap)
{
	while (!q.empty()) {
		Flip f = q.top();

		auto ittime = tmap.find(f.edge);
		Assert(ittime != tmap.end());

		if (ittime->second == f.time) {
			flap = compute_flap(f.edge, F, VF);
			FlipStatus status = _test_operation(f, flap);
			if (status == FlipStatus::Feasible) {
				return true;
			}
			else {
				q.pop();
			}
		}
		else {
			q.pop();
		}
	}
	Assert(q.empty());
	return false;
}

LocalEdgeFlip::FlipStatus LocalEdgeFlip::_test_operation(const Flip& f, const Flap& flap)
{
	FlipStatus status = FlipStatus::Feasible;

	if (f.planarity < planarity_threshold) {
		// flip unless diagonal split gets worse
		if (!flip_improves_diagonal_split(flap, V, F, radians(120))) {
			if (!flip_preserves_aspect_ratio(flap, V, F, VF))
				status = FlipStatus::Fail_AspectRatio;
			status = FlipStatus::Fail_Split;
		}
	}
	else {
		//if (!flip_follows_direction_field(flap, V, F, D, DW, radians(30))) {
		//	status = FlipStatus::Fail_DirectionFieldAlignment;
		//}
		status = FlipStatus::Fail_Planarity;
	}

	if (status == Feasible && !flip_preserves_topology(flap, F, VF))
		status = FlipStatus::Fail_Topology;

	return status;
}

int LocalEdgeFlip::execute()
{
	int nflip = 0;

	while (!q.empty()) {
		Flap flap;
		bool valid = next_valid(flap);

		if (!valid)
			break;

		Flip flip = q.top();
		q.pop();

		Vector4i ring = vertex_ring(flap, F);
		Edge cross(ring(0), ring(2));

		FlipInfo info = flip_edge(flap, V, F, VF);
		Assert(info.ok);

		nflip++;

		obsolete.insert(flip.edge);
		fresh.erase(flip.edge);
		obsolete.erase(cross);
		fresh.insert(cross);

		if (!forbid_new_flips) {
			std::vector<Edge> eval;
			eval.push_back(Edge(ring(0), ring(1)));
			eval.push_back(Edge(ring(1), ring(2)));
			eval.push_back(Edge(ring(2), ring(3)));
			eval.push_back(Edge(ring(3), ring(0)));

			for (const Edge& e : eval) {
				//Assert(obsolete.find(e) == obsolete.end()); // ensure this edge was not swapped already (should not happen)
				if (fresh.find(e) == fresh.end()) { // only add edge if it is not new (i.e. avoid flipping the edge twice)
					auto itmap = tmap.find(e);
					Flap flap = compute_flap(e, F, VF);
					if (flap.size() == 2) {
						Scalar ep = flap_planarity(flap, V, F);
						Timestamp et = (itmap == tmap.end()) ? 0 : (itmap->second + 1);
						q.push(Flip(e, ep, et));
						tmap[e] = et;
					}
				}
			}
		}
	}

	return nflip;
}

