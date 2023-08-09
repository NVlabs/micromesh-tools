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

#include "geodesic.h"
#include "utils.h"

#include <set>
#include <vector>
#include <algorithm>


struct DijkstraEntry {
	// vertex
	int vi;

	// distance
	Scalar d;

	// predecessor
	int p;

	// cluster
	int c;

	bool operator<(const DijkstraEntry& other) const { return d < other.d; }
	bool operator>(const DijkstraEntry& other) const { return d > other.d; }
};


void dijkstra(const MatrixX& V, const MatrixXi& F, const VFAdjacency& VF, const std::set<int>& sources, VectorX& dist, VectorXi& pred, VectorXi& cluster_V, VectorXi& cluster_F)
{
	Timer t;
	
	int deg = F.cols();
	Assert(deg == 3);

	std::greater<DijkstraEntry> cmp_op;

	// initialize data
	dist = VectorX::Constant(V.rows(), Infinity);
	pred = VectorXi::Constant(V.rows(), -1);
	cluster_V = VectorXi::Constant(V.rows(), -1);
	cluster_F = VectorXi::Constant(F.rows(), -1);

	std::vector<DijkstraEntry> q;
	for (int s : sources) {
		Assert(s < V.rows());
		q.push_back({ s, 0.0, s, s });
	}

	while (!q.empty()) {
		DijkstraEntry entry = q.front();
		std::pop_heap(q.begin(), q.end(), cmp_op);
		q.pop_back();

		if (entry.d < dist[entry.vi]) {
			dist[entry.vi] = entry.d;
			pred[entry.vi] = entry.p;
			cluster_V[entry.vi] = entry.c;
			
			std::set<int> visited;
			for (const VFEntry& vfe : VF[entry.vi]) {
				int fi = vfe.first;

				int vf1 = F(fi, (vfe.second + 1) % deg);
				if (visited.find(vf1) == visited.end()) {
					Scalar d1 = (V.row(entry.vi) - V.row(vf1)).norm();
					q.push_back({ vf1, entry.d + d1, entry.vi, entry.c });
					std::push_heap(q.begin(), q.end(), cmp_op);
					visited.insert(vf1);
				}

				int vf2 = F(fi, (vfe.second + 2) % deg);
				if (visited.find(vf2) == visited.end()) {
					Scalar d2 = (V.row(entry.vi) - V.row(vf2)).norm();
					q.push_back({ vf2, entry.d + d2, entry.vi, entry.c });
					std::push_heap(q.begin(), q.end(), cmp_op);
					visited.insert(vf2);
				}
			}
		}
	}

	for (int i = 0; i < F.rows(); ++i) {
		if (F(i, 0) != INVALID_INDEX) {
			RowVector3 centroid = (V.row(F(i, 0)) + V.row(F(i, 1)) + V.row(F(i, 2))) / 3.0;
			std::vector<Scalar> df;
			for (int j = 0; j < deg; ++j)
				df.push_back((centroid - V.row(F(i, j))).norm() + dist[F(i, j)]);
			int vf_nearest = std::distance(df.begin(), std::min_element(df.begin(), df.end()));
			cluster_F(i) = cluster_V(F(i, vf_nearest));
		}
	}
}

