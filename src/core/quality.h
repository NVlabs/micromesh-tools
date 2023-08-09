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

#include <vector>

class Distribution {
	std::vector<Scalar> samples;
	bool dirty;

	Scalar sum;
	Scalar sum_squares;
	Scalar min_val;
	Scalar max_val;

	void update()
	{
		if (dirty) {
			std::sort(samples.begin(), samples.end());
			sum = 0;
			sum_squares = 0;
			for (auto& s : samples) {
				sum += s;
				sum_squares += s * s;
			}
			dirty = false;
		}
	}

public:

	Distribution()
		: dirty(true),
		sum(0),
		sum_squares(0),
		min_val(std::numeric_limits<Scalar>::max()),
		max_val(std::numeric_limits<Scalar>::lowest())
	{
	}

	Scalar avg() { update(); return sum / samples.size(); }

	Scalar var() { update(); return (sum_squares / samples.size()) - avg(); }

	Scalar std_dev() { update(); return std::sqrt(var()); }

	Scalar max() { return max_val; }

	Scalar min() { return min_val; }

	Scalar percentile(Scalar perc) {
		update();
		perc = clamp(perc, Scalar(0), Scalar(1));
		std::size_t i = (samples.size() - 1) * perc;
		return samples[i];
	}

	void add(Scalar sample) {
		dirty = true;
		if (sample < min_val)
			min_val = sample;
		if (sample > max_val)
			max_val = sample;
		samples.push_back(sample);
	}
};

struct Histogram {
	Scalar _range_min;
	Scalar _range_max;
	int _n_bins;

	std::vector<Scalar> _W;

	Scalar _sum_weighted_vals;
	Scalar _sum_weights;

	Scalar _min_val;
	Scalar _max_val;

	Histogram(Scalar range_min, Scalar range_max, int n_bins)
	{
		reset(range_min, range_max, n_bins);
	}

	void reset(Scalar range_min, Scalar range_max, int n_bins)
	{
		_range_min = range_min;
		_range_max = range_max;

		_W.clear();
		_W.resize(n_bins + 2);

		_n_bins = n_bins;

		_sum_weighted_vals = 0;
		_sum_weights = 0;

		_min_val = std::numeric_limits<Scalar>::max();
		_max_val = std::numeric_limits<Scalar>::lowest();
	}

	void add(Scalar val, Scalar weight)
	{
		_sum_weighted_vals += val * weight;
		_sum_weights += weight;
		int i = _find_index(val);
		_W[i] += weight;

		_min_val = std::min(_min_val, val);
		_max_val = std::max(_max_val, val);
	}

	int _find_index(Scalar val) const
	{
		if (val < _range_min)
			return 0;
		else if (val >= _range_max)
			return _n_bins + 1;
		else
			return ((val - _range_min) / (_range_max - _range_min)) * _n_bins + 1;
	}

	Scalar _index_value(int i) const
	{
		if (i <= 1)
			return _range_min;
		else if (i >= _n_bins)
			return _range_max;
		else
			return _range_min + (i / Scalar(_n_bins)) * (_range_max - _range_min);
	}

	Scalar min() const { return _min_val; }
	Scalar max() const { return _max_val; }

	Scalar avg() const { return _sum_weighted_vals / _sum_weights; }

	Scalar percentile(Scalar p) const
	{
		p = clamp<Scalar>(p, 0, 1);

		Scalar pw = _sum_weights * p;
		Scalar w_sum = 0;
	
		int i = 0;
		for (; i < _W.size(); ++i)
			if (w_sum + _W[i] >= pw)
				break;

		return _index_value(i);
	}
};

void compute_face_quality_aspect_ratio(const MatrixX& V, const MatrixXi& F, VectorX& FQ);

void compute_face_quality_stretch(const MatrixX& V1, const MatrixX& V2, const MatrixXi& F, VectorX& FQ);

void compute_vertex_quality_hausdorff_distance(const MatrixX& from_V, const MatrixX& from_VN, const MatrixXi& from_F, const MatrixX& to_V, const MatrixX& to_VN, const MatrixXi& to_F, VectorX& VQ);

