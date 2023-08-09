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

#include "mesh_io_bary.h"

static_assert(sizeof(float) == 4);

void extract_displacement_bary_data(const SubdivisionMesh& umesh, Bary* bary, bool normalize_displacements)
{
	Assert(umesh.base_fn >= 0);
	const unsigned ntris = umesh.base_fn;

	//std::vector<float> base_minmaxs;
	std::vector<uint16_t> base_minmaxs;
	//std::vector<float> displacements;
	std::vector<uint32_t> displacements;

	baryutils::BaryBasicData* bary_data = &(bary->data);

	bary_data->minSubdivLevel = std::numeric_limits<uint32_t>::max();
	bary_data->maxSubdivLevel = 0;

	bary_data->groups = {};
	bary_data->valuesInfo = {
		//.valueFormat = bary::Format::eR32_sfloat,
		//.valueFormat = bary::Format::eR11_unorm_pack16,
		.valueFormat = bary::Format::eR11_unorm_packed_align32,
		.valueLayout = bary::ValueLayout::eTriangleUmajor,
		.valueFrequency = bary::ValueFrequency::ePerVertex,
		.valueCount = 0,
		.valueByteSize = 1,
		.valueByteAlignment = 4
	};
	bary_data->values = {};
	bary_data->triangles = {};

	bary_data->histogramEntries = {};
	bary_data->groupHistogramRanges = {};

	bary_data->triangleMinMaxsInfo = {
		//.elementFormat = bary::Format::eR32_sfloat,
		.elementFormat = bary::Format::eR11_unorm_pack16,
		.elementCount = 2 * ntris,
		.elementByteSize = 2,
		.elementByteAlignment = 4
	};
	bary_data->triangleMinMaxs = {};

	bary::Group group = {
		.triangleFirst = 0,
		.triangleCount = umesh.base_fn >= 0 ? (uint32_t)umesh.base_fn : 0,
		.valueFirst = 0,
		.valueCount = 0,
		.minSubdivLevel = 0,
		.maxSubdivLevel = 0,
		.floatBias = {0, 0, 0, 0},
		.floatScale = {1, 1, 1, 1}
	};

	float d_g_min = std::numeric_limits<Scalar>::max();
	float d_g_max = std::numeric_limits<Scalar>::lowest();

	for (const SubdivisionTri& st : umesh.faces) {
		for (int uvi = 0; uvi < st.V.rows(); ++uvi) {
			if (st.ref(uvi)) {
				// compute displacement value
				float d = st.scalar_displacement(uvi);
				d_g_min = std::min(d, d_g_min);
				d_g_max = std::max(d, d_g_max);
			}
		}
	}

	bary->min_displacement = Scalar(d_g_min);
	bary->max_displacement = Scalar(d_g_max);

	bary_data->triangles.reserve(ntris);
	base_minmaxs.reserve(2 * ntris);
	displacements.reserve(umesh.micro_fn / 2);

	for (const SubdivisionTri& st : umesh.faces) {
		bary::Triangle tri = {
			.valuesOffset = (uint32_t)displacements.size() * sizeof(uint32_t),
			.subdivLevel = st.subdivision_level(),
			.blockFormat = 0,
		};

		bary_data->minSubdivLevel = std::min(bary_data->minSubdivLevel, (uint32_t)tri.subdivLevel);
		bary_data->maxSubdivLevel = std::max(bary_data->maxSubdivLevel, (uint32_t)tri.subdivLevel);

		float d_min = std::numeric_limits<Scalar>::max();
		float d_max = std::numeric_limits<Scalar>::lowest();

		// here we need to follow the specified microvertex order
		// which is completely different from the one used by BaricentricGrid...
		Packed32 p;
		BarycentricGrid grid(st.subdivision_level());
		for (int k = 0; k < grid.samples_per_side(); ++k) {
			for (int h = 0; h < grid.samples_per_side() - k; ++h) {
				int i = k + h;
				int j = h;
				// compute displacement value
				int uvi = grid.index(i, j);
				uint32_t dq = 0;
				if (st.ref(uvi)) {
					float d = st.scalar_displacement(grid.index(i, j));
					if (normalize_displacements)
						d = ((d - d_g_min) / (d_g_max - d_g_min));
					dq = uint16_t(d * UNORM11_MAX);
					d_min = std::min(d, d_min);
					d_max = std::max(d, d_max);
				}
				else {
					// microvertex is not referenced because of edge decimation flags
					// its displacement is meaningless
				}
				uint32_t remainder;
				int left_bits = p.pack(dq, 11, &remainder);
				if (left_bits) {
					displacements.push_back(p.data);
					p.reset();
					dq >>= 11 - left_bits;
					Assert(p.pack(dq, left_bits, &remainder) == 0);
				}
			}
		}
		displacements.push_back(p.data); // push final 32-bit chunk

		//bary_data->valuesInfo.valueCount += grid.num_samples();
		//group.valueCount += grid.num_samples();

		bary_data->triangles.push_back(tri);

		base_minmaxs.push_back(uint16_t(d_min * UNORM11_MAX));
		base_minmaxs.push_back(uint16_t(d_max * UNORM11_MAX));
	}

	bary_data->valuesInfo.valueCount = displacements.size() * sizeof(uint32_t);
	group.valueCount = displacements.size() * sizeof(uint32_t);

	group.minSubdivLevel = bary_data->minSubdivLevel;
	group.maxSubdivLevel = bary_data->maxSubdivLevel;
	bary_data->groups.push_back(group);

	//std::size_t minmaxs_size = (((base_minmaxs.size() + 1) * sizeof(float) - 1)) / sizeof(decltype(bary_data->triangleMinMaxs[0]));
	std::size_t minmaxs_size = (((base_minmaxs.size() + 1) * sizeof(uint16_t) - 1)) / sizeof(decltype(bary_data->triangleMinMaxs[0]));
	bary_data->triangleMinMaxs.resize(minmaxs_size);
	//std::memcpy(bary_data->triangleMinMaxs.data(), base_minmaxs.data(), base_minmaxs.size() * sizeof(float));
	std::memcpy(bary_data->triangleMinMaxs.data(), base_minmaxs.data(), base_minmaxs.size() * sizeof(uint16_t));

	//std::size_t displacements_size = (((displacements.size() + 1) * sizeof(float) + 1)) / sizeof(decltype(bary_data->values[0]));
	std::size_t displacements_size = (((displacements.size() + 1) * sizeof(uint32_t) + 1)) / sizeof(decltype(bary_data->values[0]));
	bary_data->values.resize(displacements_size);
	//std::memcpy(bary_data->values.data(), displacements.data(), displacements.size() * sizeof(float));
	std::memcpy(bary_data->values.data(), displacements.data(), displacements.size() * sizeof(uint32_t));

	Assert(bary_data->valuesInfo.valueLayout == bary::ValueLayout::eTriangleUmajor);
}

#if 0
void extract_displacement_bary_data(const SubdivisionMesh& umesh, Bary* bary, bool normalize_displacements)
{
	Assert(umesh.base_fn >= 0);
	const unsigned ntris = umesh.base_fn;

	//std::vector<float> base_minmaxs;
	std::vector<uint16_t> base_minmaxs;
	//std::vector<float> displacements;
	std::vector<uint16_t> displacements;

	baryutils::BaryBasicData* bary_data = &(bary->data);

	bary_data->minSubdivLevel = std::numeric_limits<uint32_t>::max();
	bary_data->maxSubdivLevel = 0;

	bary_data->groups = {};
	bary_data->valuesInfo = {
		//.valueFormat = bary::Format::eR32_sfloat,
		.valueFormat = bary::Format::eR11_unorm_pack16,
		.valueLayout = bary::ValueLayout::eTriangleUmajor,
		.valueFrequency = bary::ValueFrequency::ePerVertex,
		.valueCount = 0,
		.valueByteSize = 2,
		.valueByteAlignment = 4
	};
	bary_data->values = {};
	bary_data->triangles = {};

	bary_data->histogramEntries = {};
	bary_data->groupHistogramRanges = {};

	bary_data->triangleMinMaxsInfo = {
		//.elementFormat = bary::Format::eR32_sfloat,
		.elementFormat = bary::Format::eR11_unorm_pack16,
		.elementCount = 2 * ntris,
		.elementByteSize = 2,
		.elementByteAlignment = 4
	};
	bary_data->triangleMinMaxs = {};

	bary::Group group = {
		.triangleFirst = 0,
		.triangleCount = umesh.base_fn >= 0 ? (uint32_t)umesh.base_fn : 0,
		.valueFirst = 0,
		.valueCount = 0,
		.minSubdivLevel = 0,
		.maxSubdivLevel = 0,
		.floatBias = {0, 0, 0, 0},
		.floatScale = {1, 1, 1, 1}
	};

	float d_g_min = std::numeric_limits<Scalar>::max();
	float d_g_max = std::numeric_limits<Scalar>::lowest();

	for (const SubdivisionTri& st : umesh.faces) {
		for (int uvi = 0; uvi < st.V.rows(); ++uvi) {
			if (st.ref(uvi)) {
				// compute displacement value
				float d = st.scalar_displacement(uvi);
				d_g_min = std::min(d, d_g_min);
				d_g_max = std::max(d, d_g_max);
			}
		}
	}

	bary->min_displacement = Scalar(d_g_min);
	bary->max_displacement = Scalar(d_g_max);

	bary_data->triangles.reserve(ntris);
	base_minmaxs.reserve(2 * ntris);
	displacements.reserve(umesh.micro_fn / 2);

	for (const SubdivisionTri& st : umesh.faces) {
		bary::Triangle tri = {
			.valuesOffset = (uint32_t)displacements.size(),
			.subdivLevel = st.subdivision_level(),
			.blockFormat = 0,
		};

		bary_data->minSubdivLevel = std::min(bary_data->minSubdivLevel, (uint32_t)tri.subdivLevel);
		bary_data->maxSubdivLevel = std::max(bary_data->maxSubdivLevel, (uint32_t)tri.subdivLevel);

		float d_min = std::numeric_limits<Scalar>::max();
		float d_max = std::numeric_limits<Scalar>::lowest();

		// here we need to follow the specified microvertex order
		// which is completely different from the one used by BaricentricGrid...
		BarycentricGrid grid(st.subdivision_level());
		for (int k = 0; k < grid.samples_per_side(); ++k) {
			for (int h = 0; h < grid.samples_per_side() - k; ++h) {
				int i = k + h;
				int j = h;
				// compute displacement value
				int uvi = grid.index(i, j);
				if (st.ref(uvi)) {
					float d = st.scalar_displacement(grid.index(i, j));
					if (normalize_displacements)
						d = ((d - d_g_min) / (d_g_max - d_g_min));
					uint16_t dq = uint16_t(d * UNORM11_MAX);
					displacements.push_back(dq);
					d_min = std::min(d, d_min);
					d_max = std::max(d, d_max);
				}
				else {
					// microvertex is not referenced because of edge decimation flags
					// its displacement is meaningless
					displacements.push_back(0);
				}
			}
		}

		bary_data->valuesInfo.valueCount += grid.num_samples();
		group.valueCount += grid.num_samples();

		bary_data->triangles.push_back(tri);

		base_minmaxs.push_back(uint16_t(d_min * UNORM11_MAX));
		base_minmaxs.push_back(uint16_t(d_max * UNORM11_MAX));
	}

	group.minSubdivLevel = bary_data->minSubdivLevel;
	group.maxSubdivLevel = bary_data->maxSubdivLevel;
	bary_data->groups.push_back(group);

	//std::size_t minmaxs_size = (((base_minmaxs.size() + 1) * sizeof(float) - 1)) / sizeof(decltype(bary_data->triangleMinMaxs[0]));
	std::size_t minmaxs_size = (((base_minmaxs.size() + 1) * sizeof(uint16_t) - 1)) / sizeof(decltype(bary_data->triangleMinMaxs[0]));
	bary_data->triangleMinMaxs.resize(minmaxs_size);
	//std::memcpy(bary_data->triangleMinMaxs.data(), base_minmaxs.data(), base_minmaxs.size() * sizeof(float));
	std::memcpy(bary_data->triangleMinMaxs.data(), base_minmaxs.data(), base_minmaxs.size() * sizeof(uint16_t));

	//std::size_t displacements_size = (((displacements.size() + 1) * sizeof(float) + 1)) / sizeof(decltype(bary_data->values[0]));
	std::size_t displacements_size = (((displacements.size() + 1) * sizeof(uint16_t) + 1)) / sizeof(decltype(bary_data->values[0]));
	bary_data->values.resize(displacements_size);
	//std::memcpy(bary_data->values.data(), displacements.data(), displacements.size() * sizeof(float));
	std::memcpy(bary_data->values.data(), displacements.data(), displacements.size() * sizeof(uint16_t));

	Assert(bary_data->valuesInfo.valueLayout == bary::ValueLayout::eTriangleUmajor);
}
#endif


