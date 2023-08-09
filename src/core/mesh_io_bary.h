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

#include "micro.h"

#include <baryutils/baryutils.h>

#define UNORM11_MAX ((1 << 11) - 1)

struct Packed32 {
	uint32_t data = 0;
	int n_bits = 32;

	// returns the number of bits that could not be packed
	int pack(uint32_t val, int bits_to_pack, uint32_t *remainder)
	{
		int left_bits = std::max(bits_to_pack - n_bits, 0); // bits that cannot be packed here
		// shift to align least significant digits if bits are available
		if (n_bits)
			data |= (val << (32 - n_bits));
		// return most significant digits not packed
		*remainder = (val >> (bits_to_pack - left_bits));
		n_bits = std::max(n_bits - bits_to_pack, 0);
		return left_bits;
	}

	void reset()
	{
		data = 0;
		n_bits = 32;
	}
};

struct ReadPacked32 {
	uint32_t data;
	int n_bits;

	ReadPacked32(uint32_t bits)
	{
		reset(bits);
	}

	int unpack(int bits_to_unpack, uint32_t* val)
	{
		int unread_bits = std::max(bits_to_unpack - n_bits, 0);
		if (n_bits)
			*val = (data >> (32 - n_bits)) & ((1 << bits_to_unpack) - 1);
		else
			*val = 0;
		n_bits = std::max(n_bits - bits_to_unpack, 0);
		return unread_bits;
	}

	void reset(uint32_t bits)
	{
		data = bits;
		n_bits = 32;
	}
};

struct Bary {
	baryutils::BaryBasicData data;
	Scalar min_displacement = 0;
	Scalar max_displacement = 0;
};

// Fills in the Bary.data displacement data, and reports min and max displacements
// *before* normalization, even if normalize_displacements is true
// if normalize_displacements, then displacements are remapped in [0, 1]
void extract_displacement_bary_data(const SubdivisionMesh& umesh, Bary* bary, bool normalize_displacements);

