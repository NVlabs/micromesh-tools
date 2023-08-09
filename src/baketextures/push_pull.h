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

#include <space.h>
#include <utils.h>

typedef uint8_t byte;


namespace texture_filter
{
	inline uint8_t ch3(uint32_t c) { return uint8_t((c >> 24) & 0xFF); }
	inline uint8_t ch2(uint32_t c) { return uint8_t((c >> 16) & 0xFF); }
	inline uint8_t ch1(uint32_t c) { return uint8_t((c >>  8) & 0xFF); }
	inline uint8_t ch0(uint32_t c) { return uint8_t((c      ) & 0xFF); }

	inline uint32_t pack(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
		uint32_t res =  (uint32_t(c0) << 24) | (uint32_t(c1) << 16) | (uint32_t(c2) << 8) | uint32_t(c3);
		return res;
	}

	static uint8_t mix_u8(uint8_t p1, byte w1, uint8_t p2, byte w2, uint8_t p3, byte w3, uint8_t p4, byte w4)
	{
		uint32_t result =  (uint32_t(p1) * w1 + uint32_t(p2) * w2 + uint32_t(p3) * w3 + uint32_t(p4) * w4)
		    / (uint32_t(w1) + uint32_t(w2) + uint32_t(w3) +uint32_t(w4));

		return uint8_t(result & 0xFF);
	}

	static uint32_t mix_u32(uint32_t p1, byte w1, uint32_t p2, byte w2, uint32_t p3, byte w3, uint32_t p4, byte w4)
	{
		uint32_t c0 = mix_u8(ch0(p1), w1, ch0(p2), w2, ch0(p3), w3, ch0(p4), w4);
		uint32_t c1 = mix_u8(ch1(p1), w1, ch1(p2), w2, ch1(p3), w3, ch1(p4), w4);
		uint32_t c2 = mix_u8(ch2(p1), w1, ch2(p2), w2, ch2(p3), w3, ch2(p4), w4);
		uint32_t c3 = mix_u8(ch3(p1), w1, ch3(p2), w2, ch3(p3), w3, ch3(p4), w4);

		return pack(c3, c2, c1, c0);
	}

	static void pull_push_mip(MatrixXu32& p, MatrixXu32& mip, uint32_t bkcolor)
	{
		Assert(p.rows() / 2 == mip.rows());
		Assert(p.cols() / 2 == mip.cols());
		byte w1, w2, w3, w4;
		for (int y = 0; y < mip.rows(); ++y) {
			for (int x = 0; x < mip.cols(); ++x) {
				byte w1 = p(y * 2    , x * 2) == bkcolor ? 0 : 255;
				byte w2 = p(y * 2 + 1, x * 2) == bkcolor ? 0 : 255;
				byte w3 = p(y * 2    , x * 2 + 1) == bkcolor ? 0 : 255;
				byte w4 = p(y * 2 + 1, x * 2 + 1) == bkcolor ? 0 : 255;
				if (w1 + w2 + w3 + w4 > 0)
					mip(y, x) = mix_u32(p(y * 2, x * 2), w1, p(y * 2 + 1, x * 2), w2, p(y * 2, x * 2 + 1), w3, p(y * 2 + 1, x * 2 + 1), w4);
			}
		}
	}

	static void pull_push_fill(MatrixXu32& p, MatrixXu32& mip, uint32_t bkcolor)
	{
		Assert(p.rows() / 2 == mip.rows());
		Assert(p.cols() / 2 == mip.cols());
		int rows = mip.rows();
		int cols = mip.cols();
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				if (p(y * 2, x * 2) == bkcolor)
					p(y * 2, x * 2) = mix_u32(mip(y, x), 144, mip(std::max(y - 1, 0), x), 48, mip(y, std::max(x - 1, 0)), 48, mip(std::max(y - 1, 0), std::max(x - 1, 0)), 16);

				if (p(y * 2 + 1, x * 2) == bkcolor)
					p(y * 2 + 1, x * 2) = mix_u32(mip(y, x), 144, mip(std::min(y + 1, rows - 1), x), 48, mip(y, std::max(x - 1, 0)), 48, mip(std::min(y + 1, rows - 1), std::max(x - 1, 0)), 16);

				if (p(y * 2, x * 2 + 1) == bkcolor)
					p(y * 2, x * 2 + 1) = mix_u32(mip(y, x), 144, mip(std::max(y - 1, 0), x), 48, mip(y, std::min(x + 1, cols - 1)), 48, mip(std::max(y - 1, 0), std::min(x + 1, cols - 1)), 16);

				if (p(y * 2 + 1, x * 2 + 1) == bkcolor)
					p(y * 2 + 1, x * 2 + 1) = mix_u32(mip(y, x), 144, mip(std::min(y + 1, rows - 1), x), 48, mip(y, std::min(x + 1, cols - 1)), 48, mip(std::max(y - 1, 0), std::max(x - 1, 0)), 16);

			}
		}
	}

	static void PullPush( MatrixXu32 & p, uint32_t  bkcolor )
	{
		std::vector<MatrixXu32> mip;
		int div = 2;
		int miplevel = 0;

		// pull phase: create the mipmap
		while (true) {
			mip.push_back(MatrixXu32::Constant(p.rows() / div, p.cols() / div, bkcolor));
			div *= 2;

			if (miplevel > 0)
				pull_push_mip(mip[miplevel - 1], mip[miplevel], bkcolor);
			else
				pull_push_mip(p, mip[miplevel], bkcolor);

			if (mip[miplevel].cols() <= 1 || mip[miplevel].rows() <= 1)
				break;

			++miplevel;
		}

		// push phase: refill
		for(int i = miplevel; i >= 0; --i) {
			if (i > 0)
				pull_push_fill(mip[i - 1], mip[i], bkcolor);
			else
				pull_push_fill(p, mip[i], bkcolor);
		}
	}
} // texture_filter namespace

