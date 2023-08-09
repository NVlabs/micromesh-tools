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

#include "color.h"
#include "utils.h"

#include <random>


// Colormaps sampled from Matplotlib https://matplotlib.org
static std::map<ColorMap, std::vector<Vector4u8>> color_maps = {
	{
		ColorMap::Viridis,
		{
			Vector4u8(68, 1, 84, 255),
			Vector4u8(70, 12, 95, 255),
			Vector4u8(71, 24, 106, 255),
			Vector4u8(72, 34, 115, 255),
			Vector4u8(70, 45, 124, 255),
			Vector4u8(68, 55, 129, 255),
			Vector4u8(65, 65, 134, 255),
			Vector4u8(61, 74, 137, 255),
			Vector4u8(57, 84, 139, 255),
			Vector4u8(53, 92, 140, 255),
			Vector4u8(49, 100, 141, 255),
			Vector4u8(46, 108, 142, 255),
			Vector4u8(42, 117, 142, 255),
			Vector4u8(39, 124, 142, 255),
			Vector4u8(36, 132, 141, 255),
			Vector4u8(34, 139, 141, 255),
			Vector4u8(31, 148, 139, 255),
			Vector4u8(30, 155, 137, 255),
			Vector4u8(31, 163, 134, 255),
			Vector4u8(36, 170, 130, 255),
			Vector4u8(46, 178, 124, 255),
			Vector4u8(57, 185, 118, 255),
			Vector4u8(71, 192, 110, 255),
			Vector4u8(87, 198, 101, 255),
			Vector4u8(107, 205, 89, 255),
			Vector4u8(126, 210, 78, 255),
			Vector4u8(146, 215, 65, 255),
			Vector4u8(167, 219, 51, 255),
			Vector4u8(191, 223, 36, 255),
			Vector4u8(212, 225, 26, 255),
			Vector4u8(233, 228, 25, 255),
			Vector4u8(253, 231, 36, 255),
		}
	},
	{
		ColorMap::Plasma,
		{
			Vector4u8(12, 7, 134, 255),
			Vector4u8(33, 5, 143, 255),
			Vector4u8(49, 4, 150, 255),
			Vector4u8(63, 3, 156, 255),
			Vector4u8(78, 2, 161, 255),
			Vector4u8(90, 0, 165, 255),
			Vector4u8(103, 0, 167, 255),
			Vector4u8(115, 0, 168, 255),
			Vector4u8(129, 4, 167, 255),
			Vector4u8(140, 10, 164, 255),
			Vector4u8(151, 19, 160, 255),
			Vector4u8(162, 28, 154, 255),
			Vector4u8(173, 38, 146, 255),
			Vector4u8(182, 47, 139, 255),
			Vector4u8(190, 56, 131, 255),
			Vector4u8(198, 65, 124, 255),
			Vector4u8(207, 75, 116, 255),
			Vector4u8(214, 85, 109, 255),
			Vector4u8(220, 94, 102, 255),
			Vector4u8(227, 103, 95, 255),
			Vector4u8(233, 114, 87, 255),
			Vector4u8(238, 124, 80, 255),
			Vector4u8(243, 134, 73, 255),
			Vector4u8(246, 145, 66, 255),
			Vector4u8(250, 157, 58, 255),
			Vector4u8(252, 169, 52, 255),
			Vector4u8(253, 181, 45, 255),
			Vector4u8(253, 193, 40, 255),
			Vector4u8(251, 208, 36, 255),
			Vector4u8(248, 221, 36, 255),
			Vector4u8(244, 234, 38, 255),
			Vector4u8(239, 248, 33, 255),
		}
	},
	{
		ColorMap::Cividis,
		{
			Vector4u8(0, 34, 77, 255),
			Vector4u8(0, 40, 91, 255),
			Vector4u8(0, 45, 105, 255),
			Vector4u8(4, 50, 112, 255),
			Vector4u8(28, 56, 110, 255),
			Vector4u8(40, 62, 109, 255),
			Vector4u8(50, 68, 108, 255),
			Vector4u8(59, 73, 107, 255),
			Vector4u8(69, 79, 107, 255),
			Vector4u8(77, 85, 108, 255),
			Vector4u8(84, 90, 108, 255),
			Vector4u8(91, 96, 110, 255),
			Vector4u8(99, 102, 111, 255),
			Vector4u8(106, 108, 113, 255),
			Vector4u8(113, 114, 115, 255),
			Vector4u8(120, 120, 118, 255),
			Vector4u8(128, 126, 120, 255),
			Vector4u8(135, 132, 120, 255),
			Vector4u8(143, 138, 119, 255),
			Vector4u8(151, 144, 118, 255),
			Vector4u8(160, 151, 117, 255),
			Vector4u8(168, 158, 115, 255),
			Vector4u8(176, 164, 112, 255),
			Vector4u8(184, 171, 109, 255),
			Vector4u8(194, 178, 105, 255),
			Vector4u8(202, 185, 100, 255),
			Vector4u8(211, 192, 95, 255),
			Vector4u8(219, 199, 89, 255),
			Vector4u8(229, 207, 80, 255),
			Vector4u8(238, 215, 71, 255),
			Vector4u8(248, 222, 59, 255),
			Vector4u8(253, 231, 55, 255),
		}
	},
	{
		ColorMap::Turbo,
		{
			Vector4u8(48, 18, 59, 255),
			Vector4u8(57, 41, 114, 255),
			Vector4u8(64, 64, 161, 255),
			Vector4u8(68, 86, 199, 255),
			Vector4u8(70, 109, 230, 255),
			Vector4u8(70, 130, 248, 255),
			Vector4u8(64, 150, 254, 255),
			Vector4u8(52, 170, 248, 255),
			Vector4u8(37, 192, 230, 255),
			Vector4u8(26, 209, 210, 255),
			Vector4u8(24, 224, 189, 255),
			Vector4u8(34, 235, 169, 255),
			Vector4u8(59, 244, 141, 255),
			Vector4u8(89, 251, 114, 255),
			Vector4u8(120, 254, 89, 255),
			Vector4u8(149, 254, 68, 255),
			Vector4u8(174, 249, 55, 255),
			Vector4u8(195, 241, 51, 255),
			Vector4u8(214, 229, 53, 255),
			Vector4u8(231, 215, 56, 255),
			Vector4u8(244, 196, 58, 255),
			Vector4u8(251, 179, 54, 255),
			Vector4u8(254, 158, 46, 255),
			Vector4u8(252, 134, 36, 255),
			Vector4u8(246, 107, 24, 255),
			Vector4u8(237, 85, 15, 255),
			Vector4u8(226, 66, 9, 255),
			Vector4u8(212, 50, 5, 255),
			Vector4u8(192, 35, 2, 255),
			Vector4u8(172, 22, 1, 255),
			Vector4u8(148, 12, 1, 255),
			Vector4u8(122, 4, 2, 255),
		}
	},
	{
		ColorMap::RdPu,
		{
			Vector4u8(255, 247, 243, 255),
			Vector4u8(254, 241, 237, 255),
			Vector4u8(253, 235, 231, 255),
			Vector4u8(253, 229, 226, 255),
			Vector4u8(252, 223, 219, 255),
			Vector4u8(252, 216, 212, 255),
			Vector4u8(252, 209, 205, 255),
			Vector4u8(252, 202, 198, 255),
			Vector4u8(251, 194, 191, 255),
			Vector4u8(251, 184, 188, 255),
			Vector4u8(250, 175, 185, 255),
			Vector4u8(250, 165, 182, 255),
			Vector4u8(249, 153, 178, 255),
			Vector4u8(248, 139, 173, 255),
			Vector4u8(248, 125, 168, 255),
			Vector4u8(247, 111, 163, 255),
			Vector4u8(243, 96, 159, 255),
			Vector4u8(236, 83, 157, 255),
			Vector4u8(230, 70, 154, 255),
			Vector4u8(223, 57, 152, 255),
			Vector4u8(212, 42, 146, 255),
			Vector4u8(200, 30, 140, 255),
			Vector4u8(189, 17, 134, 255),
			Vector4u8(177, 4, 127, 255),
			Vector4u8(162, 1, 124, 255),
			Vector4u8(149, 1, 122, 255),
			Vector4u8(136, 1, 121, 255),
			Vector4u8(123, 1, 119, 255),
			Vector4u8(109, 0, 115, 255),
			Vector4u8(97, 0, 112, 255),
			Vector4u8(85, 0, 109, 255),
			Vector4u8(73, 0, 106, 255),
		}
	},
	{
		ColorMap::PuRd,
		{
			Vector4u8(247, 244, 249, 255),
			Vector4u8(242, 239, 246, 255),
			Vector4u8(238, 234, 243, 255),
			Vector4u8(234, 229, 241, 255),
			Vector4u8(230, 223, 238, 255),
			Vector4u8(225, 213, 232, 255),
			Vector4u8(220, 203, 227, 255),
			Vector4u8(216, 193, 222, 255),
			Vector4u8(211, 182, 216, 255),
			Vector4u8(208, 173, 211, 255),
			Vector4u8(205, 163, 207, 255),
			Vector4u8(202, 154, 202, 255),
			Vector4u8(203, 143, 196, 255),
			Vector4u8(208, 131, 190, 255),
			Vector4u8(214, 119, 185, 255),
			Vector4u8(219, 107, 179, 255),
			Vector4u8(224, 92, 170, 255),
			Vector4u8(226, 77, 161, 255),
			Vector4u8(228, 62, 151, 255),
			Vector4u8(230, 47, 142, 255),
			Vector4u8(226, 36, 128, 255),
			Vector4u8(220, 31, 115, 255),
			Vector4u8(214, 25, 102, 255),
			Vector4u8(207, 19, 89, 255),
			Vector4u8(194, 14, 81, 255),
			Vector4u8(181, 9, 77, 255),
			Vector4u8(167, 5, 72, 255),
			Vector4u8(153, 0, 67, 255),
			Vector4u8(139, 0, 58, 255),
			Vector4u8(127, 0, 49, 255),
			Vector4u8(115, 0, 40, 255),
			Vector4u8(103, 0, 31, 255),
		}
	},
	{
		ColorMap::GnBu,
		{
			Vector4u8(247, 252, 240, 255),
			Vector4u8(241, 249, 234, 255),
			Vector4u8(235, 247, 229, 255),
			Vector4u8(229, 245, 224, 255),
			Vector4u8(223, 242, 218, 255),
			Vector4u8(218, 240, 212, 255),
			Vector4u8(213, 238, 207, 255),
			Vector4u8(208, 236, 201, 255),
			Vector4u8(201, 234, 195, 255),
			Vector4u8(192, 230, 191, 255),
			Vector4u8(183, 226, 187, 255),
			Vector4u8(174, 223, 183, 255),
			Vector4u8(163, 219, 182, 255),
			Vector4u8(151, 214, 186, 255),
			Vector4u8(140, 210, 190, 255),
			Vector4u8(129, 206, 193, 255),
			Vector4u8(116, 200, 198, 255),
			Vector4u8(105, 194, 201, 255),
			Vector4u8(94, 187, 205, 255),
			Vector4u8(82, 181, 209, 255),
			Vector4u8(71, 172, 207, 255),
			Vector4u8(63, 162, 202, 255),
			Vector4u8(54, 152, 196, 255),
			Vector4u8(45, 142, 191, 255),
			Vector4u8(35, 132, 186, 255),
			Vector4u8(26, 123, 181, 255),
			Vector4u8(18, 114, 177, 255),
			Vector4u8(9, 105, 172, 255),
			Vector4u8(8, 94, 161, 255),
			Vector4u8(8, 84, 150, 255),
			Vector4u8(8, 74, 139, 255),
			Vector4u8(8, 64, 129, 255),
		}
	},
	{
		ColorMap::PuBu,
		{
			Vector4u8(255, 247, 251, 255),
			Vector4u8(250, 242, 248, 255),
			Vector4u8(245, 238, 246, 255),
			Vector4u8(240, 234, 244, 255),
			Vector4u8(235, 230, 241, 255),
			Vector4u8(227, 224, 238, 255),
			Vector4u8(220, 219, 235, 255),
			Vector4u8(213, 213, 232, 255),
			Vector4u8(205, 207, 229, 255),
			Vector4u8(194, 202, 226, 255),
			Vector4u8(183, 197, 223, 255),
			Vector4u8(173, 192, 220, 255),
			Vector4u8(160, 186, 217, 255),
			Vector4u8(148, 181, 214, 255),
			Vector4u8(135, 176, 211, 255),
			Vector4u8(123, 171, 208, 255),
			Vector4u8(107, 165, 204, 255),
			Vector4u8(91, 159, 201, 255),
			Vector4u8(76, 152, 197, 255),
			Vector4u8(60, 146, 193, 255),
			Vector4u8(45, 138, 189, 255),
			Vector4u8(33, 130, 185, 255),
			Vector4u8(20, 122, 181, 255),
			Vector4u8(8, 114, 177, 255),
			Vector4u8(4, 107, 168, 255),
			Vector4u8(4, 101, 159, 255),
			Vector4u8(4, 96, 151, 255),
			Vector4u8(4, 90, 142, 255),
			Vector4u8(3, 81, 127, 255),
			Vector4u8(3, 73, 114, 255),
			Vector4u8(2, 64, 101, 255),
			Vector4u8(2, 56, 88, 255),
		}
	},
	{
		ColorMap::YlGnBu,
		{
			Vector4u8(255, 255, 217, 255),
			Vector4u8(250, 253, 206, 255),
			Vector4u8(245, 251, 196, 255),
			Vector4u8(241, 249, 186, 255),
			Vector4u8(235, 247, 177, 255),
			Vector4u8(226, 243, 177, 255),
			Vector4u8(216, 239, 178, 255),
			Vector4u8(207, 236, 179, 255),
			Vector4u8(193, 231, 180, 255),
			Vector4u8(175, 223, 182, 255),
			Vector4u8(157, 216, 184, 255),
			Vector4u8(139, 209, 185, 255),
			Vector4u8(120, 202, 187, 255),
			Vector4u8(104, 196, 190, 255),
			Vector4u8(89, 191, 192, 255),
			Vector4u8(73, 185, 194, 255),
			Vector4u8(59, 176, 195, 255),
			Vector4u8(50, 167, 194, 255),
			Vector4u8(41, 158, 193, 255),
			Vector4u8(32, 148, 192, 255),
			Vector4u8(29, 136, 187, 255),
			Vector4u8(31, 123, 181, 255),
			Vector4u8(32, 110, 175, 255),
			Vector4u8(33, 97, 169, 255),
			Vector4u8(34, 85, 163, 255),
			Vector4u8(35, 74, 158, 255),
			Vector4u8(36, 64, 153, 255),
			Vector4u8(36, 53, 148, 255),
			Vector4u8(29, 46, 133, 255),
			Vector4u8(22, 40, 118, 255),
			Vector4u8(15, 34, 103, 255),
			Vector4u8(8, 29, 88, 255),
		}
	},
	{
		ColorMap::GistGray,
		{
			Vector4u8(0, 0, 0, 255),
			Vector4u8(8, 8, 8, 255),
			Vector4u8(16, 16, 16, 255),
			Vector4u8(24, 24, 24, 255),
			Vector4u8(32, 32, 32, 255),
			Vector4u8(40, 40, 40, 255),
			Vector4u8(48, 48, 48, 255),
			Vector4u8(56, 56, 56, 255),
			Vector4u8(65, 65, 65, 255),
			Vector4u8(73, 73, 73, 255),
			Vector4u8(81, 81, 81, 255),
			Vector4u8(89, 89, 89, 255),
			Vector4u8(99, 99, 99, 255),
			Vector4u8(107, 107, 107, 255),
			Vector4u8(115, 115, 115, 255),
			Vector4u8(123, 123, 123, 255),
			Vector4u8(131, 131, 131, 255),
			Vector4u8(140, 140, 140, 255),
			Vector4u8(147, 147, 147, 255),
			Vector4u8(156, 156, 156, 255),
			Vector4u8(165, 165, 165, 255),
			Vector4u8(173, 173, 173, 255),
			Vector4u8(181, 181, 181, 255),
			Vector4u8(189, 189, 189, 255),
			Vector4u8(198, 198, 198, 255),
			Vector4u8(206, 206, 206, 255),
			Vector4u8(214, 214, 214, 255),
			Vector4u8(222, 222, 222, 255),
			Vector4u8(231, 231, 231, 255),
			Vector4u8(239, 239, 239, 255),
			Vector4u8(247, 247, 247, 255),
			Vector4u8(255, 255, 255, 255),
		}
	},
	{
		ColorMap::Hot,
		{
			Vector4u8(10, 0, 0, 255),
			Vector4u8(31, 0, 0, 255),
			Vector4u8(52, 0, 0, 255),
			Vector4u8(73, 0, 0, 255),
			Vector4u8(97, 0, 0, 255),
			Vector4u8(118, 0, 0, 255),
			Vector4u8(139, 0, 0, 255),
			Vector4u8(160, 0, 0, 255),
			Vector4u8(183, 0, 0, 255),
			Vector4u8(204, 0, 0, 255),
			Vector4u8(225, 0, 0, 255),
			Vector4u8(246, 0, 0, 255),
			Vector4u8(255, 15, 0, 255),
			Vector4u8(255, 36, 0, 255),
			Vector4u8(255, 57, 0, 255),
			Vector4u8(255, 78, 0, 255),
			Vector4u8(255, 102, 0, 255),
			Vector4u8(255, 123, 0, 255),
			Vector4u8(255, 144, 0, 255),
			Vector4u8(255, 165, 0, 255),
			Vector4u8(255, 188, 0, 255),
			Vector4u8(255, 209, 0, 255),
			Vector4u8(255, 230, 0, 255),
			Vector4u8(255, 251, 0, 255),
			Vector4u8(255, 255, 30, 255),
			Vector4u8(255, 255, 62, 255),
			Vector4u8(255, 255, 93, 255),
			Vector4u8(255, 255, 125, 255),
			Vector4u8(255, 255, 160, 255),
			Vector4u8(255, 255, 191, 255),
			Vector4u8(255, 255, 223, 255),
			Vector4u8(255, 255, 255, 255),
		}
	},
	{
		ColorMap::GistHeat,
		{
			Vector4u8(0, 0, 0, 255),
			Vector4u8(12, 0, 0, 255),
			Vector4u8(24, 0, 0, 255),
			Vector4u8(36, 0, 0, 255),
			Vector4u8(49, 0, 0, 255),
			Vector4u8(61, 0, 0, 255),
			Vector4u8(73, 0, 0, 255),
			Vector4u8(85, 0, 0, 255),
			Vector4u8(98, 0, 0, 255),
			Vector4u8(110, 0, 0, 255),
			Vector4u8(122, 0, 0, 255),
			Vector4u8(134, 0, 0, 255),
			Vector4u8(148, 0, 0, 255),
			Vector4u8(160, 0, 0, 255),
			Vector4u8(172, 0, 0, 255),
			Vector4u8(184, 0, 0, 255),
			Vector4u8(197, 8, 0, 255),
			Vector4u8(210, 25, 0, 255),
			Vector4u8(221, 40, 0, 255),
			Vector4u8(234, 57, 0, 255),
			Vector4u8(247, 75, 0, 255),
			Vector4u8(255, 91, 0, 255),
			Vector4u8(255, 107, 0, 255),
			Vector4u8(255, 123, 0, 255),
			Vector4u8(255, 141, 27, 255),
			Vector4u8(255, 157, 59, 255),
			Vector4u8(255, 173, 91, 255),
			Vector4u8(255, 189, 123, 255),
			Vector4u8(255, 207, 159, 255),
			Vector4u8(255, 223, 191, 255),
			Vector4u8(255, 239, 223, 255),
			Vector4u8(255, 255, 255, 255),
		}
	},
	{
		ColorMap::Copper,
		{
			Vector4u8(0, 0, 0, 255),
			Vector4u8(9, 6, 3, 255),
			Vector4u8(19, 12, 7, 255),
			Vector4u8(29, 18, 11, 255),
			Vector4u8(40, 25, 16, 255),
			Vector4u8(50, 32, 20, 255),
			Vector4u8(60, 38, 24, 255),
			Vector4u8(70, 44, 28, 255),
			Vector4u8(81, 51, 32, 255),
			Vector4u8(91, 57, 36, 255),
			Vector4u8(101, 64, 40, 255),
			Vector4u8(111, 70, 44, 255),
			Vector4u8(122, 77, 49, 255),
			Vector4u8(132, 83, 53, 255),
			Vector4u8(142, 89, 57, 255),
			Vector4u8(151, 96, 61, 255),
			Vector4u8(163, 103, 65, 255),
			Vector4u8(172, 109, 69, 255),
			Vector4u8(182, 115, 73, 255),
			Vector4u8(192, 121, 77, 255),
			Vector4u8(203, 128, 82, 255),
			Vector4u8(213, 135, 86, 255),
			Vector4u8(223, 141, 90, 255),
			Vector4u8(233, 147, 94, 255),
			Vector4u8(244, 154, 98, 255),
			Vector4u8(254, 160, 102, 255),
			Vector4u8(255, 167, 106, 255),
			Vector4u8(255, 173, 110, 255),
			Vector4u8(255, 180, 114, 255),
			Vector4u8(255, 186, 118, 255),
			Vector4u8(255, 192, 122, 255),
			Vector4u8(255, 199, 126, 255),
		}
	},
};

static inline Vector4u8 lerp(const Vector4u8 c0, const Vector4u8 c1, Scalar a)
{
	return Vector4u8(
		uint8_t(c0(0) * (1 - a) + c1(0) * a),
		uint8_t(c0(1) * (1 - a) + c1(1) * a),
		uint8_t(c0(2) * (1 - a) + c1(2) * a),
		uint8_t(c0(3) * (1 - a) + c1(3) * a)
	);
}

Vector4u8 get_color_mapping(Scalar v, Scalar minv, Scalar maxv, ColorMap cmap)
{
	int sz = int(color_maps[cmap].size());

	if (v > maxv)
		v = maxv;
	if (v < minv)
		v = minv;

	v = (v - minv) / (maxv - minv);

	Scalar v0 = v * sz;
	int n0 = int(v0);

	if (n0 < 0)
		return color_maps[cmap].front();
	if (n0 >= sz - 1)
		return color_maps[cmap].back();

	int n1 = n0 + 1;

	Scalar fract = v0 - n0;

	Vector4u8 c0 = color_maps[cmap][n0];
	Vector4u8 c1 = color_maps[cmap][n1];

	return lerp(c0, c1, fract);
}

static Vector3f hsv_color(Scalar h)
{
	const float saturation = 1;
	const float value = 1;

	float hprime = h / 60.0f;
	float c = saturation * value;
	float x = c * (1 - std::abs(std::fmod(hprime, 2.0) - 1));

	float r, g, b;
	Assert(hprime >= 0);
	Assert(hprime <= 6);
	
	if (hprime < 1) {
		r = c; g = x; b = 0;
	}
	else if (hprime < 2) {
		r = x; g = c; b = 0;
	}
	else if (hprime < 3) {
		r = 0; g = c; b = x;
	}
	else if (hprime < 4) {
		r = 0; g = x; b = c;
	}
	else if (hprime < 5) {
		r = x; g = 0; b = c;
	}
	else {
		r = c; g = 0; b = x;
	}
	
	r += value - saturation;
	g += value - saturation;
	b += value - saturation;

	return Vector3f(r, g, b);

}

MatrixXf scatter_colors(int n)
{
	MatrixXf C(n, 3);

	for (int i = 0; i < n; ++i) {
		C.row(i) = hsv_color(i * (360 / Scalar(n)));
	}

	return C;
}

