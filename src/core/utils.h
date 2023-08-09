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

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <string>

#define Assert(expr) \
    ((expr) \
     ? (void) (0) \
     : Assert_fail(#expr, __FILE__, __LINE__))

[[ noreturn ]]
inline void Assert_fail(const char *expr, const char *filename, unsigned int line)
{
    std::cerr << filename << " (line " << line << "): Failed check `" << expr << "'" << std::endl;
    std::abort();
}

struct Timer {
    typedef std::chrono::high_resolution_clock hrc;

    hrc::time_point start;
    hrc::time_point last;

    Timer() : start(hrc::now()) { last = start; }

    double time_elapsed() {
        last = hrc::now();
        return std::chrono::duration<double>(last - start).count();
    }

    double time_since_last_check() {
        hrc::time_point t = last;
        last = hrc::now();
        return std::chrono::duration<double>(last - t).count();
	}

    void reset() {
        start = last = hrc::now();
    }
};

inline bool whitespace(char c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline std::string rtrim(const std::string& s)
{
	auto r = s.crbegin();
	
	while (r != s.rend())
		if (whitespace(*r))
			r++;
		else
			break;

	return s.substr(0, s.rend() - r);
}

inline std::string lowercase(std::string s)
{
    for (char& c : s)
        c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

