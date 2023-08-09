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

#include "push_pull.h"

#include <mesh_io.h>
#include <mesh_io_gltf.h>
#include <mesh_io_bary.h>
#include <utils.h>
#include <tangent.h>
#include <clean.h>
#include <bvh.h>

#include <iostream>
#include <filesystem>

#include <algorithm>
#include <execution>

#include "stb_image.h"
#include "stb_image_write.h"


namespace fs = std::filesystem;

static inline Vector2 floor(const Vector2& v)
{
	return Vector2(floor(v.x()), floor(v.y()));
}

static inline Vector2 fract(const Vector2& v)
{
	return v - floor(v);
}

static inline Vector4u8 mix(const Vector4u8& x, const Vector4u8& y, Scalar a)
{
	return (x.cast<Scalar>() * (1 - a) + y.cast<Scalar>() * a).cast<uint8_t>();
}

struct TextureImage {
	int _w;
	int _h;
	uint8_t* _data;
	Vector4u8* _cdata;

	const int CHANNELS = 4;

	TextureImage(const std::string& path)
	: _w(0), _h(0), _data(nullptr), _cdata(nullptr)
	{
		int n;
		_data = stbi_load(path.c_str(), &_w, &_h, &n, CHANNELS);
		Assert(_data && "Error reading texture image");
		
		_cdata = reinterpret_cast<Vector4u8*>(_data);
	}

	~TextureImage()
	{
		stbi_image_free(_data);
	}

	Vector4u8 sample(Vector2 p) const
	{
		p.x() *= _w;
		p.y() *= _h;

		p -= Vector2(0.5, 0.5);

		Vector2 p0 = floor(p);
		Vector2 p1 = p0 + Vector2(1, 1);
		Vector2 w = fract(p);

		return mix(
			mix(sample(int(p0.x()), int(p0.y())), sample(int(p1.x()), int(p0.y())), w.x()),
			mix(sample(int(p0.x()), int(p1.y())), sample(int(p1.x()), int(p1.y())), w.x()),
			w.y()
		);
	}

	Vector4u8 sample(int x, int y) const
	{
		x = (x + _w) % _w;
		y = (y + _h) % _h;
		return _cdata[y * _w + x];
	}
};

struct RasterizationSample {
	// image coordinates of the sample
	Vector2i coord;
	// barycentric weights of the sample
	Vector3 w;
};

inline Scalar determinant(Vector2 a, Vector2 b)
{
	return a(0) * b(1) - a(1) * b(0);
}

uint8_t quantize_ldr_value(Scalar v)
{
	return uint8_t(clamp(v, Scalar(0), Scalar(1)) * 255);
}

// encodes a normal vector in a BGRA uint32
inline uint32_t quantize_normal_vector(const Vector3& v)
{
	return uint32_t(quantize_ldr_value((v[0] + Scalar(1)) / Scalar(2)))
		| uint32_t(quantize_ldr_value((v[1] + Scalar(1)) / Scalar(2)) << 8)
		| uint32_t(quantize_ldr_value((v[2] + Scalar(1)) / Scalar(2)) << 16)
		| uint32_t(0xFF000000);
}

// produces a list of rasterization samples from triangle vertices in image coordinates
std::vector<RasterizationSample> rasterize_triangle(Vector2 p0, Vector2 p1, Vector2 p2)
{
	Box2 box;
	box.add(p0);
	box.add(p1);
	box.add(p2);
	box.add(Vector2(std::floor(box.cmin(0)), std::floor(box.cmin(1))));
	box.add(Vector2(std::ceil(box.cmax(0)), std::ceil(box.cmax(1))));

	int w = (int)box.diagonal().x();
	int h = (int)box.diagonal().y();

	Scalar w_tri = determinant(p1 - p0, p2 - p0);

	std::vector<RasterizationSample> samples;

	for (int y_step = 0; y_step < h; ++y_step) {
		for (int x_step = 0; x_step < w; ++x_step) {
			Vector2 p = box.cmin + Vector2(x_step + 0.5, y_step + 0.5);
			Vector3 w;
			w(0) = determinant(p2 - p1, p - p1) / w_tri;
			w(1) = determinant(p0 - p2, p - p2) / w_tri;
			w(2) = determinant(p1 - p0, p - p0) / w_tri;
			if (w.minCoeff() >= 0)
				samples.push_back(RasterizationSample{ .coord = p.cast<int>(), .w = w });
		}
	}

	return samples;
}

MatrixXu32 sample_normal_map(int res, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN,
	const MatrixX& V, const MatrixXi& F, const MatrixX& UV, const MatrixX& VN, const MatrixX& VT,
	SubdivisionMesh& umesh)
{
	BVHTree bvh;
	bvh.build_tree(&hi_V, &hi_F, &hi_VN);

	MatrixX FN = compute_face_normals(hi_V, hi_F);
	auto bvh_test = [&FN](const IntersectionInfo& ii) -> bool {
		return ii.d.dot(FN.row(ii.fi)) >= 0;
	};

	// compute bitangents
	MatrixX VBT; 
	VBT.resizeLike(VN);
	for (int i = 0; i < V.rows(); ++i) {
		Vector3 vn = VN.row(i);
		Vector3 vt = VT.row(i).head(3);
		Scalar s = VT(i, 3);
		VBT.row(i) = vn.cross(vt) * s;
	}

	// RGBA normal texture (stored as ABGR for compat with stb_image_write, which addresses bytes)
	uint32_t background = 0x00FF7F7F;
	MatrixXu32 N = MatrixXu32::Constant(res, res, background);

	// Mask
	MatrixXu8 mask = MatrixXu8::Constant(res, res, 0);

	RowVector2 img_size(res, res);

	auto sample_base_tri = [&](const SubdivisionTri& st) {
		BarycentricGrid grid(st.subdivision_level());

		// extract base vertex attributes
		MatrixX base_UV = MatrixX::Constant(3, 2, 0);
		Matrix3 base_VN;
		MatrixX base_VT = MatrixX::Constant(3, 4, 0);
		Matrix3 base_VBT;

		for (int i = 0; i < 3; ++i) {
			base_UV.row(i) = UV.row(F(st.base_fi, i));
			base_VN.row(i) = VN.row(F(st.base_fi, i));
			base_VT.row(i) = VT.row(F(st.base_fi, i));
			base_VBT.row(i) = VBT.row(F(st.base_fi, i));
		}

		// iterate over microfaces
		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			
			// compute displaced microvertex positions
			Matrix3 disp_V;
			for (int i = 0; i < 3; ++i)
				disp_V.row(i) = st.V.row(st.F(ufi, i)) + st.VD.row(st.F(ufi, i));

			Matrix3 micro_VN;
			for (int i = 0; i < 3; ++i)
				micro_VN.row(i) = st.VN.row(st.F(ufi, i));

			// compute barycentric weights of the face microvertices
			Matrix3 W;
			for (int i = 0; i < 3; ++i)
				W.row(i) = grid.barycentric_coord(st.F(ufi, i));

			MatrixX micro_UV = W * base_UV;

			// rasterize microface
			std::vector<RasterizationSample> samples = rasterize_triangle(
				micro_UV.row(0).cwiseProduct(img_size),
				micro_UV.row(1).cwiseProduct(img_size),
				micro_UV.row(2).cwiseProduct(img_size));

			// iterate over raster samples, and sample normals
			for (const RasterizationSample& rs : samples) {
				if (!mask(rs.coord.x(), rs.coord.y())) {
					// compute barycentric weights of the raster sample wrt base (interpolate weights of the microvertices)
					Vector3 sw = rs.w.transpose() * W;

					//// find interpolated ray
					//// start from base position
					//Vector3 p = sw[0] * st.base_V.row(0) + sw[1] * st.base_V.row(1) + sw[2] * st.base_V.row(2);
					//// travel along displacement direction
					//Vector3 d = sw[0] * st.base_VD.row(0) + sw[1] * st.base_VD.row(1) + sw[2] * st.base_VD.row(2);

					// find interpolated ray - explode style
					// start from the microdisplaced position
					Vector3 p = rs.w.transpose() * disp_V;
					// travel along the interpolated per-microvertex normal
					Vector3 d = rs.w.transpose() * micro_VN;

					// ray-cast and sample the true hi-res normal
					IntersectionInfo isect;
					if (bvh.ray_intersection(p, d, &isect, bvh_test)) {
						Vector3 n0 = hi_VN.row(hi_F(isect.fi, 0));
						Vector3 n1 = hi_VN.row(hi_F(isect.fi, 1));
						Vector3 n2 = hi_VN.row(hi_F(isect.fi, 2));
						Vector3 sampled_normal = (isect.b[0] * n0 + isect.b[1] * n1 + isect.b[2] * n2).normalized();

						// transform to tangent space
						Matrix3 M;
						M.col(0) = (sw.transpose() * base_VT.block(0, 0, 3, 3)).normalized();
						M.col(1) = (sw.transpose() * base_VBT).normalized();
						M.col(2) = (sw.transpose() * base_VN).normalized();

						Vector3 tangent_normal = M.inverse() * sampled_normal;
						tangent_normal[0] = std::clamp(tangent_normal[0], Scalar(-1), Scalar(1));
						tangent_normal[1] = std::clamp(tangent_normal[1], Scalar(-1), Scalar(1));
						tangent_normal[2] = std::clamp(tangent_normal[2], Scalar(0), Scalar(1));

						// quantize to [0,255] per channel and store
						N(rs.coord.x(), rs.coord.y()) = quantize_normal_vector(tangent_normal);
					}

					mask(rs.coord.x(), rs.coord.y()) = 1;
				}
			}
		}
	};

	Timer t;

	std::for_each(std::execution::par_unseq, umesh.faces.begin(), umesh.faces.end(), sample_base_tri);

	texture_filter::PullPush(N, background);

	for (int y = 0; y < N.rows(); ++y) {
		for (int x = 0; x < N.cols(); ++x)
			N(y, x) |= 0xFF000000;
	}

	std::cerr << "Normal sampling took " << t.time_elapsed() << " seconds" << std::endl;

	return N;
}

MatrixXu32 sample_color_map(const std::string& texture_path, int res, const MatrixX& hi_V, const MatrixXi& hi_F, const MatrixX& hi_VN, const MatrixX& hi_UV,
	const MatrixX& V, const MatrixXi& F, const MatrixX& UV,
	SubdivisionMesh& umesh)
{
	BVHTree bvh;
	bvh.build_tree(&hi_V, &hi_F, &hi_VN);

	MatrixX FN = compute_face_normals(hi_V, hi_F);
	auto bvh_test = [&FN](const IntersectionInfo& ii) -> bool {
		return ii.d.dot(FN.row(ii.fi)) >= 0;
	};

	// RGBA normal texture (stored as ABGR for compat with stb_image_write, which addresses bytes)
	uint32_t background = 0x00FF7F7F;
	MatrixXu32 C = MatrixXu32::Constant(res, res, background);

	// Mask
	MatrixXu8 mask = MatrixXu8::Constant(res, res, 0);

	RowVector2 img_size(res, res);

	// Load texture image
	TextureImage texture(texture_path);

	auto sample_base_tri = [&](const SubdivisionTri& st) {
		BarycentricGrid grid(st.subdivision_level());

		// extract base vertex attributes
		MatrixX base_UV = MatrixX::Constant(3, 2, 0);

		for (int i = 0; i < 3; ++i) {
			base_UV.row(i) = UV.row(F(st.base_fi, i));
		}

		// iterate over microfaces
		for (int ufi = 0; ufi < st.F.rows(); ++ufi) {
			
			// compute displaced microvertex positions
			Matrix3 disp_V;
			for (int i = 0; i < 3; ++i)
				disp_V.row(i) = st.V.row(st.F(ufi, i)) + st.VD.row(st.F(ufi, i));

			Matrix3 micro_VN;
			for (int i = 0; i < 3; ++i)
				micro_VN.row(i) = st.VN.row(st.F(ufi, i));

			// compute barycentric weights of the face microvertices
			Matrix3 W;
			for (int i = 0; i < 3; ++i)
				W.row(i) = grid.barycentric_coord(st.F(ufi, i));

			MatrixX micro_UV = W * base_UV;

			// rasterize microface
			std::vector<RasterizationSample> samples = rasterize_triangle(
				micro_UV.row(0).cwiseProduct(img_size),
				micro_UV.row(1).cwiseProduct(img_size),
				micro_UV.row(2).cwiseProduct(img_size));

			// iterate over raster samples, and sample normals
			for (const RasterizationSample& rs : samples) {
				if (!mask(rs.coord.x(), rs.coord.y())) {
					// compute barycentric weights of the raster sample wrt base (interpolate weights of the microvertices)
					//Vector3 sw = rs.w.transpose() * W;

					// find interpolated ray - explode style
					// start from the microdisplaced position
					Vector3 p = rs.w.transpose() * disp_V;
					// travel along the interpolated per-microvertex normal
					Vector3 d = rs.w.transpose() * micro_VN;

					// ray-cast and sample the surface color
					IntersectionInfo isect;
					if (bvh.ray_intersection(p, d, &isect, bvh_test)) {
						Vector2 uv0 = hi_UV.row(hi_F(isect.fi, 0));
						Vector2 uv1 = hi_UV.row(hi_F(isect.fi, 1));
						Vector2 uv2 = hi_UV.row(hi_F(isect.fi, 2));
						Vector2 uv = isect.b[0] * uv0 + isect.b[1] * uv1 + isect.b[2] * uv2;

						Vector4u8 color = texture.sample(uv);

						// quantize to [0,255] per channel and store
						C(rs.coord.x(), rs.coord.y()) = *(reinterpret_cast<uint32_t*>(color.data()));
					}

					mask(rs.coord.x(), rs.coord.y()) = 1;
				}
			}
		}
	};

	Timer t;

	std::for_each(std::execution::par_unseq, umesh.faces.begin(), umesh.faces.end(), sample_base_tri);

	texture_filter::PullPush(C, background);

	for (int y = 0; y < C.rows(); ++y) {
		for (int x = 0; x < C.cols(); ++x)
			C(y, x) |= 0xFF000000;
	}

	std::cerr << "Color sampling took " << t.time_elapsed() << " seconds" << std::endl;

	return C;
}

int main(int argc, char* argv[])
{
	if (argc < 5) {
		std::cerr << "Usage: " << fs::path(argv[0]).filename().string() << " HI MICROMESH RESOLUTION [NAME]" << std::endl;
		std::cerr << "    HI         The high-resolution mesh glTF file" << std::endl;
		std::cerr << "    MICROMESH  The micromesh glTF file" << std::endl;
		std::cerr << "    RESOLUTION The resolution of the normal map" << std::endl;
		std::cerr << "    NAME       The name of the output mesh file" << std::endl;
		return -1;
	}

	GLTFReadInfo read_hi;
	if (!read_gltf(argv[1], read_hi)) {
		std::cerr << "Error reading gltf file " << argv[1] << std::endl;
		return -1;
	}

	Assert(read_hi.has_vertices());
	Assert(read_hi.has_faces());

	MatrixX hi_V = read_hi.get_vertices();
	MatrixXi hi_F = read_hi.get_faces();
	MatrixX hi_VN = read_hi.has_normals() ? read_hi.get_normals() : compute_vertex_normals(hi_V, hi_F);

	GLTFReadInfo read_micromesh;
	if (!read_gltf(argv[2], read_micromesh)) {
		std::cerr << "Error reading gltf file " << argv[2] << std::endl;
		return -1;
	}

	Assert(read_micromesh.has_vertices());
	Assert(read_micromesh.has_faces());
	Assert(read_micromesh.has_uvs());
	Assert(read_micromesh.has_normals());
	Assert(read_micromesh.has_tangents());
	Assert(read_micromesh.has_directions());
	Assert(read_micromesh.has_subdivisions());
	Assert(read_micromesh.has_topology_flags());
	Assert(read_micromesh.has_subdivision_mesh());

	MatrixX V = read_micromesh.get_vertices();
	MatrixXi F = read_micromesh.get_faces();
	MatrixX UV = read_micromesh.get_uvs();
	MatrixX VN = read_micromesh.get_normals();
	MatrixX VT = read_micromesh.get_tangents();
	MatrixX VD = read_micromesh.get_directions();
	VectorXu8 S = read_micromesh.get_subdivisions();
	VectorXu8 E = read_micromesh.get_topology_flags();
	SubdivisionMesh umesh = read_micromesh.get_subdivision_mesh();

	int resolution = std::atoi(argv[3]);
	Assert(resolution > 0);

	// set microvertex normals from displaced geometry
	{
		MatrixX uV;
		MatrixXi uF;
		umesh.extract_mesh(uV, uF);

		MatrixXu8 FEB = per_face_edge_border_flag(uV, uF);
		VectorXu8 VB = per_vertex_border_flag(uV, uF, FEB);

		unify_vertices(uV, uF, VB, FEB);
		MatrixX uVN = compute_vertex_normals(uV, uF);

		int ufi = 0;
		for (SubdivisionTri& st : umesh.faces) {
			for (int fi = 0; fi < st.F.rows(); ++fi) {
				Assert(ufi < uF.rows());
				for (int j = 0; j < 3; ++j) {
					st.VN.row(st.F(fi, j)) = uVN.row(uF(ufi, j));
				}
				ufi++;
			}
		}
	}

	MatrixXu32 NMAP = sample_normal_map(resolution, hi_V, hi_F, hi_VN, V, F, UV, VN, VT, umesh);
	MatrixXu32 CMAP;

	if (read_hi.color_textures.size() > 0 && read_hi.has_uvs()) {
		MatrixX hi_UV = read_hi.get_uvs();
		Assert(read_hi.color_textures.size() == 1);

		fs::path gltf_dir = fs::path(argv[1]).parent_path();
		std::string texture_path = gltf_dir.append(read_hi.color_textures[0]).string();
		CMAP = sample_color_map(texture_path, resolution, hi_V, hi_F, hi_VN, hi_UV, V, F, UV, umesh);
	}

	fs::path gltf_name(argv[4]);
	gltf_name.replace_extension();

	// pack subdivisions and edge flags
	VectorXu8 SB = S;
	for (int i = 0; i < SB.size(); ++i)
		SB(i) = (S(i) << 3) | E(i);


	// write mesh
	Bary bary;
	extract_displacement_bary_data(umesh, &bary, true);

	// write the uv-mapped micromesh
	GLTFWriteInfo write_info;
	write_info
		.write_faces(&F)
		.write_vertices(&V)
		.write_uvs(&UV)
		.write_normals(&VN)
		.write_tangents(&VT)
		.write_directions(&VD)
		.write_subdivision_bits(&SB)
		.write_bary(&bary);

	if (NMAP.size() > 0) {
		fs::path normal_map_filename = gltf_name.string() + "_normal.png";
		int status = stbi_write_png(normal_map_filename.string().c_str(), NMAP.rows(), NMAP.cols(), 4, NMAP.data(), 0);
		if (status != 0) {
			std::cerr << "Written " << normal_map_filename << std::endl;
		}
		else {
			std::cerr << "Error writing " << normal_map_filename << std::endl;
			return -1;
		}
		write_info.write_normal_texture(normal_map_filename.string());
	}

	if (CMAP.size() > 0) {
		fs::path color_map_filename = gltf_name.string() + "_color.png";
		int status = stbi_write_png(color_map_filename.string().c_str(), CMAP.rows(), CMAP.cols(), 4, CMAP.data(), 0);
		if (status != 0) {
			std::cerr << "Written " << color_map_filename << std::endl;
		}
		else {
			std::cerr << "Error writing " << color_map_filename << std::endl;
			return -1;
		}
		write_info.write_color_texture(color_map_filename.string());
	}

	write_gltf(gltf_name.string(), write_info);

	return 0;
}

