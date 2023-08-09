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

#include "mesh_io_gltf.h"
#include "mesh_io_bary.h"
#include "utils.h"
#include "micro.h"

#include "mesh_io.h"

#define CGLTF_IMPLEMENTATION
#define CGLTF_WRITE_IMPLEMENTATION
#include <cgltf_write.h>

#include <map>
#include <filesystem>
#include <cstdio>

#include <baryutils/baryutils.h>

namespace fs = std::filesystem;

struct BaryEntry {
	uint32_t start_fn;
	uint32_t fn;
	cgltf_size group_index;
};

template <typename T>
uint64_t byte_size(const std::vector<T>& vec) { return vec.size() * sizeof(T); }

template <typename T> T* allocate(std::vector<T>& vec)
{
    // vector cannot grow : reserve enough capacity before calling !
    assert(vec.size() < vec.capacity());
    vec.push_back({});
    T* item = &vec.back();
    return item;
}

static const uint8_t* get_byte_pointer(const cgltf_buffer_view* view)
{
	if(view->data)
		return (const uint8_t*)view->data;

	if(!view->buffer->data)
		return nullptr;

	const uint8_t* result = (const uint8_t*)view->buffer->data;
	return result + view->offset;
}

bool read_gltf(const std::string& filename, GLTFReadInfo& read_info)
{
	cgltf_options options = {};
	cgltf_data* data = nullptr;
	cgltf_result result = cgltf_parse_file(&options, filename.c_str(), &data);

	if (result == cgltf_result_success)
		result = cgltf_load_buffers(&options, data, filename.c_str());

	if (result != cgltf_result_success) {
		std::cerr << "cgltf error " << result << std::endl;
		return false;
	}

	std::vector<cgltf_float> vpos_vec;
	std::vector<uint32_t> face_vec;
	std::vector<cgltf_float> uv_vec;
	std::vector<cgltf_float> norm_vec;
	std::vector<cgltf_float> tg_vec;
	std::vector<cgltf_float> vdir_vec;
	std::vector<uint8_t> subd_vec; // subdivisions
	std::vector<uint8_t> flags_vec; // edge flags

	std::map<cgltf_nv_micromap*, std::vector<BaryEntry>> bary_map;

	for (int mi = 0; mi < data->meshes_count; ++mi) {
		cgltf_mesh* mesh = data->meshes + mi;
		for (int pi = 0; pi < mesh->primitives_count; ++pi) {
			cgltf_primitive* prim = mesh->primitives + pi;
			if (prim->type == cgltf_primitive_type::cgltf_primitive_type_triangles) {
				Assert(prim->indices && "TODO Add support for non-indexed geometry");

				uint32_t start_vn = vpos_vec.size() / 3;
				uint32_t start_fn = face_vec.size() / 3;

				// vertex attributes

				for (int ai = 0; ai < prim->attributes_count; ++ai) {
					cgltf_attribute* attribute = prim->attributes + ai;
					if (attribute->type == cgltf_attribute_type_position) {
						Assert(!attribute->data->is_sparse && "TODO Add support for sparse accessors");
						Assert(attribute->data->type == cgltf_type::cgltf_type_vec3);
						vpos_vec.resize(vpos_vec.size() + 3 * attribute->data->count);
						for (int i = 0; i < attribute->data->count; ++i) {
							cgltf_accessor_read_float(attribute->data, i, &vpos_vec[3 * (start_vn + i)], 3);
						}
					}
					else if (attribute->type == cgltf_attribute_type_texcoord) {
						Assert(attribute->data->type == cgltf_type::cgltf_type_vec2);
						Assert(attribute->data->component_type == cgltf_component_type::cgltf_component_type_r_32f);
						uv_vec.resize(uv_vec.size() + 2 * attribute->data->count);
						for (int i = 0; i < attribute->data->count; ++i) {
							cgltf_accessor_read_float(attribute->data, i, &uv_vec[2 * (start_vn + i)], 2);
						}
					}
					else if (attribute->type == cgltf_attribute_type_normal) {
						Assert(attribute->data->type == cgltf_type::cgltf_type_vec3);
						norm_vec.resize(norm_vec.size() + 3 * attribute->data->count);
						for (int i = 0; i < attribute->data->count; ++i) {
							cgltf_accessor_read_float(attribute->data, i, &norm_vec[3 * (start_vn + i)], 3);
						}
					}
					else if (attribute->type == cgltf_attribute_type_tangent) {
						Assert(attribute->data->type == cgltf_type::cgltf_type_vec3 || attribute->data->type == cgltf_type::cgltf_type_vec4);
						tg_vec.resize(tg_vec.size() + 4 * attribute->data->count);
						for (int i = 0; i < attribute->data->count; ++i) {
							int dims = attribute->data->type == cgltf_type::cgltf_type_vec4 ? 4 : 3;
							cgltf_accessor_read_float(attribute->data, i, &tg_vec[4 * (start_vn + i)], dims);
							if (dims == 3)
								tg_vec[4 * (start_vn + i) + 3] = 1.0f;
						}
					}
				}

				uint32_t vn = (vpos_vec.size() / 3) - start_vn;

				// faces

				for (int i = 0; i < prim->indices->count; ++i) {
					uint32_t index = start_vn + (cgltf_accessor_read_index(prim->indices, i));
					face_vec.push_back(index);
				}
				uint32_t fn = (face_vec.size() / 3) - start_fn;

				// also get texture paths if present TODO FIXME
				if (prim->material) {
					cgltf_material* mat = prim->material;
					if (mat->normal_texture.texture) {
						cgltf_image *normal_image = mat->normal_texture.texture->image;
						if (normal_image && normal_image->uri)
							read_info.normal_textures.push_back(std::string(normal_image->uri));
					}
					if (mat->has_pbr_metallic_roughness) {
						cgltf_pbr_metallic_roughness mat_mr = mat->pbr_metallic_roughness;
						if (mat_mr.base_color_texture.texture) {
							cgltf_image* color_image = mat_mr.base_color_texture.texture->image;
							if (color_image && color_image->uri)
								read_info.color_textures.push_back(std::string(color_image->uri));
						}
					}
					if (mat->has_pbr_specular_glossiness) {
						cgltf_pbr_specular_glossiness mat_sg = mat->pbr_specular_glossiness;
						if (mat_sg.diffuse_texture.texture) {
							cgltf_image* color_image = mat_sg.diffuse_texture.texture->image;
							if (color_image && color_image->uri)
								read_info.color_textures.push_back(std::string(color_image->uri));
						}
					}
				}

				// micromesh extensions

				if (prim->has_nv_displacement_micromap) {
					cgltf_nv_displacement_micromap* nv_disp = &prim->nv_displacement_micromap;

					// displacement directions
					if (nv_disp->directions) {
						cgltf_accessor* directions = nv_disp->directions;
						Assert(!directions->is_sparse && "No sparse accessors");
						Assert(directions->count == vn);

						vdir_vec.resize(vdir_vec.size() + 3 * directions->count);
						for (int i = 0; i < directions->count; ++i) {
							cgltf_accessor_read_float(directions, i, &vdir_vec[3 * (start_vn + i)], 3);
						}
					}

					// subdivision levels will be written to and read from the .bary file

					// edge flags
					if (nv_disp->primitive_flags) {
						flags_vec.resize(flags_vec.size() + fn);
						cgltf_accessor* topology_flags = nv_disp->primitive_flags;
						Assert(topology_flags->count == fn);
						std::memcpy(flags_vec.data() + start_fn,
							get_byte_pointer(topology_flags->buffer_view) + topology_flags->offset,
							topology_flags->count * sizeof(uint8_t));
					}

					bary_map[nv_disp->micromap].push_back(BaryEntry{ .start_fn = start_fn, .fn = fn, .group_index = nv_disp->group_index });

				}
			}
		}
	}

	// XXX I'm assuming that each mesh primitive is mapped 1:1 with a group in the corresponding .bary

	// now build the base and subdivision mesh structure

	Assert(vpos_vec.size() % 3 == 0);
	Assert(face_vec.size() % 3 == 0);

	int vn = vpos_vec.size() / 3;
	int fn = face_vec.size() / 3;

	Assert(uv_vec.size() == 0 || uv_vec.size() == 2 * vn);
	Assert(norm_vec.size() == 0 || norm_vec.size() == 3 * vn);
	Assert(tg_vec.size() == 0 || tg_vec.size() == 4 * vn);

	Assert(vdir_vec.size() == 0 || vdir_vec.size() == 3 * vn);
	Assert(subd_vec.size() == 0 || subd_vec.size() == fn);
	Assert((subd_vec.size() == 0 || (flags_vec.size() == subd_vec.size())) && "Subdivision levels and topology flags sizes do not match");

	auto fill_buffer = []<typename MatrixType, typename T>(MatrixType & M, const std::vector<T>& buffer, int nprim, int dim) {
		Assert(buffer.size() % dim == 0);
		for (int i = 0; i < nprim; ++i)
			for (int j = 0; j < dim; ++j)
				M(i, j) = (typename MatrixType::Scalar) buffer[dim * i + j];
	};

	if (vpos_vec.size() > 0) {
		read_info._V.resize(vn, 3);
		fill_buffer(read_info._V, vpos_vec, vn, 3);
	}

	if (face_vec.size() > 0) {
		read_info._F.resize(fn, 3);
		fill_buffer(read_info._F, face_vec, fn, 3);
	}

	if (uv_vec.size() > 0) {
		read_info._UV.resize(vn, 2);
		fill_buffer(read_info._UV, uv_vec, vn, 2);
	}

	if (norm_vec.size() > 0) {
		read_info._VN.resize(vn, 3);
		fill_buffer(read_info._VN, norm_vec, vn, 3);
	}

	if (tg_vec.size() > 0) {
		read_info._VT.resize(vn, 4);
		fill_buffer(read_info._VT, tg_vec, vn, 4);
	}

	if (vdir_vec.size() > 0) {
		read_info._VD.resize(vn, 3);
		fill_buffer(read_info._VD, vdir_vec, vn, 3);
	}

	if (subd_vec.size() > 0) {
		read_info._subdivisions.resize(fn);
		fill_buffer(read_info._subdivisions, subd_vec, fn, 1);
	}

	if (flags_vec.size() > 0) {
		read_info._topology_flags.resize(fn);
		fill_buffer(read_info._topology_flags, flags_vec, fn, 1);
	}

	// if the data is there, also initialize the micromesh
	if (read_info.has_directions() && read_info.has_topology_flags()) {

		// For each bary file (micromap) I have a list of BaryEntry objects each with start_fn, fn, group
		// the primitive index in bary_data.basePrimitives is start_fn + i
		// the index of the corresponding base primitive is 
		//     base_index = (basePrim.valueByteOffset / uncompressed->base.valueByteSize);
		// therefore, the j-th microvertex index is 
		//     uvert_index = (basePrim.valueByteOffset / uncompressed->base.valueByteSize) + j;
		// 
		// Now, in bary_data.distances are the packed bytes of possibly *multiple* primitive groups;
		// to retrieve the displacement value we must first offset by the index of the first displacement
		// of the group
		//     value_index = uvert_index + group.valueFirst;
		// This is the index corresponding to the displacement of j-th microvertex within base primitive i
		// Warning: the distances vector is in bytes, and there are the scale and bias to account for
		// 
		// Ref. line 565 of baryset.cpp (vk_micro_displacements)
		// 
		// Warning: see the XXX above about the 1:1 map between mesh base primitives and groups

		// we first load the bary data, then we extract the subdivision levels, next we subdivide the mesh and finally we load the displacements

		// load the bary data

		std::map<cgltf_nv_micromap*, baryutils::BaryBasicData> bary_data_map;
		for (const auto& entry : bary_map) {
			cgltf_nv_micromap* micromap = entry.first;

			if (micromap == nullptr)
				continue;

			// load the bary data

			baryutils::BaryBasicData& bbd = bary_data_map[micromap];

			fs::path gltf_path = filename;
			fs::path micromap_uri = micromap->uri;
			fs::path bary_path = gltf_path.parent_path() / micromap_uri;
			bary::Result result = bbd.load(bary_path.string());

			bary::Format bary_format = bbd.valuesInfo.valueFormat;

			if (result == bary::Result::eSuccess) {
				// TODO add support for all the encodings
				if (bbd.valuesInfo.valueLayout != bary::ValueLayout::eTriangleUmajor) {
					std::cerr << "Unsupported bary value order, only WV_TO_U is accepted" << std::endl;
					return false;
				}
				if (bary_format != bary::Format::eR32_sfloat && bary_format != bary::Format::eR11_unorm_pack16 && bary_format != bary::Format::eR11_unorm_packed_align32) {
					std::cerr << "Unsupported displacement value format" << std::endl;
					return false;
				}
			}
			else {
				std::cerr << "Error reading .bary resource file " << bary_path << ": "
					<< bary::baryResultGetName(result) << std::endl;
				return false;
			}
		}

		// extract subdivision levels and topology flags, and compute the umesh structure

		VectorXu8 subdivision_bits = VectorXu8::Constant(fn, 0);

		for (const auto& entry : bary_map) {
			cgltf_nv_micromap* micromap = entry.first;
			const baryutils::BaryBasicData& bbd = bary_data_map[entry.first];
			for (const BaryEntry& be : entry.second) {
				Assert(be.fn == bbd.groups[be.group_index].triangleCount);
				const bary::Group& group = bbd.groups[be.group_index];
				for (uint32_t i = 0; i < be.fn; ++i) {
					uint32_t base_fi = be.start_fn + i;
					const bary::Triangle& bary_primitive = bbd.triangles[group.triangleFirst + i];
					subdivision_bits(base_fi) = (uint8_t(bary_primitive.subdivLevel) << 3) | read_info._topology_flags(base_fi);
				}
			}
		}

		read_info._umesh.compute_mesh_structure(read_info._V, read_info._F, subdivision_bits);

		// displace the micromesh using the referenced bary files
		// TODO refactor this in a different function
		// TODO handle URIs to network resources etc...
		for (const auto& entry : bary_map) {
			cgltf_nv_micromap* micromap = entry.first;
			const baryutils::BaryBasicData& bbd = bary_data_map[entry.first];
			bary::Format bary_format = bbd.valuesInfo.valueFormat;

			for (const BaryEntry& be : entry.second) {
				Assert(be.fn == bbd.groups[be.group_index].triangleCount);
				const bary::Group& group = bbd.groups[be.group_index];
				for (uint32_t i = 0; i < be.fn; ++i) {
					uint32_t base_fi = be.start_fn + i;
					const bary::Triangle& bary_primitive = bbd.triangles[group.triangleFirst + i];
					SubdivisionTri& st = read_info._umesh.faces[base_fi];

					Assert(bary_format == bary::Format::eR32_sfloat || bary_format == bary::Format::eR11_unorm_pack16 || bary_format == bary::Format::eR11_unorm_packed_align32);

					BarycentricGrid grid(st.subdivision_level());

					std::vector<Scalar> displacements;
					if (bary_format == bary::Format::eR11_unorm_packed_align32) {
						const uint32_t* u32_ptr = bbd.getValues<uint32_t>() + (bary_primitive.valuesOffset / sizeof(uint32_t));
						ReadPacked32 r(*u32_ptr++);
						for (uint32_t j = 0; j < grid.num_samples(); ++j) {
							uint32_t dq;
							int unread_bits = r.unpack(11, &dq);
							if (unread_bits) {
								//Assert(it != v.end());
								r.reset(*u32_ptr++);
								uint32_t leftover;
								Assert(r.unpack(unread_bits, &leftover) == 0);
								dq = (leftover << (11 - unread_bits)) | dq;
							}
							displacements.push_back((dq / float(UNORM11_MAX)) * group.floatScale.r + group.floatBias.r);
						}
					}
					else {
						uint32_t first_microdisp = bary_primitive.valuesOffset; // uncompressed values -> valuesOffset is an index
						for (uint32_t j = 0; j < grid.num_samples(); ++j) {
							uint32_t microdisp_index = first_microdisp + j;
							float d = 0;
							switch (bary_format) {
							case bary::Format::eR32_sfloat:
								d = bbd.getValues<float>()[microdisp_index];
								break;
							case bary::Format::eR11_unorm_pack16:
								d = bbd.getValues<uint16_t>()[microdisp_index] / float(UNORM11_MAX);
								break;
							default:
								Assert(0 && "Unsupported bary format");
							}
							//displacements.push_back(microdisp_buffer[microdisp_index] * group.floatScale.r + group.floatBias.r);
							displacements.push_back(d * group.floatScale.r + group.floatBias.r);
						}
					}

					st.base_VD.resize(3, 3);
					for (int k = 0; k < 3; ++k) {
						st.base_VD.row(k) = read_info._VD.row(read_info._F(st.base_fi, k));
					}

					// now displace the subdivision tris
					// careful with the order of displacements in the bary file and the one used in SubdivisionMesh...
					auto itd = displacements.begin();
					st.VD.resize(grid.num_samples(), 3);
					for (int k = 0; k < grid.samples_per_side(); ++k) {
						for (int h = 0; h < grid.samples_per_side() - k; ++h) {
							int i = k + h;
							int j = h;
							int vi = grid.index(i, j);
							// compute displacement value
							Vector3 w = grid.barycentric_coord(i, j);

							st.VD.row(vi) = w(0) * st.base_VD.row(0) + w(1) * st.base_VD.row(1) + w(2) * st.base_VD.row(2);
							st.VD.row(vi) *= *itd++;
						}
					}
				}
			}

			read_info._umesh._status_bits |= SubdivisionMesh::Displaced;
		}
	}

	return true;
}

enum class BufferViewType : uint8_t {
	Positions = 0,
	Colors,
	Tex_Coords,
	Normals,
	Tangents,
	Bone_Ids,
	Bone_Weights,
	Indices,
	Displacement_Vectors,
	Subdivision_Levels,
	Topology_Flags
};

// Simple gltf exporter, exports a file with a single mesh
// and a single array of primitives, nothing fancy
bool write_gltf(const std::string& name, const GLTFWriteInfo& write_info)
{
	Assert(write_info.has_faces());
	Assert(write_info.has_vertices());

	// count valid (non-degenerate) faces
	int fn = write_info.F().rows();
	std::vector<bool> valid(write_info.F().rows(), true);
	for (int i = 0; i < write_info.F().rows(); ++i) {
		if (write_info.F()(i, 0) == INVALID_INDEX || write_info.F()(i, 1) == INVALID_INDEX && write_info.F()(i, 2) == INVALID_INDEX) {
			valid[i] = false;
			fn--;
		}
	}

	Assert(fn >= 0);

	std::list<std::string> strings;

	// preallocate enough memory to avoid invalidating pointers when calling allocate()
	const int allocate_capacity = 100; // at least position, normals, colors, directions, uvs, tangents index, subdivision_level, topology_flags
	std::vector<cgltf_buffer_view> buffer_views;
	std::vector<cgltf_accessor> accessors;
	std::vector<cgltf_attribute> attributes;

	buffer_views.reserve(allocate_capacity);
	accessors.reserve(allocate_capacity);
	attributes.reserve(allocate_capacity);

	std::vector<cgltf_image> images;
	images.reserve(allocate_capacity);

	std::vector<cgltf_nv_micromap> micromaps;
	micromaps.reserve(allocate_capacity);

	std::vector<cgltf_texture> textures;
	textures.reserve(allocate_capacity);

	std::vector<cgltf_material> materials;
	materials.reserve(allocate_capacity);

	std::vector<float> positions;
	std::vector<float> normals;
	std::vector<float> colors;
	std::vector<float> directions;
	std::vector<float> uvs;
	std::vector<float> tangents;
	std::vector<uint8_t> subdivision_levels;
	std::vector<uint8_t> topology_flags;

	positions.reserve(3 * write_info.V().rows());
	if (write_info.has_normals())
		normals.reserve(3 * write_info.VN().rows());
	if (write_info.has_colors())
		colors.reserve(3 * write_info.VC().rows());
	if (write_info.has_directions())
		directions.reserve(3 * write_info.VD().rows());
	if (write_info.has_uvs())
		uvs.reserve(2 * write_info.UV().rows());
	if (write_info.has_tangents())
		tangents.reserve(4 * write_info.VT().rows());
	if (write_info.has_subdivision_bits()) {
		subdivision_levels.reserve(fn);
		topology_flags.reserve(fn);
	}

	Vector3f pos_min(Infinityf, Infinityf, Infinityf);
	Vector3f pos_max(-Infinityf, -Infinityf, -Infinityf);

	auto fill_vertex_buffer = [&](const MatrixX& VDATA, std::vector<float>& buffer, int dim, bool is_position) {
		for (int i = 0; i < (int)VDATA.rows(); ++i) {
			for (int j = 0; j < dim; ++j) {
				float v_j = (float)VDATA(i, j);
				buffer.push_back(v_j);
				if (is_position) {
					pos_min(j) = std::min(pos_min(j), v_j);
					pos_max(j) = std::max(pos_max(j), v_j);
				}
			}
		}
	};

	fill_vertex_buffer(write_info.V(), positions, 3, true);
	if (write_info.has_normals())
		fill_vertex_buffer(write_info.VN(), normals, 3, false);
	if (write_info.has_colors())
		fill_vertex_buffer(write_info.VC(), colors, write_info.VC().cols(), false);
	if (write_info.has_directions())
		fill_vertex_buffer(write_info.VD(), directions, 3, false);
	if (write_info.has_uvs())
		fill_vertex_buffer(write_info.UV(), uvs, 2, false);
	if (write_info.has_tangents())
		fill_vertex_buffer(write_info.VT(), tangents, 4, false);

	if (write_info.has_subdivision_bits()) {
		for (int i = 0; i < write_info.F().rows(); ++i) {
			if (valid[i]) {
				subdivision_levels.push_back(write_info.subdivision_bits()(i) >> 3);
				topology_flags.push_back(write_info.subdivision_bits()(i) & 0x07);
			}
		}
	}

	// cgltf does not export signed index values
	std::vector<uint32_t> indices;
	indices.reserve(3 * fn);
	for (int i = 0; i < write_info.F().rows(); ++i) {
		if (valid[i]) {
			indices.push_back(write_info.F()(i, 0));
			indices.push_back(write_info.F()(i, 1));
			indices.push_back(write_info.F()(i, 2));
		}
	}

	uint64_t buffer_size = 0;
	buffer_size += byte_size(indices);
	buffer_size += byte_size(positions);
	if (write_info.has_normals())
		buffer_size += byte_size(normals);
	if (write_info.has_colors())
		buffer_size += byte_size(colors);
	if (write_info.has_directions())
		buffer_size += byte_size(directions);
	if (write_info.has_uvs())
		buffer_size += byte_size(uvs);
	if (write_info.has_tangents())
		buffer_size += byte_size(tangents);
	if (write_info.has_subdivision_bits()) {
		buffer_size += byte_size(subdivision_levels);
		buffer_size += byte_size(topology_flags);
	}

	std::vector<uint8_t> blob(buffer_size, 0);

	cgltf_buffer buffer = {};
	buffer.data = blob.data();
	buffer.size = blob.size();

	struct BufferViewRef {
		cgltf_buffer_view* buffer_view;
		BufferViewType type;
	};

	std::map<std::string, BufferViewRef> buffer_view_map;
	
	int offset = 0;
	auto init_buffer_view = [&]<typename T>(const std::vector<T> &vec, BufferViewType type, const char* name) {
		uint64_t size = byte_size(vec);
		std::memcpy(blob.data() + offset, vec.data(), size);

		cgltf_buffer_view* view = allocate(buffer_views);

		buffer_view_map[name] = { .buffer_view = view, .type = type };

		view->name = (char *) buffer_view_map.find(name)->first.c_str();
		view->buffer = &buffer;
		view->offset = offset;
		view->size = size;
		view->stride = 0;
		view->type = type == BufferViewType::Indices ?
			cgltf_buffer_view_type_indices : cgltf_buffer_view_type_vertices;

		offset += size;
	};

	init_buffer_view(indices, BufferViewType::Indices, "indices");
	init_buffer_view(positions, BufferViewType::Positions, "positions");
	if (write_info.has_normals())
		init_buffer_view(normals, BufferViewType::Normals, "normals");
	if (write_info.has_colors())
		init_buffer_view(colors, BufferViewType::Colors, "colors");
	if (write_info.has_directions())
		init_buffer_view(directions, BufferViewType::Displacement_Vectors, "directions");
	if (write_info.has_uvs())
		init_buffer_view(uvs, BufferViewType::Tex_Coords, "uvs");
	if (write_info.has_tangents())
		init_buffer_view(tangents, BufferViewType::Tangents, "tangents");
	if (write_info.has_subdivision_bits()) {
		init_buffer_view(subdivision_levels, BufferViewType::Subdivision_Levels, "subdivision_levels");
		init_buffer_view(topology_flags, BufferViewType::Topology_Flags, "topology_flags");
	}

	std::vector<cgltf_mesh> meshes;
	std::vector<cgltf_primitive> primitives;

	cgltf_primitive primitive = {};
	primitive.type = cgltf_primitive_type_triangles;

	cgltf_accessor* indices_accessor = allocate(accessors);
	indices_accessor->component_type = cgltf_component_type_r_32u;
	indices_accessor->name = (char *) "indices_accessor";
	indices_accessor->type = cgltf_type_scalar;
	indices_accessor->offset = 0;
	indices_accessor->count = 3 * fn;
	indices_accessor->buffer_view = buffer_view_map["indices"].buffer_view;

	primitive.indices = indices_accessor;

	struct Attribute {
		std::string name;
		cgltf_type type;
		cgltf_attribute_type attr_type;
		cgltf_component_type comp_type;
	};

	static const std::map<BufferViewType, Attribute> attribute_specs = {
		{BufferViewType::Positions, { "POSITION", cgltf_type_vec3, cgltf_attribute_type_position, cgltf_component_type_r_32f }},
		{BufferViewType::Colors, { "COLOR_0", cgltf_type_vec3, cgltf_attribute_type_color, cgltf_component_type_r_32f }},
		{BufferViewType::Tex_Coords, { "TEXCOORD_0", cgltf_type_vec2, cgltf_attribute_type_texcoord, cgltf_component_type_r_32f }},
		{BufferViewType::Normals, { "NORMAL", cgltf_type_vec3, cgltf_attribute_type_normal, cgltf_component_type_r_32f }},
		{BufferViewType::Tangents, { "TANGENT", cgltf_type_vec4, cgltf_attribute_type_tangent, cgltf_component_type_r_32f }},
		{BufferViewType::Bone_Ids, { "JOINTS", cgltf_type_vec4, cgltf_attribute_type_joints, cgltf_component_type_r_32u }},
		{BufferViewType::Bone_Weights, { "WEIGHTS", cgltf_type_vec4, cgltf_attribute_type_joints, cgltf_component_type_r_32f }},
	};

	auto add_attribute = [&](const std::string& attribute_name) {
		auto const buffer_view_ref_it = buffer_view_map.find(attribute_name);
		Assert(buffer_view_ref_it != buffer_view_map.end());

		BufferViewType view_type = buffer_view_ref_it->second.type;
		auto const it = attribute_specs.find(view_type);
		Assert(it != attribute_specs.end());

		cgltf_accessor* accessor = allocate(accessors);
		accessor->name = (char*)strings.insert(strings.end(), attribute_name + "_accessor")->c_str();
		accessor->buffer_view = buffer_view_ref_it->second.buffer_view;
		accessor->type = it->second.type;
		accessor->component_type = it->second.comp_type;
		accessor->count = write_info.V().rows();
		accessor->offset = 0;

		if (view_type == BufferViewType::Positions) {
			accessor->has_min = true;
			accessor->has_max = true;
			for (int i = 0; i < 3; ++i) {
				accessor->min[i] = pos_min(i);
				accessor->max[i] = pos_max(i);
			}
		}

		cgltf_attribute* attribute = allocate(attributes);
		attribute->name = (char*)it->second.name.c_str();
		attribute->type = it->second.attr_type;
		attribute->index = primitive.attributes_count;
		attribute->data = accessor;

		if (!primitive.attributes)
			primitive.attributes = attribute;

		++primitive.attributes_count;
	};

	add_attribute("positions");
	if (write_info.has_normals())
		add_attribute("normals");
	if (write_info.has_colors())
		add_attribute("colors");
	if (write_info.has_uvs())
		add_attribute("uvs");
	if (write_info.has_tangents())
		add_attribute("tangents");

	if (write_info.has_directions()) {
		primitive.has_nv_displacement_micromap = true;
		auto it = buffer_view_map.find("directions");
		Assert(it != buffer_view_map.end());

		cgltf_accessor* accessor = allocate(accessors);
		accessor->name = (char*)strings.insert(strings.end(), "directions_accessor")->c_str();
		accessor->buffer_view = it->second.buffer_view;
		accessor->type = cgltf_type_vec3;
		accessor->component_type = cgltf_component_type_r_32f;
		accessor->count = write_info.V().rows();
		accessor->offset = 0;

		primitive.nv_displacement_micromap.directions = accessor;
	}

	if (write_info.has_subdivision_bits()) {
		primitive.has_nv_displacement_micromap = true;

		auto it = buffer_view_map.find("topology_flags");
		Assert(it != buffer_view_map.end());

		cgltf_accessor *accessor = allocate(accessors);
		accessor->name = (char*)strings.insert(strings.end(), "topology_flags_accessor")->c_str();
		accessor->buffer_view = it->second.buffer_view;
		accessor->type = cgltf_type_scalar;
		accessor->component_type = cgltf_component_type_r_8u;
		accessor->count = fn;
		accessor->offset = 0;

		primitive.nv_displacement_micromap.primitive_flags = accessor;
	}

	const std::string bary_filename = fs::path(name).replace_extension(".bary").string();
	if (write_info.has_bary()) {
		primitive.has_nv_displacement_micromap = true;
		cgltf_nv_micromap* bary_image = allocate(micromaps);
		bary_image->uri = (char*)bary_filename.c_str();
		primitive.nv_displacement_micromap.micromap = bary_image;

		bary::Result result = write_info.bary().data.save(bary_filename.c_str());
		if (result != bary::Result::eSuccess) {
			std::cerr << "Error exporting bary file: " << bary::baryResultGetName(result) << std::endl;
			return false;
		}

		primitive.nv_displacement_micromap.group_index = 0;
	}

	cgltf_mesh mesh = {};
	mesh.name = (char *)name.c_str();
	mesh.primitives = &primitive;
	mesh.primitives_count = 1;

	cgltf_node node = {};
	node.name = (char *)strings.insert(strings.end(), "mesh_node")->c_str();
	node.mesh = &mesh;

	std::vector<cgltf_node*> scene_nodes = { &node };
	cgltf_scene scene = {};
	scene.name = (char*)name.c_str();
	scene.nodes = scene_nodes.data();
	scene.nodes_count = 1;

	// fill the cgltf_data struct and write stuff
	cgltf_options opts = {
		.type = cgltf_file_type_gltf,
		.json_token_count = 0,
		.memory = { nullptr, nullptr, nullptr },
		.file = { nullptr, nullptr, nullptr }
	};

	fs::path gltf_path = name;
	gltf_path.replace_extension(".gltf");

	fs::path bin_path = name;
	bin_path.replace_extension(".bin");

	std::string buffer_uri = bin_path.generic_string();
	buffer.uri = (char*)buffer_uri.c_str();

	static const char* version = "2.0";
	cgltf_data data = {};
	data.file_type = opts.type;
	data.asset.version = (char *)version;

	data.buffers = &buffer;
	data.buffers_count = 1;

	data.buffer_views = buffer_views.data();
	data.buffer_views_count = buffer_views.size();

	data.accessors = accessors.data();
	data.accessors_count = accessors.size();

	if (write_info.has_color_texture() || write_info.has_normal_texture()) {
		// fill in material and texture data

		static const char *material_name = "material_0";

		cgltf_material* material = allocate(materials);
		material->name = (char*)material_name;
		material->has_pbr_metallic_roughness = true;
		material->pbr_metallic_roughness.metallic_factor = 0;
		material->pbr_metallic_roughness.roughness_factor = 1;
		material->pbr_metallic_roughness.base_color_factor[0] = 1;
		material->pbr_metallic_roughness.base_color_factor[1] = 1;
		material->pbr_metallic_roughness.base_color_factor[2] = 1;
		material->pbr_metallic_roughness.base_color_factor[3] = 1;

		if (write_info.has_color_texture()) {
			cgltf_image* color_image = allocate(images);
			color_image->uri = (char *)write_info.color_texture().c_str();
			cgltf_texture* color_texture = allocate(textures);
			color_texture->image = color_image;
			material->pbr_metallic_roughness.base_color_texture.texture = color_texture;
		}

		if (write_info.has_normal_texture()) {
			cgltf_image* normal_image = allocate(images);
			normal_image->uri = (char*)write_info.normal_texture().c_str();
			cgltf_texture* normal_texture = allocate(textures);
			normal_texture->image = normal_image;
			material->normal_texture.texture = normal_texture;
		}

		primitive.material = material;
	}

	data.nv_micromaps = micromaps.data();
	data.nv_micromaps_count = micromaps.size();

	data.images = images.data();
	data.images_count = images.size();

	data.textures = textures.data();
	data.textures_count = textures.size();

	data.materials = materials.data();
	data.materials_count = materials.size();

	data.meshes = &mesh;
	data.meshes_count = 1;

	data.nodes = &node;
	data.nodes_count = 1;

	data.scenes = &scene;
	data.scenes_count = 1;

	// write JSON and binary blob

	std::string json;
	if (cgltf_size expected = cgltf_write(&opts, nullptr, 0, &data)) {
		json.resize(expected);
		cgltf_size actual = cgltf_write(&opts, json.data(), expected, &data);
		if (expected != actual) {
			std::cerr << "cgltf_write error: returned " << actual << " bytes written, expected " << expected << std::endl;
			return false;
		}
	}

	if (FILE* f = std::fopen(gltf_path.generic_string().c_str(), "w")) {
		std::fwrite(json.data(), json.size() - 1, 1, f);
		std::fclose(f);
		std::cout << "Written " << gltf_path << std::endl;
	}
	else
		return false;

	if (FILE* f = std::fopen(bin_path.generic_string().c_str(), "wb")) {
		std::fwrite(blob.data(), byte_size(blob), 1, f);
		std::fclose(f);
		std::cout << "Written " << bin_path << std::endl;
	}
	else
		return false;

	return true;
}

