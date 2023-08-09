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

#include "gui_view.h"
#include "flip.h"
#include "color.h"
#include "clean.h"
#include "mesh_utils.h"
#include "direction_field.h"
#include "mesh_io.h"
#include "mesh_io_gltf.h"
#include "mesh_io_bary.h"

#include <imgui.h>

#include <nfd.h>

#ifdef USE_SINGLE_PRECISION
static const ImGuiDataType ImGuiScalarType = ImGuiDataType_Float;
#else
static const ImGuiDataType ImGuiScalarType = ImGuiDataType_Double;
#endif

namespace ImGui {
	bool InputScalar(const char* label, Scalar* v, Scalar step, Scalar step_fast, const char* format, ImGuiInputTextFlags flags = 0)
	{
		return InputScalar(label, ImGuiScalarType, v, &step, &step_fast, format, flags);
	}
}


// custom imgui 'framed group' thingy, taken from  https://github.com/ocornut/imgui/issues/1496
static void BeginGroupPanel(const char* name, const ImVec2& size = ImVec2(0.0f, 0.0f));
static void EndGroupPanel();

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"


static ImVector<ImRect> s_GroupPanelLabelStack;

static void BeginGroupPanel(const char* name, const ImVec2& size)
{
	ImGui::BeginGroup();

	auto cursorPos = ImGui::GetCursorScreenPos();
	auto itemSpacing = ImGui::GetStyle().ItemSpacing;
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

	auto frameHeight = ImGui::GetFrameHeight();
	ImGui::BeginGroup();

	ImVec2 effectiveSize = size;
	if (size.x < 0.0f)
		effectiveSize.x = ImGui::GetContentRegionAvailWidth();
	else
		effectiveSize.x = size.x;
	ImGui::Dummy(ImVec2(effectiveSize.x, 0.0f));

	ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
	ImGui::SameLine(0.0f, 0.0f);
	ImGui::BeginGroup();
	ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
	ImGui::SameLine(0.0f, 0.0f);
	ImGui::TextUnformatted(name);
	auto labelMin = ImGui::GetItemRectMin();
	auto labelMax = ImGui::GetItemRectMax();
	ImGui::SameLine(0.0f, 0.0f);
	ImGui::Dummy(ImVec2(0.0, frameHeight + itemSpacing.y));
	ImGui::BeginGroup();

	//ImGui::GetWindowDrawList()->AddRect(labelMin, labelMax, IM_COL32(255, 0, 255, 255));

	ImGui::PopStyleVar(2);

#if IMGUI_VERSION_NUM >= 17301
	ImGui::GetCurrentWindow()->ContentRegionRect.Max.x -= frameHeight * 0.5f;
	ImGui::GetCurrentWindow()->WorkRect.Max.x -= frameHeight * 0.5f;
	ImGui::GetCurrentWindow()->InnerRect.Max.x -= frameHeight * 0.5f;
#else
	ImGui::GetCurrentWindow()->ContentsRegionRect.Max.x -= frameHeight * 0.5f;
#endif
	ImGui::GetCurrentWindow()->Size.x -= frameHeight;

	auto itemWidth = ImGui::CalcItemWidth();
	ImGui::PushItemWidth(ImMax(0.0f, itemWidth - frameHeight));

	s_GroupPanelLabelStack.push_back(ImRect(labelMin, labelMax));
}

static void EndGroupPanel()
{
	ImGui::PopItemWidth();

	auto itemSpacing = ImGui::GetStyle().ItemSpacing;

	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

	auto frameHeight = ImGui::GetFrameHeight();

	ImGui::EndGroup();

	//ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(0, 255, 0, 64), 4.0f);

	ImGui::EndGroup();

	ImGui::SameLine(0.0f, 0.0f);
	ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
	ImGui::Dummy(ImVec2(0.0, frameHeight - frameHeight * 0.5f - itemSpacing.y));

	ImGui::EndGroup();

	auto itemMin = ImGui::GetItemRectMin();
	auto itemMax = ImGui::GetItemRectMax();
	//ImGui::GetWindowDrawList()->AddRectFilled(itemMin, itemMax, IM_COL32(255, 0, 0, 64), 4.0f);

	auto labelRect = s_GroupPanelLabelStack.back();
	s_GroupPanelLabelStack.pop_back();

	ImVec2 halfFrame = ImVec2(frameHeight * 0.25f, frameHeight) * 0.5f;
	ImRect frameRect = ImRect(itemMin + halfFrame, itemMax - ImVec2(halfFrame.x, 0.0f));
	labelRect.Min.x -= itemSpacing.x;
	labelRect.Max.x += itemSpacing.x;
	for (int i = 0; i < 4; ++i)
	{
		switch (i)
		{
			// left half-plane
		case 0: ImGui::PushClipRect(ImVec2(-FLT_MAX, -FLT_MAX), ImVec2(labelRect.Min.x, FLT_MAX), true); break;
			// right half-plane
		case 1: ImGui::PushClipRect(ImVec2(labelRect.Max.x, -FLT_MAX), ImVec2(FLT_MAX, FLT_MAX), true); break;
			// top
		case 2: ImGui::PushClipRect(ImVec2(labelRect.Min.x, -FLT_MAX), ImVec2(labelRect.Max.x, labelRect.Min.y), true); break;
			// bottom
		case 3: ImGui::PushClipRect(ImVec2(labelRect.Min.x, labelRect.Max.y), ImVec2(labelRect.Max.x, FLT_MAX), true); break;
		}

		ImGui::GetWindowDrawList()->AddRect(
			frameRect.Min, frameRect.Max,
			ImColor(ImGui::GetStyleColorVec4(ImGuiCol_Border)),
			halfFrame.x);

		ImGui::PopClipRect();
	}

	ImGui::PopStyleVar(2);

#if IMGUI_VERSION_NUM >= 17301
	ImGui::GetCurrentWindow()->ContentRegionRect.Max.x += frameHeight * 0.5f;
	ImGui::GetCurrentWindow()->WorkRect.Max.x += frameHeight * 0.5f;
	ImGui::GetCurrentWindow()->InnerRect.Max.x += frameHeight * 0.5f;
#else
	ImGui::GetCurrentWindow()->ContentsRegionRect.Max.x += frameHeight * 0.5f;
#endif
	ImGui::GetCurrentWindow()->Size.x += frameHeight;

	ImGui::Dummy(ImVec2(0.0f, 0.0f));

	ImGui::EndGroup();
}

// ------------------------------------------------------------------

static void HelpMarker(const char* desc, const char *str = "(?)")
{
	ImGui::TextDisabled(str);
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

void GUIApplication::_main_menu()
{
	int color_flags = ImGuiColorEditFlags_RGB | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_AlphaPreviewHalf | ImGuiColorEditFlags_AlphaBar;

	float indent_val = 0.f;

	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(gui.layout.menu_width, window.height));
	ImGui::Begin("Menu", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
	ImGui::PushItemWidth(ImGui::GetFontSize() * -10);

	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
	bool input = ImGui::CollapsingHeader("Input Mesh", ImGuiTreeNodeFlags_DefaultOpen);
	ImGui::PopStyleColor();
	if (input) {
		_input_widgets();
		ImGui::Dummy(ImVec2(0, 20.0f));
	}

	//if (file_loaded) {
	//	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
	//	bool pm = ImGui::CollapsingHeader("Proxy-Mesh Generation", ImGuiTreeNodeFlags_DefaultOpen);
	//	ImGui::PopStyleColor();
	//	if (pm) {
	//		//ImGui::Indent(indent_val);
	//		_proxy_mesh_widgets();
	//		ImGui::Dummy(ImVec2(0, 20.0f));
	//		//ImGui::Unindent(-indent_val);
	//	}

	//	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
	//	bool bmg = ImGui::CollapsingHeader("Base-Mesh Generation", ImGuiTreeNodeFlags_DefaultOpen);
	//	ImGui::PopStyleColor();
	//	if (bmg) {
	//		//ImGui::Indent(indent_val);
	//		_base_mesh_generation_widgets();
	//		ImGui::Spacing();
	//		_base_mesh_optimization_widgets();
	//		ImGui::Dummy(ImVec2(0, 20.0f));
	//		//ImGui::Unindent(-indent_val);
	//	}

	//	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
	//	bool mmg = ImGui::CollapsingHeader("Micro-Mesh Generation", ImGuiTreeNodeFlags_DefaultOpen);
	//	ImGui::PopStyleColor();
	//	if (mmg) {
	//		//ImGui::Indent(indent_val);
	//		_micromesh_generation_widgets();
	//		ImGui::Spacing();
	//		_micromesh_optimization_widgets();
	//		ImGui::Dummy(ImVec2(0, 20.0f));
	//		//ImGui::Unindent(-indent_val);
	//	}

	//	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
	//	bool aq = ImGui::CollapsingHeader("Assessment", ImGuiTreeNodeFlags_DefaultOpen);
	//	ImGui::PopStyleColor();
	//	if (aq) {
	//		//ImGui::Indent(indent_val);
	//		_assessment_widgets();
	//		ImGui::Dummy(ImVec2(0, 20.0f));
	//		//ImGui::Unindent(-indent_val);
	//	}

	//	ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
	//	bool sv = ImGui::CollapsingHeader("Save", ImGuiTreeNodeFlags_DefaultOpen);
	//	ImGui::PopStyleColor();
	//	if (sv) {
	//		//ImGui::Indent(indent_val);
	//		_save_widgets();
	//		//ImGui::Unindent(-indent_val);
	//	}
	//}

	ImGui::PopItemWidth();
	ImGui::End();
}


void GUIApplication::_input_widgets()
{
	float w1 = ImGui::GetContentRegionAvailWidth();// *(file_loaded ? 0.6f : 1.0f);
	if (ImGui::Button("Load Micromesh file...", ImVec2(w1, 0))) {
		nfdchar_t* file_path = NULL;
		nfdresult_t result = NFD_OpenDialog("gltf", NULL, &file_path);

		if (result == NFD_OKAY) {
			std::cout << "Opening file " << file_path << std::endl;
			load_mesh(file_path);
			std::free(file_path);
		}
		else if (result == NFD_CANCEL) {
		}
		else {
			std::cerr << "Error: " << NFD_GetError() << std::endl;
		}
	}
	if (file_loaded) {
		//ImGui::SameLine();
		//if (ImGui::Button("Reload", ImVec2(ImGui::GetContentRegionAvailWidth(), 0))) {
		//	std::cout << "Opening file " << s.current_mesh << std::endl;
		//	load_mesh(s.current_mesh.string(), false);
		//}

		//ImGui::BeginDisabled(!gui.proxy.enabled && s.algo.handle->dcb->has_quadrics());
		//ImGui::PushItemWidth(110);
		//ImGui::InputScalar("Error multiplier at borders##input", &gui.base.decimation.dparams.border_error_scaling, 1.0, 10.0, "%.1f");
		//gui.proxy.dparams.border_error_scaling = gui.base.decimation.dparams.border_error_scaling;
		//ImGui::PopItemWidth();
		//ImGui::EndDisabled();
	}
}

#if 0
static std::string sep_decimal(int64_t n)
{
	Assert(n >= 0);
	std::string s = std::to_string(n);
	std::string fmt;
	for (int i = 0; i < (int) s.size(); ++i) {
		if (i > 0 && i % 3 == 0)
			fmt.push_back(',');
		fmt.push_back(s[s.size() - i - 1]);
	}

	std::reverse(fmt.begin(), fmt.end());

	return fmt;
}

void GUIApplication::_layer_widgets()
{
	const int hskip = 120;

	const char* load_label = "Load...";

	const int sep_width = 11;

	bool redraw = false;

	int previous_layer = ro.current_layer;

	ImGui::RadioButton("Input Mesh##layer_select_input", &ro.current_layer, RenderingOptions::RenderLayer_InputMesh);
	ImGui::SameLine(hskip);
	ImGui::AlignTextToFramePadding();
	std::string fn_in = sep_decimal((int)s.hi.F.rows());
	ImGui::Text("%11s F", fn_in.c_str());
	ImGui::SameLine();
	
	ImGui::PushItemWidth(-1);
	redraw |= ImGui::Checkbox("Show border mesh", &ro.render_border);
	ImGui::PopItemWidth();

	if (s.proxy.F.rows() > 0) {
		ImGui::RadioButton("Proxy mesh##layer_select_input", &ro.current_layer, RenderingOptions::RenderLayer_ProxyMesh);
		ImGui::SameLine(hskip);
		ImGui::AlignTextToFramePadding();
		std::string fn_proxy = sep_decimal((int)s.proxy.F.rows());
		ImGui::Text("%11s F", fn_proxy.c_str());
		ImGui::SameLine();

		ImGui::PushItemWidth(-1);
		ImGui::Checkbox("Vertex areas", &gl_mesh_proxy.use_color_attribute);
		ImGui::PopItemWidth();
	}
	else if (ro.current_layer == RenderingOptions::RenderLayer_ProxyMesh) {
		ro.current_layer = RenderingOptions::RenderLayer_InputMesh;
	}

	ImGui::BeginDisabled(!gui.micromesh_exists);
	ImGui::RadioButton("Base Mesh##select_base", &ro.current_layer, RenderingOptions::RenderLayer_BaseMesh);
	ImGui::SameLine(hskip);
	ImGui::AlignTextToFramePadding();
	std::string fn_base = gui.micromesh_exists ? sep_decimal(s.base.F.rows()) : "-";
	ImGui::Text("%11s F", fn_base.c_str());
	ImGui::SameLine();
	ImGui::EndDisabled();

	ImGui::BeginDisabled(!s.micromesh.is_subdivided());
	ImGui::PushItemWidth(-1);
	bool changed_show_directions = ImGui::Checkbox("Directions##layer_select", &ro.visible_directions);
	ImGui::PopItemWidth();
	ImGui::SameLine();
	ImGui::PushItemWidth(-1);
	bool changed_directions_scale = ImGui::SliderFloat("##layer_select_direction_scale", &ro.displacement_direction_scale, 0.0f, 1.0f, "%.1f");
	ImGui::PopItemWidth();
	ImGui::EndDisabled();

	ImGui::BeginDisabled(!gui.micromesh_exists);
	ImGui::RadioButton("Micromesh##layer_select_micro", &ro.current_layer, RenderingOptions::RenderLayer_MicroMesh);
	ImGui::SameLine(hskip);
	ImGui::AlignTextToFramePadding();
	std::string fn_micro = gui.micromesh_exists ? sep_decimal(s.micromesh.micro_fn) : "-";
	ImGui::Text("%11s F", fn_micro.c_str());
	ImGui::SameLine();
	float x_align = ImGui::GetCursorPosX(); // align slider below
	ImGui::BeginDisabled(!s.micromesh.is_displaced());
	bool changed_show_microdisplacements = ImGui::Checkbox("Displacement vectors##layer_select", &ro.visible_displacements);
	ro.displacement_direction_scale = clamp(ro.displacement_direction_scale, 0.0f, 1.0f);
	if (changed_show_directions || changed_directions_scale)
		_init_gl_directions();
	if (changed_show_microdisplacements)
		_init_gl_displacements();
	ImGui::EndDisabled();
	ImGui::EndDisabled();

	//ImGui::Dummy(ImVec2(1, 0));
	ImGui::Checkbox("Lock view", &gui.lock_view);
	ImGui::SameLine(x_align);
	ImGui::BeginDisabled(!gui.micromesh_exists || !s.micromesh.is_displaced());
	ImGui::PushItemWidth(120);
	redraw |= ImGui::SliderFloat("Displace##micro", &ro.displacement_scale, 0.0f, 1.0f, "%.3f");
	ImGui::PopItemWidth();
	ImGui::EndDisabled();

	redraw |= previous_layer != ro.current_layer;
	redraw |= changed_show_directions;
	redraw |= changed_directions_scale;
	redraw |= changed_show_microdisplacements;

	if (redraw)
		_require_full_draw();
}

void GUIApplication::_rendering_widgets()
{
	if (!file_loaded)
		return;

	bool redraw = false;

	redraw |= ImGui::Checkbox("Wireframe", &ro.wire);
	ImGui::SameLine();
	redraw |= ImGui::Checkbox("Flat shading", &ro.flat);
	//ImGui::SameLine();
	//ImGui::Checkbox("Direction field", &ro.visible_direction_field);

	ImGui::BeginDisabled(!(s.micromesh.micro_fn > 0));
	redraw |= ImGui::Checkbox("Base-mesh edge status (debug)", &ro.visible_op_status);
	ImGui::EndDisabled();

	if (redraw)
		_require_full_draw();
}

void GUIApplication::_proxy_mesh_widgets()
{
	ImGui::Checkbox("Enable", &gui.proxy.enabled);

	if (gui.proxy.enabled) {
		ImGui::Text("Pre-smoothing");

		ImGui::Indent(gui.layout.widget_indentation);

		ImGui::PushItemWidth(120);
		ImGui::InputInt("Laplacian passes##proxy", &gui.proxy.smoothing_iterations, 1, 5);
		ImGui::PopItemWidth();
		gui.proxy.smoothing_iterations = std::max(gui.proxy.smoothing_iterations, 0);

		ImGui::PushItemWidth(120);
		ImGui::InputInt("Anisotropic passes##proxy", &gui.proxy.anisotropic_smoothing_iterations, 1, 5);
		ImGui::SameLine(); HelpMarker("Smooths vertex quadrics to sharpen creases.");
		ImGui::PopItemWidth();
		gui.proxy.anisotropic_smoothing_iterations = std::max(gui.proxy.anisotropic_smoothing_iterations, 0);

		static constexpr Scalar min_strength = 0.0;
		static constexpr Scalar max_strength = 1.0;
		ImGui::PushItemWidth(120);
		ImGui::SliderScalar("Anisotropic strength##proxy", ImGuiScalarType, &gui.proxy.anisotropic_smoothing_weight, &min_strength, &max_strength, "%.3f");
		ImGui::PopItemWidth();
		ImGui::SameLine(); HelpMarker("Blending weight of the smoothed quadric minimizer and the original vertex position.\nPrevents sliding and flips.");

		ImGui::Unindent(gui.layout.widget_indentation);

		static bool advanced_controls = false;
		ImGui::Spacing();

		ImGui::AlignTextToFramePadding();
		ImGui::Text("Decimation params");

		ImGui::SameLine();

		if (ImGui::RadioButton("Same as base##proxy_params", advanced_controls == false))
			advanced_controls = false;

		ImGui::SameLine();

		if (ImGui::RadioButton("Custom##proxy_params", advanced_controls == true))
			advanced_controls = true;

		static const Scalar smin = 0;
		static const Scalar smax = 1;

		ImGui::Indent(gui.layout.widget_indentation);

		if (advanced_controls) {
			ImGui::Checkbox("##vsmooth_flag_proxy", &gui.proxy.dparams.use_vertex_smoothing);
			ImGui::SameLine();
			ImGui::BeginDisabled(!gui.proxy.dparams.use_vertex_smoothing);
			if (ImGui::InputScalar("Vertex smoothing##vsmooth_val_proxy", &gui.proxy.dparams.smoothing_coefficient, 0.05, 0.1, "%.3f"))
				gui.proxy.dparams.smoothing_coefficient = std::max(Scalar(0), gui.proxy.dparams.smoothing_coefficient);
			ImGui::EndDisabled();

			ImGui::Text("Penalty multipliers");
			ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4, 2));
			ImGui::BeginTable("penalty_fct_controls##proxy", 3, ImGuiTableFlags_BordersInnerV);
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::Text("AspectRatio");
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("Visibility");
			ImGui::TableSetColumnIndex(2);
			ImGui::Text("Normals");
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::PushItemWidth(-FLT_MIN);
			ImGui::SliderScalar("##ar_scaling_proxy", ImGuiScalarType, &gui.proxy.dparams.ar_scaling, &smin, &smax, "%.2f");
			ImGui::PopItemWidth();
			ImGui::TableSetColumnIndex(1);
			ImGui::PushItemWidth(-FLT_MIN);
			ImGui::SliderScalar("##vis_scaling_proxy", ImGuiScalarType, &gui.proxy.dparams.visibility_scaling, &smin, &smax, "%.2f");
			ImGui::PopItemWidth();
			ImGui::TableSetColumnIndex(2);
			ImGui::PushItemWidth(-FLT_MIN);
			ImGui::SliderScalar("##n_scaling_proxy", ImGuiScalarType, &gui.proxy.dparams.normals_scaling, &smin, &smax, "%.2f");
			ImGui::PopItemWidth();
			ImGui::EndTable();
			ImGui::PopStyleVar();
		}

		ImGui::Text("Stopping criteria");

		if (advanced_controls) {
			ImGui::Checkbox("##error_flag_proxy", &gui.proxy.dparams.bound_geometric_error);
			ImGui::SameLine();
			ImGui::BeginDisabled(!gui.proxy.dparams.bound_geometric_error);
			ImGui::InputScalar("Max error##proxy", &gui.proxy.dparams.max_relative_error, 0.01, 0.1, "%.3f");
			ImGui::SameLine(); HelpMarker("As a fraction of the average input edge length");
			ImGui::EndDisabled();
			gui.proxy.dparams.max_relative_error = std::max(Scalar(0), gui.proxy.dparams.max_relative_error);

			static constexpr double min_ar = 0.0;
			static constexpr double max_ar = 1.0;

			ImGui::Checkbox("##bound_ar_flag_proxy", &gui.proxy.dparams.bound_aspect_ratio);
			ImGui::SameLine();
			ImGui::BeginDisabled(!gui.proxy.dparams.bound_aspect_ratio);
			ImGui::SliderScalar("Min aspect ratio##proxy", ImGuiScalarType, &gui.proxy.dparams.min_aspect_ratio, &min_ar, &max_ar,
				"%.3f");
			ImGui::EndDisabled();
		}

		// if not custom, copy dparams from base
		if (!advanced_controls)
			gui.proxy.dparams = gui.base.decimation.dparams;

		ImGui::BeginDisabled(true); // proxy decimation MUST specify the target face count
		ImGui::Checkbox("##fn_flag_proxy", &gui.proxy.fn_flag);
		ImGui::EndDisabled();
		ImGui::SameLine();
		if (ImGui::InputInt("Min #faces##proxy", &gui.proxy.fn_target, 1000, 10000)) {
			gui.proxy.fn_target = std::max(1, gui.proxy.fn_target);
			gui.proxy.target_reduction_factor = std::max(1, (int)std::round(s.hi.F.rows() / (Scalar)gui.proxy.fn_target));
		}

		ImGui::Dummy(ImVec2(19, 0));
		ImGui::SameLine();
		ImGui::AlignTextToFramePadding();
		ImGui::Text("or 1 :"); ImGui::SameLine();
		if (ImGui::InputInt("Reduction##target_reduction_factor_proxy", &gui.proxy.target_reduction_factor, 1, 5)) {
			gui.proxy.target_reduction_factor = std::max(1, gui.proxy.target_reduction_factor);
			gui.proxy.fn_target = s.hi.F.rows() / (Scalar)gui.proxy.target_reduction_factor;
		}


		if (ImGui::Button("Generate proxy-mesh", ImVec2(ImGui::GetContentRegionAvailWidth(), 0))) {
			_log_init_proxy();
		}
		
		ImGui::Unindent(gui.layout.widget_indentation);
	}
}

static void DrawStatusColorCue(bool guard, uint32_t color)
{
	if (guard) {
		ImGui::AlignTextToFramePadding();
		float sy = ImGui::GetTextLineHeight() + 2 * ImGui::GetStyle().FramePadding.y;
		float sx = 0.75 * sy;
		ImVec2 p = ImGui::GetCursorScreenPos();
		ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + sx, p.y + sy), color, sy / 8.0f);
		ImGui::Dummy(ImVec2(sx, sy));
		ImGui::SameLine();
	}
}

void GUIApplication::_base_mesh_generation_widgets()
{
	static const Scalar smin = 0;
	static const Scalar smax = 1;

	ImGui::Text("Decimation params");

	ImGui::Indent(gui.layout.widget_indentation);

	ImGui::Checkbox("##vsmooth_flag", &gui.base.decimation.dparams.use_vertex_smoothing);
	ImGui::SameLine();
	ImGui::BeginDisabled(!gui.base.decimation.dparams.use_vertex_smoothing);
	if (ImGui::InputScalar("Vertex smoothing##vsmooth_val", &gui.base.decimation.dparams.smoothing_coefficient, 0.05, 0.1, "%.3f"))
		gui.base.decimation.dparams.smoothing_coefficient = std::max(Scalar(0), gui.base.decimation.dparams.smoothing_coefficient);
	ImGui::EndDisabled();

	ImGui::Text("Penalty multipliers");
	ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4, 2));
	ImGui::BeginTable("penalty_fct_controls", 3, ImGuiTableFlags_BordersInnerV);
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::Text("AspectRatio");
	ImGui::TableSetColumnIndex(1);
	ImGui::Text("Visibility");
	ImGui::TableSetColumnIndex(2);
	ImGui::Text("Normals");
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::PushItemWidth(-FLT_MIN);
	ImGui::SliderScalar("##ar_scaling_base", ImGuiScalarType, &gui.base.decimation.dparams.ar_scaling, &smin, &smax, "%.2f");
	ImGui::PopItemWidth();
	ImGui::TableSetColumnIndex(1);
	ImGui::PushItemWidth(-FLT_MIN);
	ImGui::SliderScalar("##vis_scaling_base", ImGuiScalarType, &gui.base.decimation.dparams.visibility_scaling, &smin, &smax, "%.2f");
	ImGui::PopItemWidth();
	ImGui::TableSetColumnIndex(2);
	ImGui::PushItemWidth(-FLT_MIN);
	ImGui::SliderScalar("##n_scaling_base", ImGuiScalarType, &gui.base.decimation.dparams.normals_scaling, &smin, &smax, "%.2f");
	ImGui::PopItemWidth();
	ImGui::EndTable();
	ImGui::PopStyleVar();

	ImGui::Text("Stopping criteria");

	if (ro.visible_op_status) {
		DrawStatusColorCue(ro.visible_op_status, operations.colors[Decimation::OpStatus_FailTopology]);
		ImGui::AlignTextToFramePadding();
		ImGui::Text("Topology (preserve manifoldness)");
	
		DrawStatusColorCue(ro.visible_op_status, operations.colors[Decimation::OpStatus_FailNormals]);
		ImGui::AlignTextToFramePadding();
		ImGui::Text("Face normals (no folding)");
	}

	DrawStatusColorCue(ro.visible_op_status, operations.colors[Decimation::OpStatus_FailGeometricError]);
	ImGui::Checkbox("##error_flag_base", &gui.base.decimation.dparams.bound_geometric_error);
	ImGui::SameLine();
	ImGui::BeginDisabled(!gui.base.decimation.dparams.bound_geometric_error);
	ImGui::InputScalar("Max error##base", &gui.base.decimation.dparams.max_relative_error, 0.01, 0.1, "%.3f");
	ImGui::SameLine(); HelpMarker("As a fraction of the average input edge length");
	ImGui::EndDisabled();
	gui.base.decimation.dparams.max_relative_error = std::max(Scalar(0), gui.base.decimation.dparams.max_relative_error);

	static constexpr double min_ar = 0.0;
	static constexpr double max_ar = 1.0;

	DrawStatusColorCue(ro.visible_op_status, operations.colors[Decimation::OpStatus_FailAspectRatio]);
	ImGui::Checkbox("##bound_ar_flag_base", &gui.base.decimation.dparams.bound_aspect_ratio);
	ImGui::SameLine();
	ImGui::BeginDisabled(!gui.base.decimation.dparams.bound_aspect_ratio);
	ImGui::SliderScalar("Min aspect ratio##base", ImGuiScalarType, &gui.base.decimation.dparams.min_aspect_ratio, &min_ar, &max_ar,
		"%.3f");
	ImGui::EndDisabled();

	DrawStatusColorCue(ro.visible_op_status, operations.colors[Decimation::OpStatus_Unknown]);
	ImGui::Checkbox("##fn_flag_base", &gui.base.decimation.fn_flag);
	ImGui::SameLine();
	ImGui::BeginDisabled(!gui.base.decimation.fn_flag);
	if (ImGui::InputInt("Min #faces##base", &gui.base.decimation.fn_target, 1000, 10000)) {
		gui.base.decimation.fn_target = std::max(1, gui.base.decimation.fn_target);
		gui.base.decimation.target_reduction_factor = std::max(1, (int)std::round(s.hi.F.rows() / (Scalar)gui.base.decimation.fn_target));
	}

	DrawStatusColorCue(ro.visible_op_status, 0); // transparent
	ImGui::Dummy(ImVec2(19, 0));
	ImGui::SameLine();
	ImGui::AlignTextToFramePadding();
	ImGui::Text("or 1 :"); ImGui::SameLine();
	if (ImGui::InputInt("Reduction##target_reduction_factor", &gui.base.decimation.target_reduction_factor, 1, 5)) {
		gui.base.decimation.target_reduction_factor = std::max(1, gui.base.decimation.target_reduction_factor);
		gui.base.decimation.fn_target = s.hi.F.rows() / (Scalar)gui.base.decimation.target_reduction_factor;
	}
	ImGui::EndDisabled();

	if (ImGui::Button("Generate base-mesh", ImVec2(ImGui::GetContentRegionAvailWidth(), 0))) {
		bool use_proxy = gui.proxy.enabled && s.proxy.F.rows() > 0;
		_log_decimate();
	}

	ImGui::Unindent(gui.layout.widget_indentation);
}

void GUIApplication::_base_mesh_optimization_widgets()
{
	constexpr int PosMode_LS = 0;
	constexpr int PosMode_Quadric = 1;
	constexpr int PosMode_ClearSmoothing = 2;

	static int current_pos_mode = 0;

	ImGui::Text("Re-optimize");

	ImGui::Indent(gui.layout.widget_indentation);

	ImGui::BeginDisabled(!gui.micromesh_exists);

	if (ImGui::Button("Topology##base_opt", ImVec2(ImGui::GetContentRegionAvailWidth() * 0.35f, 0))) {
		_log_reset_tessellation_offsets();
		_log_optimize_base_topology();
	}
	ImGui::SameLine();
	ImGui::PushItemWidth(-60);
	ImGui::Dummy(ImVec2(1, 0));
	ImGui::PopItemWidth();

	ImGui::SameLine(275);
	HelpMarker("To manually tweak the base mesh topology:\n"
		"SHIFT + RIGHT CLICK to flip an edge\n"
		"ALT   + RIGHT CLICK to split an edge",
		"(tweak?)");

	if (ImGui::Button("Positions", ImVec2(ImGui::GetContentRegionAvailWidth() * 0.35f, 0))) {
		if (current_pos_mode == PosMode_LS) {
			_log_optimize_base_positions(cmdargs::optimize_base_positions::modes::least_squares);
		}
		else if (current_pos_mode == PosMode_Quadric) {
			_log_optimize_base_positions(cmdargs::optimize_base_positions::modes::reprojected_quadrics);
		}
		else if (current_pos_mode == PosMode_ClearSmoothing) {
			_log_optimize_base_positions(cmdargs::optimize_base_positions::modes::clear_smoothing_term);
		}
	}

	ImGui::SameLine();
	ImGui::PushItemWidth(-60);
	ImGui::Combo("##positions_mode", &current_pos_mode, "Least squares\0Re-projected quadrics\0Clear smoothing\0\0");
	ImGui::PopItemWidth();

	ImGui::SameLine(275);
	HelpMarker("To manually tweak a base vertex position:\n"
		"ALT + LEFT CLICK on a base vertex to drag it over the view plane",
		"(tweak?)");

	ImGui::BeginDisabled(current_displacement_mode == DisplacementMode_Vector);
	gui.directions.changed = ImGui::Button("Directions##subdivision", ImVec2(ImGui::GetContentRegionAvailWidth() * 0.35f, 0));

	ImGui::SameLine();
	ImGui::PushItemWidth(-60);
	ImGui::Combo("##subdivision_directions_popdown", &gui.directions.current_type, gui.directions.types);
	ImGui::PopItemWidth();

	ImGui::SameLine(275);
	HelpMarker("To manually tweak a displacement direction:\n"
		"CTRL + RIGHT CLICK on a base vertex to set it toward the eye",
		"(tweak?)");

	if (gui.directions.changed)
		_log_set_displacement_dirs();

	ImGui::EndDisabled();

	ImGui::EndDisabled();
	
	ImGui::Unindent(gui.layout.widget_indentation);
}

static void microface_warning(int ufn)
{
	if (ufn > 10000000) {
		ImGui::AlignTextToFramePadding();
		ImGui::SameLine();
		ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
		ImGui::Text("[!]");
		ImGui::PopStyleColor();
	}
}

void GUIApplication::_micromesh_generation_widgets()
{
	ImGui::Text("Tessellation Options");
	ImGui::SameLine(275);
	HelpMarker("To manually tweak the local tessellation level:\n"
		"SHIFT + CLICK on a face to increase its level\n"
		"CTRL  + CLICK to decrease it",
		"(tweak?)");

	ImGui::Indent(gui.layout.widget_indentation);

	ImGui::Combo("Mode##tessellation", &gui.tessellate.current_mode, gui.tessellate.mode);
	
	gui.tessellate.ufn = gui.tessellate.microexpansion * s.hi.F.rows();

	bool ufn_changed = false;
	bool uexp_changed = false;

	switch (gui.tessellate.current_mode) {
	case 0:
		{
			ImGui::RadioButton("##ufn_auto_adaptive", &gui.tessellate.adaptive.target, gui.tessellate.TARGET_PRIMITIVE_COUNT);
			
			ImGui::BeginDisabled(gui.tessellate.adaptive.target != gui.tessellate.TARGET_PRIMITIVE_COUNT);
				ImGui::SameLine();
				ufn_changed = ImGui::InputInt("Microfaces##auto_adaptive", &gui.tessellate.ufn, 1000, 10000);
				microface_warning(gui.tessellate.ufn);
				ImGui::Dummy(ImVec2(20, 0)); ImGui::SameLine(); ImGui::AlignTextToFramePadding(); ImGui::Text("or");
				ImGui::SameLine();
				uexp_changed = ImGui::InputScalar("Microexpansion##auto_adaptive", &gui.tessellate.microexpansion, 0.1, 1.0, "%.2f");
				ImGui::SameLine();
				HelpMarker("Relative to the # of input faces");
			ImGui::EndDisabled();

			ImGui::RadioButton("##error_auto_adaptive", &gui.tessellate.adaptive.target, gui.tessellate.TARGET_ERROR);
			ImGui::BeginDisabled(gui.tessellate.adaptive.target != gui.tessellate.TARGET_ERROR);
				ImGui::SameLine();
				ImGui::SliderFloat("Max Error##auto_adaptive", &gui.tessellate.max_error, 0.001, 5, "%.3f", ImGuiSliderFlags_Logarithmic);
			ImGui::EndDisabled();
		}
		break;
	case 1:
		ImGui::RadioButton("##level_fixed_level", &gui.tessellate.constant.target, gui.tessellate.TARGET_LEVEL);
		ImGui::SameLine();
		ImGui::InputInt("Level##fixed_level", &gui.tessellate.level, 1, 1);

		ImGui::BeginDisabled();
		ImGui::AlignTextToFramePadding(); ImGui::Dummy(ImVec2(20, 0)); ImGui::SameLine(); 
		ImGui::TextUnformatted("-");
		ImGui::AlignTextToFramePadding(); ImGui::Dummy(ImVec2(20, 0)); ImGui::SameLine(); 
		ImGui::TextUnformatted("-");
		ImGui::EndDisabled();

		break;
	case 2:
		{
			ImGui::RadioButton("##ufn_target_fixed_size", &gui.tessellate.uniform.target, gui.tessellate.TARGET_PRIMITIVE_COUNT);
			ImGui::SameLine();
			ufn_changed = ImGui::InputInt("Microfaces##fixed_size", &gui.tessellate.ufn, 1000, 10000);
			microface_warning(gui.tessellate.ufn);
			ImGui::Dummy(ImVec2(20, 0)); ImGui::SameLine(); ImGui::AlignTextToFramePadding(); ImGui::Text("or");
			ImGui::SameLine();
			uexp_changed = ImGui::InputScalar("Microexpansion##fixed_size", &gui.tessellate.microexpansion, 0.1, 1.0, "%.2f");
			ImGui::SameLine();
			HelpMarker("Relative to the # of input faces");

			ImGui::BeginDisabled();
			ImGui::AlignTextToFramePadding(); ImGui::Dummy(ImVec2(20, 0)); ImGui::SameLine(); 
			ImGui::TextUnformatted("-");
			ImGui::EndDisabled();
		}
		break;
	case 3:
		{
			ImGui::RadioButton("##ufn_auto_adaptive2", &gui.tessellate.adaptive2.target, gui.tessellate.TARGET_PRIMITIVE_COUNT);
			
			ImGui::SameLine();
			ufn_changed = ImGui::InputInt("Microfaces##auto_adaptive2", &gui.tessellate.ufn, 1000, 10000);
			microface_warning(gui.tessellate.ufn);
			ImGui::Dummy(ImVec2(20, 0)); ImGui::SameLine(); ImGui::AlignTextToFramePadding(); ImGui::Text("or");
			ImGui::SameLine();
			uexp_changed = ImGui::InputScalar("Microexpansion##auto_adaptive2", &gui.tessellate.microexpansion, 0.1, 1.0, "%.2f");
			ImGui::SameLine();
			HelpMarker("Relative to the # of input faces");

			ImGui::BeginDisabled();
			ImGui::AlignTextToFramePadding(); ImGui::Dummy(ImVec2(20, 0)); ImGui::SameLine(); 
			ImGui::TextUnformatted("-");
			ImGui::EndDisabled();
		}
		break;
	default:
		break;
	}

	if (ufn_changed)
		gui.tessellate.microexpansion = gui.tessellate.ufn / (Scalar)s.hi.F.rows();
	if (uexp_changed)
		gui.tessellate.ufn = gui.tessellate.microexpansion * s.hi.F.rows();

	gui.tessellate.ufn = clamp(gui.tessellate.ufn, 0, std::numeric_limits<int>::max());
	gui.tessellate.microexpansion = clamp(gui.tessellate.microexpansion, Scalar(0), Scalar(20));
	gui.tessellate.level = clamp(gui.tessellate.level, 0, 8);

	//ImGui::BeginDisabled(gui.tessellate.current_mode == 1);
	//ImGui::InputInt("Max level##tessellation", &gui.tessellate.max_level, 1, 1);
	//gui.tessellate.max_level = clamp<int>(gui.tessellate.max_level, 0, 8);
	//ImGui::EndDisabled();

	ImGui::BeginDisabled(!gui.micromesh_exists);
	if (ImGui::Button("Tessellate", ImVec2(ImGui::GetContentRegionAvailWidth(), 0))) {
		_log_reset_tessellation_offsets();
		_log_tessellate();
	}
	ImGui::EndDisabled();

	ImGui::InputInt("quantization bits", &s.quantization_bits, 1, 1);

	ImGui::Unindent(gui.layout.widget_indentation);

	bool displacement_mode_changed = false;

	ImGui::Spacing();
	ImGui::Text("Displace");

	ImGui::Indent(gui.layout.widget_indentation);

	ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5f, 0.5f));
	ImGui::BeginTable("Displacement Mode", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders);
	ImGui::TableNextColumn();
	if (ImGui::Selectable("Ray-cast (scalar)", current_displacement_mode == DisplacementMode_Scalar, 0, ImVec2(0, 20))) {
		current_displacement_mode = current_displacement_mode == DisplacementMode_Scalar ? DisplacementMode_None : DisplacementMode_Scalar;
		displacement_mode_changed = true;
	}
	ImGui::TableNextColumn();
	if (ImGui::Selectable("Project (vector)", current_displacement_mode == DisplacementMode_Vector, 0, ImVec2(0, 20))) {
		current_displacement_mode = current_displacement_mode == DisplacementMode_Vector ? DisplacementMode_None : DisplacementMode_Vector;
		displacement_mode_changed = true;
	}
	ImGui::EndTable();
	ImGui::PopStyleVar();

	ImGui::Unindent(gui.layout.widget_indentation);

	if (gui.micromesh_exists) {
		bool micromesh_needs_displacement = !s.micromesh.is_displaced() && current_displacement_mode != DisplacementMode_None;
		if (micromesh_needs_displacement || displacement_mode_changed || gui.directions.changed) {
			if (current_displacement_mode == DisplacementMode_Scalar) {
				_log_displace();
			}
			if (current_displacement_mode == DisplacementMode_Vector) {
				per_vertex_micro_nearest_projection(s.micromesh, s.base.V, s.base.F, s.hi.V, s.hi.F, s.hi.VN, *s.algo.base_to_input);
				_init_gl_umesh();
			}
		}
	}
}

void GUIApplication::_micromesh_optimization_widgets()
{
	ImGui::BeginDisabled(!s.micromesh.is_displaced());

	ImGui::Text("Re-optimize");

	ImGui::Indent(gui.layout.widget_indentation);

	if (ImGui::Button("Minimize prismoids volume", ImVec2(ImGui::GetContentRegionAvailWidth(), 0)))
		_log_minimize_prismoids();

	ImGui::Unindent(gui.layout.widget_indentation);

	ImGui::EndDisabled();
}

void GUIApplication::_assessment_widgets()
{
	if (!file_loaded)
		return;

	//ImGui::BeginDisabled((quality.current_mode == Quality::DistanceInputToMicro && !gui.micromesh_exists)
	//	|| (quality.current_mode != Quality::None && !s.micromesh.is_displaced()));
	
	if (ImGui::Button("Set color as##quality"))
		_compute_quality();

	//ImGui::EndDisabled();

	ImGui::SameLine();

	ImGui::PushItemWidth(-1);
	bool quality_changed = ImGui::Combo("##quality_type", &quality.current_mode, quality.modes);
	ImGui::PopItemWidth();

	ImGui::BeginDisabled(quality.points.size() == 0);
	bool goto_point = ImGui::Button("Go to n-th worst point##quality");
	ImGui::SameLine();
	ImGui::AlignTextToFramePadding();
	ImGui::Text("n=");
	ImGui::SameLine();
	ImGui::PushItemWidth(80.0f);
	goto_point |= ImGui::InputInt("##quality_goto", &quality.goto_rank, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AlwaysOverwrite);
	ImGui::PopItemWidth();

	quality.goto_rank = std::min((int)quality.points.size(), quality.goto_rank);
	quality.goto_rank = std::max(1, quality.goto_rank);

	if (goto_point)
		set_view_from_quality_rank();

	ImGui::SameLine();
	HelpMarker("Press 1 to 9 to fly to n-th worst point");
	ImGui::EndDisabled();
}

void GUIApplication::_save_widgets()
{
	if (!file_loaded)
		return;

	static ImVec2 gltf_button_size = ImVec2(0.33f * ImGui::GetContentRegionAvailWidth(), 0);

	static bool export_subdivision = false;
	static bool export_displacement_dirs = false;
	static bool export_bary_file = false;

	if (!gui.micromesh_exists) {
		export_subdivision = false;
		export_displacement_dirs = false;
		export_bary_file = false;
	}

	if (!export_subdivision || !export_displacement_dirs)
		export_bary_file = false;

	ImGui::BeginDisabled(!gui.micromesh_exists);
	if (ImGui::Button("As glTF##base", gltf_button_size)) {
		ImGui::OpenPopup("Save glTF");
	}
	ImGui::EndDisabled();

	ImVec2 center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	if (ImGui::BeginPopupModal("Save glTF", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		constexpr int bufsz = 256;
		static char filename[bufsz] = "base_mesh";

		ImGui::Text("Filename:");
		ImGui::PushItemWidth(-1);
		ImGui::InputText("##gltf_filename", filename, bufsz);
		ImGui::PopItemWidth();
		//ImGui::SameLine();
		ImVec2 btn_sz(120, 0);
		if (ImGui::Button("Save##gltf", btn_sz)) {
			VectorXu8 subdivision_bits = export_subdivision ? s.compute_subdivision_bits() : VectorXu8();

			Bary bary;
			if (export_bary_file)
				extract_displacement_bary_data(s.micromesh, &bary, true);

			Bary *bary_arg = (export_displacement_dirs && export_bary_file) ? &bary : nullptr;

			MatrixX V = s.base.V + bary.min_displacement * s.base.VD;
			MatrixX VD = (bary.max_displacement - bary.min_displacement) * s.base.VD;

			GLTFWriteInfo write_info;
			write_info
				.write_faces(&s.base.F)
				.write_vertices(&V)
				.write_normals(&s.base.VN)
				.write_directions(export_displacement_dirs ? &VD : nullptr)
				.write_subdivision_bits(export_subdivision ? &subdivision_bits : nullptr)
				.write_bary(export_bary_file ? &bary : nullptr);
			write_gltf(filename, write_info);
			
			sequence.write_to_file("log_" + std::string(filename) + ".txt");

			ImGui::CloseCurrentPopup();
		}

		ImGui::SameLine();

		if (ImGui::Button("Cancel##gltf", btn_sz))
			ImGui::CloseCurrentPopup();

		ImGui::EndPopup();
	}

	ImGui::SameLine();
	ImGui::BeginGroup();
	ImGui::BeginDisabled(!s.micromesh.is_subdivided());
	ImGui::Checkbox("Subdivision bits", &export_subdivision);
	ImGui::EndDisabled();

	ImGui::BeginDisabled(!s.micromesh.is_displaced());
	ImGui::Checkbox("Displacement directions", &export_displacement_dirs);
	ImGui::BeginDisabled(!export_subdivision || !export_displacement_dirs);
	ImGui::Checkbox(".bary file", &export_bary_file);
	ImGui::EndDisabled();
	ImGui::EndDisabled();

	ImGui::EndGroup();

	if (gltf_button_size.y == 0)
		gltf_button_size.y = ImGui::GetItemRectSize().y;

	ImGui::BeginDisabled(!(gui.micromesh_exists && s.micromesh.is_displaced()));
	if (ImGui::Button("Displaced micromesh as OBJ", ImVec2(ImGui::GetContentRegionAvailWidth(), 0))) {
		ImGui::OpenPopup("Save displaced micromesh as OBJ");
	}
	ImGui::EndDisabled();

	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	if (ImGui::BeginPopupModal("Save displaced micromesh as OBJ", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		constexpr int bufsz = 256;
		static char filename[bufsz] = "displaced.obj";
		ImGui::Text("Filename:");
		ImGui::PushItemWidth(-1);
		ImGui::InputText("##obj_filename", filename, bufsz);
		ImGui::PopItemWidth();

		ImVec2 btn_sz(120, 0);
		if (ImGui::Button("Save##obj", btn_sz)) {
			MatrixX V;
			MatrixXi F;
			s.micromesh.extract_mesh(V, F);
			// remove unreferenced vertices due to decimation flags
			compact_vertex_data(F, V);
			//write_obj(filename, s.micromesh, 1.0);
			write_obj(filename, V, F);
			std::cout << "Written " << filename << std::endl;
			ImGui::CloseCurrentPopup();
		}

		ImGui::SameLine();

		if (ImGui::Button("Cancel##obj", btn_sz))
			ImGui::CloseCurrentPopup();

		ImGui::EndPopup();
	}
}

void GUIApplication::_view_overlay()
{
	if (!file_loaded)
		return;

	ImGui::SetNextWindowPos(ImVec2(gui.layout.menu_width + gui.layout.pad, gui.layout.pad), ImGuiCond_Always);
	ImGui::Begin("View mesh", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

	_layer_widgets();
	ImGui::Separator();
	_rendering_widgets();

	ImGui::End();
}

void GUIApplication::_log_overlay()
{
	if (!file_loaded)
		return;

	ImGui::SetNextWindowSize(ImVec2(450, 200), ImGuiCond_Appearing);
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImVec2 work_pos = viewport->WorkPos;
	ImVec2 work_size = viewport->WorkSize;
 
	ImVec2 window_pos, window_pos_pivot;
	window_pos.x = work_pos.x + work_size.x - gui.layout.pad;
	window_pos.y = work_pos.y + gui.layout.pad;
	window_pos_pivot.x = 1.0f;
	window_pos_pivot.y = 0.0f;
	
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	//ImGui::SetNextWindowPos(ImVec2(gui.layout.pad, gui.layout.pad), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
	ImGui::Begin("Commands log", nullptr, ImGuiWindowFlags_NoResize);

	bool copy = ImGui::Button("Copy to clipboard");
	ImGui::Separator();
	
	ImGui::BeginChild("Log window", ImVec2(0,0), false, ImGuiWindowFlags_HorizontalScrollbar);

	if (copy)
		ImGui::LogToClipboard();

	for (const std::string& s : cmd_strings) {
		ImGui::TextUnformatted(s.c_str());
		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
			ImGui::SetScrollHereY(1.0f);
	}

	for (const SessionCommand& cmd : cmd_queue) {
		std::string next_cmd = cmd.to_string();
		ImGui::TextUnformatted(next_cmd.c_str());
		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
			ImGui::SetScrollHereY(1.0f);
	}

	if (copy)
		ImGui::LogFinish();

	ImGui::EndChild();

	ImGui::End();
}

void GUIApplication::_extra_overlay()
{
	static char export_filename[256] = "view.txt";
	static char import_filename[256] = "view.txt";

	if (!file_loaded || !gui.show_rendering_options)
		return;

	bool redraw = false;

	ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Appearing);
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImVec2 work_pos = viewport->WorkPos;
	ImVec2 work_size = viewport->WorkSize;
 
	ImVec2 window_pos, window_pos_pivot;
	window_pos.x = work_pos.x + work_size.x - gui.layout.pad;
	window_pos.y = work_pos.y + work_size.y - gui.layout.pad;
	window_pos_pivot.x = 1.0f;
	window_pos_pivot.y = 1.0f;
	
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_Appearing);
	ImGui::Begin("Rendering & Misc", nullptr, ImGuiWindowFlags_NoResize);

	ImGui::Checkbox("Parallel proxy decimation", &s.parallel_proxy);
	ImGui::SameLine();
	ImGui::BeginDisabled(!s.parallel_proxy);
	ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
	ImGui::InputInt("nw", &s.nw, 1, 1);
	ImGui::PopItemWidth();
	ImGui::EndDisabled();
	s.nw = std::max(1, s.nw);

	ImGui::Separator();

	if (ImGui::Button("Load base mesh...", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
		nfdchar_t* file_path = NULL;
		nfdresult_t result = NFD_OpenDialog("obj,gltf", NULL, &file_path);

		if (result == NFD_OKAY) {
			std::string base_file_path(file_path);
			_log_load_base_mesh(base_file_path);
			std::free(file_path);
		}
		else if (result == NFD_CANCEL) {
		}
		else {
			std::cerr << "Error: " << NFD_GetError() << std::endl;
		}
	}

	ImGui::Checkbox("Microdisp interpolation", &s.micromesh._interpolate);

	ImGui::Separator();

	static int num_samples = 16;
	if (ImGui::Button("Ambient Occlusion")) {
		_init_gl_hi_colors_from_ambient_occlusion(num_samples);
		gl_mesh_in.use_color_attribute = true;
		_require_full_draw();
	}
	ImGui::SameLine();
	ImGui::InputInt("NumSamples##AO", &num_samples, 1, 2);
	num_samples = std::max(num_samples, 1);

	if (ImGui::Checkbox("Use HI color", &gl_mesh_in.use_color_attribute))
		_require_full_draw();

	ImGui::Separator();

	ImGui::BeginDisabled(!gui.micromesh_exists);
	if (ImGui::Button("Export polyline", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
		std::vector<MatrixX> Vs(s.micromesh.faces.size());
		std::vector<std::vector<int>> Ls(s.micromesh.faces.size());
		for (unsigned i = 0; i < s.micromesh.faces.size(); ++i) {
			s.micromesh.faces[i].extract_boundary_polyline(Vs[i], Ls[i]);
		}
		std::string filename = "Micromesh-polyline.obj";
		write_obj_lines(filename, Vs, Ls);
		std::cout << "Written " << std::quoted(filename) << std::endl;
	}
	ImGui::EndDisabled();
	ImGui::Checkbox("Save envelopes on minimization", &s.save_envelopes);

	ImGui::Separator();

	gui.screenshot.on = ImGui::Button("Screenshot");
	ImGui::SameLine();
	ImGui::InputText("##screenshot_string", gui.screenshot.string, 256);
	ImGui::InputInt("Counter##screenshot_counter", &gui.screenshot.counter, 1, 1);
	ImGui::InputInt("Multiplier##screenshot_counter", &gui.screenshot.multiplier, 1, 1);

	ImGui::Separator();

	bool btn_export = ImGui::Button("Export view");
	ImGui::SameLine();
	ImGui::InputText("##import_view", export_filename, 256);
	if (btn_export)
		control.write(export_filename);

	bool btn_import = ImGui::Button("Import view");
	ImGui::SameLine();
	ImGui::InputText("##export_view", import_filename, 256);
	if (btn_import)
		control.read(import_filename);

	redraw |= btn_import;

	ImGui::Separator();
	ImGui::Combo("Colormap", &gui.color.current_map, gui.color.maps);
	ImGui::InputFloat("max_hi_quality", &quality.max_hi_quality, 0.001f, 0.01f, "%.4f");
	ImGui::Separator();

	redraw |= ImGui::Checkbox("Draw inner umesh", &ro.draw_inner_umesh);

	redraw |= ImGui::SliderFloat("Metallic##ro", &ro.metallic, 0, 1);
	redraw |= ImGui::SliderFloat("Roughness##ro", &ro.roughness, 0, 1);
	redraw |= ImGui::SliderFloat("Ambient##ro", &ro.ambient, 0, 20, "%.2f",ImGuiSliderFlags_Logarithmic);

	redraw |= ImGui::SliderFloat("Shading##ro", &ro.shading_weight, 0, 1);
	redraw |= ImGui::SliderFloat("Wire width##ro", &ro.wire_line_width, 0.01, 4);

	redraw |= ImGui::ColorEdit3("Wire color 1##ro", ro.wire_color.data(), ImGuiColorEditFlags_RGB);
	redraw |= ImGui::ColorEdit3("Wire color 2##ro", ro.wire_color2.data(), ImGuiColorEditFlags_RGB);
	redraw |= ImGui::ColorEdit3("Mesh color##ro", ro.color.data(), ImGuiColorEditFlags_RGB);
	//redraw |= ImGui::ColorEdit3("Light color##ro", ro.lightColor.data(), ImGuiColorEditFlags_RGB);
	redraw |= ImGui::InputFloat("Light intensity##ro", &ro.light_intensity, 1.0f, 2.0f, "%.2f");
	
	redraw |= ImGui::SliderFloat("FOV##ro", &control.camera.fov, 1, 179, "%.1f");

	ImGui::End();

	if (redraw)
		_require_full_draw();
}

#endif

static std::string sep_decimal(int64_t n)
{
	Assert(n >= 0);
	std::string s = std::to_string(n);
	std::string fmt;
	for (int i = 0; i < (int) s.size(); ++i) {
		if (i > 0 && i % 3 == 0)
			fmt.push_back(',');
		fmt.push_back(s[s.size() - i - 1]);
	}

	std::reverse(fmt.begin(), fmt.end());

	return fmt;
}

void GUIApplication::_layer_widgets()
{
	const int hskip = 120;

	const int sep_width = 11;

	bool redraw = false;

	_input_widgets();

	ImGui::BeginDisabled(!file_loaded);
	ImGui::RadioButton("Micromesh##layer_select_micro", &ro.current_layer, RenderingOptions::RenderLayer_MicroMesh);
	ImGui::SameLine(hskip);
	ImGui::AlignTextToFramePadding();
	std::string fn_micro = file_loaded ? sep_decimal(umesh.micro_fn) : "-";
	ImGui::Text("%11s F", fn_micro.c_str());
	ImGui::SameLine();
	float x_align = ImGui::GetCursorPosX(); // align slider below

	//ImGui::Dummy(ImVec2(1, 0));
	ImGui::PushItemWidth(160);
	redraw |= ImGui::SliderFloat("Displace##micro", &ro.displacement_scale, 0.0f, 1.0f, "%.3f");
	ImGui::PopItemWidth();
	ImGui::EndDisabled();

	if (redraw)
		_require_full_draw();
}

void GUIApplication::_rendering_widgets()
{
	//if (!file_loaded)
	//	return;

	bool redraw = false;

	ImGui::BeginDisabled(!file_loaded);
	redraw |= ImGui::Checkbox("Wireframe", &ro.wire);
	//ImGui::SameLine();
	//redraw |= ImGui::Checkbox("Base wire only", &ro.wire_skip_inner_umesh);
	//ImGui::SameLine();
	//redraw |= ImGui::SliderFloat("Width", &ro.wire_line_width, 0.1f, 2.0f, "%.2f");
	//ImGui::Checkbox("Direction field", &ro.visible_direction_field);
	ImGui::EndDisabled();

	if (redraw)
		_require_full_draw();
}

void GUIApplication::_view_overlay()
{
	//if (!file_loaded)
	//	return;

	//ImGui::SetNextWindowPos(ImVec2(gui.layout.menu_width + gui.layout.pad, gui.layout.pad), ImGuiCond_Always);
	ImGui::SetNextWindowPos(ImVec2(0 + gui.layout.pad, gui.layout.pad), ImGuiCond_Always);
	ImGui::Begin("View mesh", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

	_layer_widgets();
	ImGui::Separator();
	_rendering_widgets();

	ImGui::End();
}

void GUIApplication::_draw_gui()
{
//	_main_menu();
	_view_overlay();
	//_log_overlay();
	//_extra_overlay();
}

