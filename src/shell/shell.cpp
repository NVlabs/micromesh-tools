#include "space.h"
#include "mesh_io.h"
#include "micro.h"
#include "tangent.h"
#include "mesh_utils.h"

#include "utils.h"

#include <filesystem>

int main(int argc, char* argv[])
{
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << "HI-RES BASE TOP TARGET_MICROFN" << std::endl;
		return -1;
	}

	// 1. load data (reference mesh, base surface and top surface

	MatrixX hi_V, base_V, top_V;
	MatrixXi hi_F, base_F, top_F;

	Assert(base_V.rows() == top_V.rows());
	Assert(base_F.rows() == top_F.rows());

	std::cout << "Reading hi-res mesh " << argv[1] << "..." << std::endl;
	read_obj(argv[1], hi_V, hi_F);
	std::cout << "Reading base mesh " << argv[2] << "..." << std::endl;
	read_obj(argv[2], base_V, base_F);
	std::cout << "Reading top mesh " << argv[3] << "..." << std::endl;
	read_obj(argv[3], top_V, top_F);

	// 2. compute displacement directions

	MatrixX base_VD = top_V - base_V;

	// 3. compute subdivision level -- just regular subdivision for now, nothing fancy...

	int fn = int(base_F.rows());
	int ufn = std::atoi(argv[4]);
	//Assert(ufn > 0);
	//Assert(ufn < 20000000);

	//// ufn = 2 ^ (2l) * base_F --> log2(ufn) = 2l + log2(base_F) --> l = 0.5 * (log2(ufn) - log2(base_F))
	//uint8_t sl = std::round(0.5 * (std::log2(ufn) - std::log2(fn)));
	//sl = clamp(sl, uint8_t(0), uint8_t(8));

	//{
		Scalar microexpansion = ufn / (Scalar)hi_F.rows();
		
		Scalar subdivision_level = (std::log2(microexpansion) + std::log2(hi_F.rows() / (Scalar)base_F.rows())) / 2;
		Scalar avg_area_inv = 1.0 / average_area(base_V, base_F);
		VectorXu8 base_subdivisions = compute_subdivision_levels_uniform_area(base_V, base_F, avg_area_inv, subdivision_level);
		VectorXi8 base_corrections = VectorXi8::Constant(base_F.rows(), 0);

		VectorXu8 base_flags = adjust_subdivision_levels(base_F, base_subdivisions, base_corrections, RoundMode::Up);
		VectorXu8 subdivision_bits = VectorXu8::Constant(base_F.rows(), 0);
		for (int i = 0; i < base_F.rows(); ++i) {
			subdivision_bits(i) = (uint8_t(base_subdivisions(i) + base_corrections(i)) << 3) | base_flags(i);
		}
	//}

	// 4. subdivide

	std::cout << "Generating micromesh structure..." << std::endl;

	SubdivisionMesh micromesh;
	micromesh.compute_mesh_structure(base_V, base_F, subdivision_bits);

	std::cout << "Micromesh has " << micromesh.micro_fn << " faces" << std::endl;

	// 5. ray-cast
	
	std::cout << "Computing microdisplacements..." << std::endl;

	MatrixX hi_VN = compute_vertex_normals(hi_V, hi_F);

	BVHTree bvh;
	bvh.build_tree(&hi_V, &hi_F, &hi_VN, 32);

	auto bvh_test = [&](const IntersectionInfo& ii) -> bool {
		return ii.d.dot(compute_face_normal(ii.fi, hi_V, hi_F)) >= 0;
	};

	// disable culling and interpolation of microdisplacement around stretched faces
	micromesh._cull = false;
	micromesh._interpolate = false;

	micromesh.compute_micro_displacements(base_V, base_VD, base_F, bvh, bvh_test, {}, MatrixX(), MatrixXi(), MatrixX());

	// 6. export

	std::filesystem::path hires_path(argv[1]);

	std::string outname = "ushell." + hires_path.filename().string();

	std::cout << "Saving file " << outname << "..." << std::endl;

	write_obj(outname, micromesh, 1.0);

	return 0;
}