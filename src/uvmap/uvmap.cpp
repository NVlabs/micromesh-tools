/* uMeshTools: a tool for the creation and optimization of micromeshes
 * 
 * Developed by Andrea Maggiordomo, University of Milan, 2021
 * 
 * Micromeshes are intellectual property of NVIDIA CORPORATION
 */

#include "mesh_io.h"

#include "micro.h"
#include "adjacency.h"
#include "arap.h"
#include "rectangle_packer.h"
#include "aabb.h"

#include "utils.h"

#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cerr << "Usage: " << fs::path(argv[0]).filename().string() << " OBJ" << std::endl;
		std::cerr << "    OBJ the input .obj mesh" << std::endl;
		return -1;
	}

	fs::path input_mesh(argv[1]);

	MatrixX V;
	MatrixX VT;
	MatrixX VN;
	MatrixXi F;
	MatrixXi FT;
	MatrixXi FN;
	read_obj(input_mesh.string(), V, F, VT, FT, VN, FN);

	Assert(V.rows() == VN.rows());

	// generate a non-subdivided micromesh (used as reference for the 'base ARAP')
	SubdivisionMesh base;
	base.compute_mesh_structure(V, F, 0);
	// setup dummy micro-displacements
	std::vector<std::vector<Scalar>> dummy_displacements;
	for (int i = 0; i < F.rows(); ++i) {
		// each base face has only 3 zero displacements
		dummy_displacements.push_back({ 0, 0, 0 });
	}
	base.compute_micro_displacements(MatrixX::Constant(V.rows(), 3, 0), F, dummy_displacements);

	MatrixX UV_init = MatrixX::Constant(V.rows(), 2, 0);

	Box3 box;
	for (int i = 0; i < V.rows(); ++i) {
		box.add(V.row(i));
	}

	std::cout << "Box extents are " << box.diagonal().transpose() << std::endl;

	int dim = box.min_extent();

	for (int i = 0; i < V.rows(); ++i) {
		UV_init(i, 0) = V(i, (dim + 1) % 3);
		UV_init(i, 1) = V(i, (dim + 2) % 3);
	}

	// optimize UV atlas wrt base
	MatrixX UV_base = UV_init;
	ARAP arap_base(base, V, F);
	arap_base.solve(UV_base, 1000);
	
	// pack
	VFAdjacency VF = compute_adjacency_vertex_face(V, F);
	std::vector<Chart> charts = extract_charts(UV_base, F, VF);
	MatrixX UV_packed = pack_charts(charts, 2048, UV_base, F, VF);

	write_obj("uvmapped_" + fs::path(argv[1]).filename().string(), V, F, UV_packed, F, VN, F);

	return 0;
}

