#ifndef __OUTPUT_HANDLER_H_INCLUDED__
#define __OUTPUT_HANDLER_H_INCLUDED__

#include <boost/algorithm/string.hpp>
#include "H5Cpp.h"
#include <filesystem>

using namespace H5;
namespace fs = std::filesystem;

const H5std_string MEMBER1( "x" );
const H5std_string MEMBER2( "y" );
const H5std_string MEMBER3( "z" );

bool pathExists(hid_t id, const std::string& path);

class OutputHandler {
public:
	OutputHandler(const options OPT, const char* outputName) {
		opt = OPT;
		outputName = outputName;
		std::string ext = std::string(fs::path(outputName).extension());
		use_hdf5 = boost::iequals(ext, ".hdf5") || boost::iequals(ext, ".h5");
		if (use_hdf5) {
			f5 = new H5::H5File(outputName, H5F_ACC_TRUNC);
			write_options();
		} else {
			f = fopen(outputName, "wb");
		}
		nsave = (int) ((opt.tf - opt.t0)/opt.ioutInt) + 1;
	}
	// It is assumed that the number of items to write = # of particles
	template <typename T> void write(T* data, const char* datasetName) {
		if (use_hdf5) {
			DataType myType = get_h5_type(data);
			hsize_t fdim[] = {nsave, opt.numParticles};
			DataSpace fspace(2, fdim);
			if (!pathExists(f5->getId(), datasetName)) {
				// Create Dataset
				DataSet* dset = new DataSet(f5->createDataSet(datasetName, myType, fspace));
				//IntType int_type(PredType::NATIVE_INT);
				DataSpace att_space(H5S_SCALAR);
				Attribute att = dset->createAttribute("nt", PredType::NATIVE_INT, att_space);
				int nt0 = 0;
				att.write(PredType::NATIVE_INT, &nt0);
				delete dset;
			}			

			DataSet* dataset = new DataSet(f5->openDataSet(datasetName));
			Attribute attr = dataset->openAttribute("nt");
			int nt;
			attr.read(PredType::NATIVE_INT, &nt);
			hsize_t count_f[2] = {1, opt.numParticles};
			hsize_t start_f[2] = {nt, 0};
			fspace.selectHyperslab(H5S_SELECT_SET, count_f, start_f);
			hsize_t dim_m[] = {opt.numParticles};
			//hsize_t count_m[] = {opt.numParticles};
			//hsize_t start_m[] = {0};
			DataSpace mspace(1, dim_m);
			//mspace.selectHypevectorrslab(H5S_SELECT_SET, count_m, start_m);			
			dataset->write(data, myType, mspace, fspace);

			// Increment row counter
			nt = nt+1;
			attr.write(PredType::NATIVE_INT, &nt);

			dataset->close();
			delete dataset;

		} else {
			fwrite(data, sizeof(int), opt.numParticles, f);
		}
	}
	DataType get_h5_type(coords* data) {
		CompType mtype(sizeof(coords));
		mtype.insertMember(MEMBER1, HOFFSET(coords, x), PredType::NATIVE_DOUBLE);
		mtype.insertMember(MEMBER2, HOFFSET(coords, y), PredType::NATIVE_DOUBLE);
		mtype.insertMember(MEMBER3, HOFFSET(coords, z), PredType::NATIVE_DOUBLE);
		return mtype;
	}
	DataType get_h5_type(double* data) {
		return PredType::NATIVE_DOUBLE;
	}
	DataType get_h5_type(int* data) {
		return PredType::NATIVE_INT;
	}
	DataType get_h5_type(size_t* data) {
		return PredType::NATIVE_ULONG;
	}

	void write_options() {
		string coord_names[] = {"B0", "E", "L", "yi", "Gx", "Gy", "Gz"};
		coords coord_values[] = {opt.B0, opt.E, opt.L, opt.yi, opt.Gx, opt.Gy, opt.Gz};
		mass_write_options<coords>(coord_names, coord_values, 7);
		string double_names[] = {
			"m", "t0", "tf", "rtol", "atol",
			"beta", "uround", "safe", "fac1", "fac2",
			"hmax", "hmin", "h", "T", "sqrtKT_m",
			"tc", "gamma", "V", "a", "w",
			"swqpStepSize", "maxPosStep", "ioutInt", "diffuse"
		};
		_PREC double_values[] = {
			opt.m, opt.t0, opt.tf, opt.rtol, opt.atol,
			opt.beta, opt.uround, opt.safe, opt.fac1, opt.fac2,
			opt.hmax, opt.hmin, opt.h, opt.T, opt.sqrtKT_m,
			opt.tc, opt.gamma, opt.V, opt.a, opt.w,
			opt.swapStepSize, opt.maxPosStep, opt.ioutInt, opt.diffuse
		};
		mass_write_options<double>(double_names, double_values, 7);
	}

	template <typename T> void mass_write_options(string* names, T* values, int n) {
		for (int i = 0; i < n; i++) {
			write_option<T>(names[i], values[i]);
		}
	}

	template <typename T> void write_option(string name, T value) {
		DataSpace att_space(H5S_SCALAR);
		DataType myType = get_h5_type(&value);
		Attribute attr = f5->createAttribute(name, myType, att_space);
		attr.write(myType, &value);
	}

	void close() {
		if (use_hdf5) {
			f5->close();
		} else {
			fclose(f);
		}
	}
	
private:
	void write_parameters();
	FILE* f;
	H5File* f5;
	bool use_hdf5;
	options opt;
	char* outputName;
	int nsave;
};

#endif
