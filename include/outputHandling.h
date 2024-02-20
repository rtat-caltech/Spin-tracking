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
	OutputHandler(const options OPT, char* outputName) {
		opt = OPT;
		outputName = outputName;
		std::string ext = std::string(fs::path(outputName).extension());
		use_hdf5 = boost::iequals(ext, ".hdf5") || boost::iequals(ext, ".h5");
		if (use_hdf5) {
			f5 = new H5::H5File(outputName, H5F_ACC_TRUNC);
		} else {
			f = fopen(outputName, "wb");
		}
		nsave = (int) ((opt.tf - opt.t0)/opt.ioutInt) + 1;
	}
	// It is assumed that the number of items to write = # of particles
	template <typename T> void write(T* data, char* datasetName) {
		if (use_hdf5) {
			/*
			DataType myType = get_h5_type(data);
			hsize_t fdim[] = {opt.numParticles, nsave};
			DataSpace fspace(2, fdim);
			if (!pathExists(f5->getId(), datasetName)) {
				// Create Dataset
				DataSet* dset = new DataSet(f5->createDataSet(datasetName, myType, fspace));
				//IntType int_type(PredType::NATIVE_INT);
				DataSpace att_space(H5S_SCALAR);
				Attribute att = dset->createAttribute("nt", PredType::NATIVE_INT, att_space);
				int nt0;
				att.write(PredType::NATIVE_INT, &nt0);
				delete dset;
			}
			*/
			/*
			DataSet* dataset = new DataSet(f5->openDataSet(datasetName));
			// Add row
			Attribute attr = dataset->openAttribute("nt");
			int nt;
			attr.read(PredType::NATIVE_INT, &nt);
			nt = nt+1;
			attr.write(PredType::NATIVE_INT, &nt);
			
			hsize_t count_f[2] = {1, opt.numParticles};
			hsize_t start_f[2] = {nt, 0};
			fspace.selectHyperslab(H5S_SELECT_SET, count_f, start_f);
			hsize_t dim_m[] = {opt.numParticles};
			//hsize_t count_m[] = {opt.numParticles};
			//hsize_t start_m[] = {0};
			DataSpace mspace(1, dim_m);
			//mspace.selectHypevectorrslab(H5S_SELECT_SET, count_m, start_m);			
			dataset->write(data, myType, mspace, fspace);
			delete dataset;
			*/
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
		return PredType::NATIVE_UINT;
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
