#ifndef __LOGGER_H_INCLUDED__
#define __LOGGER_H_INCLUDED__

#include <boost/algorithm/string.hpp>
#include "H5Cpp.h"
#include "options.h"
#include "double3.h"
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

bool pathExists(hid_t id, const std::string& path);

class Logger {
public:
	Logger(const options opt, const char* outputName);
	virtual void openFile(const char* outputName) = 0;
	virtual void writeOptions() = 0;
	virtual void writeSnapshot(_PREC* t, coords* pos, coords* v, coords* S,
							   int* failureState, size_t* n_coll, size_t* n_bounce,
							   size_t* n_steps) = 0;
	virtual void writeSingle(string name, coords value) {}; // Used for saving Floquet result
	virtual ~Logger(){};

protected:
	options opt;
	int nsave;
	char* outputName;
};

class BinaryLogger : public Logger {
public:
	BinaryLogger(const options OPT, const char* outputName);
	void openFile(const char* outputName) override;
	void writeOptions() override;
	void writeSnapshot(_PREC* t, coords* pos, coords* v, coords* S,
					   int* failureState, size_t* n_coll, size_t* n_bounce,
					   size_t* n_steps) override;
	void writeSingle(string name, coords value);
	~BinaryLogger() override;
private:
	template <typename T> void write(T* data);
	FILE* f;
};


using namespace H5;

const H5std_string MEMBER1( "x" );
const H5std_string MEMBER2( "y" );
const H5std_string MEMBER3( "z" );

class HDF5Logger : public Logger {
public:
	HDF5Logger(const options OPT, const char* outputName);
	void openFile(const char* outputName) override;	
	void writeOptions() override;
	void writeSnapshot(_PREC* t, coords* pos, coords* v, coords* S,
					   int* failureState, size_t* n_coll, size_t* n_bounce,
					   size_t* n_steps) override;
	void writeSingle(string name, coords value);
	~HDF5Logger() override;

private:
	template <typename T> void write(T* data, const char* datasetName);
	template <typename T> void writeOption(string name, T value);
	template <typename T> void massWriteOptions(string* names, T* values, int n);
	DataType getH5Type(coords* data);
	DataType getH5Type(double* data);
	DataType getH5Type(int* data);
	DataType getH5Type(size_t* data);
	DataType getH5Type(unsigned int* data);
	DataType getH5Type(bool* data);
	DataType getH5Type(char* data);
	H5File* f5;
};

std::unique_ptr<Logger> createLogger(options opt, const char* outputName);

#endif
