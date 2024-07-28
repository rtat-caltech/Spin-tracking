#include "../include/logger.h"

Logger::Logger(const options OPT, int n_records) {
	opt = OPT;
	nsave = opt.stopTimes.size();
	// n_records: number of states to save per time point (i.e. for 1024 particles, 1024).
	npart = n_records;
}

BinaryLogger::BinaryLogger(const options OPT, int n_records) : Logger(OPT, n_records) {}

void BinaryLogger::openFile(const char* outputName) {
	f = fopen(outputName, "wb");
}

void BinaryLogger::writeSnapshot(_PREC* t, coords* pos, coords* v, coords* S,
							   int* failureState, size_t* n_coll, size_t* n_bounce,
							   size_t* n_steps) {
	write<_PREC>(t);
	write<coords>(pos);
	write<coords>(v);
	write<coords>(S);
	write<int>(failureState);
	write<size_t>(n_coll);
	write<size_t>(n_bounce);
	write<size_t>(n_steps);
}

void BinaryLogger::writeOptions() {
	fwrite(&opt, sizeof(options), 1, f);
}

void BinaryLogger::writeSingle(string name, coords value) {
	fwrite(&value, sizeof(coords), 1, f);
}

void BinaryLogger::writeSpin(_PREC t, coords S) {
	fwrite(&t, sizeof(_PREC), 1, f);
	fwrite(&S, sizeof(coords), 1, f);
}

void BinaryLogger::writeInt(string name, int value) {
	fwrite(&value, sizeof(int), 1, f);
}

template <typename T>
void BinaryLogger::write(T* data) {
	fwrite(data, sizeof(T), npart, f);
}

BinaryLogger::~BinaryLogger() {
	fclose(f);
	delete f;
}

#ifdef USEHDF5

using namespace H5;

HDF5Logger::HDF5Logger(const options OPT, int n_records) : Logger(OPT, n_records) {}

void HDF5Logger::openFile(const char* outputName) {
	f5 = new H5File(outputName, H5F_ACC_TRUNC);
}

void HDF5Logger::writeSnapshot(_PREC* t, coords* pos, coords* v, coords* S,
							   int* failureState, size_t* n_coll, size_t* n_bounce,
							   size_t* n_steps) {
	write<_PREC>(t, "Time");
	write<coords>(pos, "Position");
	write<coords>(v, "Velocity");
	write<coords>(S, "Spin");
	write<int>(failureState, "Failure State");
	write<size_t>(n_coll, "Num Collisions");
	write<size_t>(n_bounce, "Num Bounces");
	write<size_t>(n_steps, "Num Steps");
}

void HDF5Logger::writeSpin(_PREC t, coords S) {
	write<_PREC>(&t, "Time");
	write<coords>(&S, "Spin");
}

void HDF5Logger::writeSingle(string name, coords value) {
	writeOption<coords>(name, value);
}

void HDF5Logger::writeInt(string name, int value) {
	writeOption<int>(name, value);
}

void HDF5Logger::writeOptions() {
	string coord_names[] = {"B0", "E", "L", "yi", "Gx", "Gy", "Gz", "testNoiseAmp", "testNoiseFreq"};
	coords coord_values[] = {opt.B0, opt.E, opt.L, opt.yi, opt.Gx, opt.Gy, opt.Gz, opt.noiseAmplitudes, opt.noiseFrequencies};
	massWriteOptions<coords>(coord_names, coord_values, sizeof(coord_values)/sizeof(coords));
	string double_names[] = {
		"m", "t0", "tf", "rtol", "atol",
		"beta", "uround", "safe", "fac1", "fac2",
		"hmax", "hmin", "h", "T", "sqrtKT_m",
		"tc", "gamma", "V", "a", "w",
		"swapStepSize", "maxPosStep", "ioutInt", "diffuse"
	};
	_PREC double_values[] = {
		opt.m, opt.t0, opt.tf, opt.rtol, opt.atol,
		opt.beta, opt.uround, opt.safe, opt.fac1, opt.fac2,
		opt.hmax, opt.hmin, opt.h, opt.T, opt.sqrtKT_m,
		opt.tc, opt.gamma, opt.V, opt.a, opt.w,
		opt.swapStepSize, opt.maxPosStep, opt.ioutInt, opt.diffuse
	};
	massWriteOptions<double>(double_names, double_values, sizeof(double_values)/sizeof(double));

	string int_names[] = {"integratorType", "numParticles", "numPerGPUBlock", "iout"};
	int int_values[] = {opt.integratorType, opt.numParticles, opt.numPerGPUBlock, opt.iout};
	massWriteOptions<int>(int_names, int_values, sizeof(int_values)/sizeof(int));

	string bool_names[] = {"gas_coll", "gravity", "fixedStepSize", "keepStepSize"};
	bool bool_values[] = {opt.gas_coll, opt.gravity, opt.fixedStepSize, opt.keepStepSize};
	massWriteOptions<bool>(bool_names, bool_values, sizeof(bool_values)/sizeof(bool));

	writeOption<char>("dist", opt.dist);
	writeOption<unsigned int>("nmax", opt.nmax);
	writeOption<unsigned int>("seed", opt.seed);
}

template <typename T>
void HDF5Logger::writeOption(string name, T value) {
	DataSpace att_space(H5S_SCALAR);
	DataType myType = getH5Type(&value);
	Attribute attr = f5->createAttribute(name, myType, att_space);
	attr.write(myType, &value);
}

template <typename T>
void HDF5Logger::massWriteOptions(string* names, T* values, int n) {
	for (int i = 0; i < n; i++) {
		writeOption<T>(names[i], values[i]);
	}
}


template <typename T>
void HDF5Logger::write(T* data, const char* datasetName) {
	DataType myType = getH5Type(data);
	hsize_t fdim[] = {nsave, npart};
	DataSpace fspace(2, fdim);
	if (!pathExists(f5->getId(), datasetName)) {
		// Create Dataset
		DataSet* dset = new DataSet(f5->createDataSet(datasetName, myType, fspace));
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
	hsize_t count_f[2] = {1, npart};
	hsize_t start_f[2] = {nt, 0};
	fspace.selectHyperslab(H5S_SELECT_SET, count_f, start_f);
	hsize_t dim_m[] = {npart};
	DataSpace mspace(1, dim_m);
	dataset->write(data, myType, mspace, fspace);

	// Increment row counter
	nt = nt+1;
	attr.write(PredType::NATIVE_INT, &nt);

	dataset->close();
	delete dataset;
}

DataType HDF5Logger::getH5Type(coords* data) {
	CompType mtype(sizeof(coords));
	mtype.insertMember(MEMBER1, HOFFSET(coords, x), PredType::NATIVE_DOUBLE);
	mtype.insertMember(MEMBER2, HOFFSET(coords, y), PredType::NATIVE_DOUBLE);
	mtype.insertMember(MEMBER3, HOFFSET(coords, z), PredType::NATIVE_DOUBLE);
	return mtype;
}
DataType HDF5Logger::getH5Type(double* data) {
	return PredType::NATIVE_DOUBLE;
}
DataType HDF5Logger::getH5Type(int* data) {
	return PredType::NATIVE_INT;
}
DataType HDF5Logger::getH5Type(size_t* data) {
	return PredType::NATIVE_ULONG;
}
DataType HDF5Logger::getH5Type(unsigned int* data) {
	return PredType::NATIVE_UINT;
}
DataType HDF5Logger::getH5Type(bool* data) {
	return PredType::NATIVE_UCHAR;
}
DataType HDF5Logger::getH5Type(char* data) {
	return PredType::NATIVE_UCHAR;
}

HDF5Logger::~HDF5Logger() {
	f5->close();
	delete f5;
}

bool pathExists(hid_t id, const std::string& path) {
	return H5Lexists(id, path.c_str(), H5P_DEFAULT) > 0;
}

#endif

std::unique_ptr<Logger> createLogger(options opt, const char* outputName, int n_records) {
	std::string ext = fs::path(outputName).extension().string();
	bool use_hdf5 = boost::iequals(ext, ".hdf5") || boost::iequals(ext, ".h5");
	std::unique_ptr<Logger> log;
	if (use_hdf5) {
#if USEHDF5
		log = std::unique_ptr<Logger>(new HDF5Logger(opt, n_records));
#else
		throw runtime_error("HDF5 saving is not enabled, but output file has hdf5 extension.");
#endif
	} else {
		log = std::unique_ptr<Logger>(new BinaryLogger(opt, n_records));
	}
	log->openFile(outputName);
	log->writeOptions();
	return log;
}
