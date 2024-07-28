// This script is responsible for parsing the various option parameters that the user can pass in

#include "../include/optionsParser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

std::string removeWhitespace(std::string s){
	s.erase(remove_if(s.begin(), s.end(), [](char c){return isspace(c);}), s.end());
	return s;
}

bool to_bool(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    std::istringstream is(str);
    bool b;
    is >> std::boolalpha >> b;
    return b;
}

std::vector<std::string> splitString(std::string s, std::string delimiter) {
	size_t pos = 0;
	std::string token;
	std::vector<std::string> elements;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		elements.push_back(removeWhitespace(token));
		s.erase(0, pos + delimiter.length());
	}
	token = s.substr(0, pos);
	elements.push_back(removeWhitespace(token));
	return elements;
}

std::vector<std::string> grabElementsInLine(std::string s){
	std::string delimiter = ",";
	return splitString(s, delimiter);
}


void parseLine(options& opt, std::string s){
	//first parse the line into it's different elements
	std::vector<std::string> elements = grabElementsInLine(s);
	//make sure it has enough elements to be useful
	if(elements.size() < 2){
		std::cout<<"Misunderstood input line in options file"<<std::endl;
		std::cout<<"\t"<<s<<std::endl;
		return;
	}
	//now start the wall of comparisons for most things
	if(elements.at(0) == "L"){
		opt.L = {(_PREC)std::stod(elements.at(1)), (_PREC)std::stod(elements.at(2)), (_PREC)std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "yi"){
		opt.yi = {(_PREC)std::stod(elements.at(1)), (_PREC)std::stod(elements.at(2)), (_PREC)std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "Gx"){
		opt.Gx = {(_PREC)std::stod(elements.at(1)), (_PREC)std::stod(elements.at(2)), (_PREC)std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "Gy"){
		opt.Gy = {(_PREC)std::stod(elements.at(1)), (_PREC)std::stod(elements.at(2)), (_PREC)std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "Gz"){
		opt.Gz = {(_PREC)std::stod(elements.at(1)), (_PREC)std::stod(elements.at(2)), (_PREC)std::stod(elements.at(3))};
	}

	else if(elements.at(0) == "dist"){
		opt.dist = elements.at(1)[0];
	}
	else if(elements.at(0) == "m"){
		opt.m = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "gas_coll"){
		opt.gas_coll = to_bool(elements.at(1));
	}
	else if(elements.at(0) == "T"){
		opt.T = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "diffuse"){
		opt.diffuse = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "gamma"){
		opt.gamma = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "V"){
		opt.V = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "a"){
		opt.a = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "w"){
		opt.w = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "gravity"){
		opt.gravity = to_bool(elements.at(1));
	}
    else if(elements.at(0) == "fixedStepSize"){
		opt.fixedStepSize = to_bool(elements.at(1));
	}
    else if(elements.at(0) == "keepStepSize"){
		opt.keepStepSize = to_bool(elements.at(1));
	}
	else if(elements.at(0) == "B0"){
		opt.B0 = {(_PREC)std::stod(elements.at(1)), (_PREC)std::stod(elements.at(2)), (_PREC)std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "E"){
		opt.E = {(_PREC)std::stod(elements.at(1)), (_PREC)std::stod(elements.at(2)), (_PREC)std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "t0"){
		opt.t0 = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "tf"){
		opt.tf = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "rtol"){
		opt.rtol = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "atol"){
		opt.atol = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "beta"){
		opt.beta = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "uround"){
		opt.uround = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "safe"){
		opt.safe = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "fac1"){
		opt.fac1 = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "fac2"){
		opt.fac2 = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "hmax"){
		opt.hmax = (_PREC)std::stod(elements.at(1));
	}
    else if(elements.at(0) == "hmin"){
		opt.hmin = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "h"){
		opt.h = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "nmax"){
		opt.nmax = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "seed"){
		opt.seed = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "maxPosStep"){
		opt.maxPosStep = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "integratorType"){
		opt.integratorType = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "swapStepSize"){
		opt.swapStepSize = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "numParticles"){
		opt.numParticles = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "numPerGPUBlock"){
		opt.numPerGPUBlock = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "ioutInt"){
		opt.ioutInt = (_PREC)std::stod(elements.at(1));
	}
	else if(elements.at(0) == "iout"){
		opt.iout = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "stopTimes") {
		for (int i = 1; i < elements.size(); i++) {
			opt.stopTimes.push_back(parseRange(elements.at(i)));
		}
	}
	else{
		std::cout<<"Unrecognized Option: "<<elements.at(0)<<std::endl;
		std::cout<<"Line Ignored"<<std::endl;
	}
}

Range parseRange(std::string rangeString) {
	std::vector<std::string> strings = splitString(rangeString, ":");
	if (strings.size() != 3) {
		std::cout << "Invalid range specification " << rangeString << std::endl;
	}
	Range range((_PREC) std::stod(strings[0]), (_PREC) std::stod(strings[2]),(_PREC) std::stod(strings[1]));
	return range;
}

options optionParser(char * filename){
	//first create the file object
	std::ifstream file;
	file.open(filename);
	//create the default option list
	options opt;
	if (file.is_open()) { 
		std::string sa;
		// Read data from the file object and put it into a string
		
		while (getline(file, sa)) { 
			// Print the data of the string.
			parseLine(opt, sa);
		}
		// Close the file object.
		file.close(); 
	}
	
	file.close();
	compileOptions(opt);
	return opt;
}

void compileOptions(options& opt) {
	// Any post-processing that needs to be done on user-defined options
	_PREC k = 1.380649e-23;
	opt.tc = 1.6e-4*opt.m/(k*pow(opt.T, 8.0));
	opt.sqrtKT_m = sqrt(k*opt.T/opt.m);
	opt.stopTimes.push_back(Range(opt.t0, opt.tf, opt.ioutInt));
}

