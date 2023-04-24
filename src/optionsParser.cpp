// This script is responsible for parsing the various option parameters that the user can pass in



#include "../include/optionsParser.h"
#include <iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<algorithm>

std::string removeWhitespace(std::string s){
	s.erase(remove_if(s.begin(), s.end(), isspace), s.end());
	return s;
}

bool to_bool(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    std::istringstream is(str);
    bool b;
    is >> std::boolalpha >> b;
    return b;
}

std::vector<std::string> grabElementsInLine(std::string s){
	size_t pos = 0;
	std::string token;
	std::string delimiter = ",";
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
		opt.L = {std::stod(elements.at(1)), std::stod(elements.at(2)), std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "L"){
		opt.yi = {std::stod(elements.at(1)), std::stod(elements.at(2)), std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "dist"){
		opt.dist = elements.at(1)[0];
	}
	else if(elements.at(0) == "m"){
		opt.m = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "gas_coll"){
		opt.gas_coll = to_bool(elements.at(1));
	}
	else if(elements.at(0) == "tc"){
		opt.tc = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "T"){
		opt.T = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "diffuse"){
		opt.diffuse = to_bool(elements.at(1));
	}
	else if(elements.at(0) == "gamma"){
		opt.gamma = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "V"){
		opt.V = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "gravity"){
		opt.gravity = to_bool(elements.at(1));
	}
	else if(elements.at(0) == "B0"){
		opt.B0 = {std::stod(elements.at(1)), std::stod(elements.at(2)), std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "E"){
		opt.E = {std::stod(elements.at(1)), std::stod(elements.at(2)), std::stod(elements.at(3))};
	}
	else if(elements.at(0) == "t0"){
		opt.t0 = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "tf"){
		opt.tf = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "rtol"){
		opt.rtol = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "atol"){
		opt.atol = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "beta"){
		opt.beta = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "uround"){
		opt.uround = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "safe"){
		opt.safe = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "fac1"){
		opt.fac1 = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "fac2"){
		opt.fac2 = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "hmax"){
		opt.hmax = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "h"){
		opt.h = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "nmax"){
		opt.nmax = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "integratorType"){
		opt.integratorType = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "swapStepSize"){
		opt.swapStepSize = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "numParticles"){
		opt.numParticles = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "numPerGPUBlock"){
		opt.numPerGPUBlock = std::stoi(elements.at(1));
	}
	else if(elements.at(0) == "output"){
		opt.output = elements.at(1)[0];
	}
	else if(elements.at(0) == "gridSize"){
		opt.gridSize = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "vecBinSize"){
		opt.vecBinSize = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "ioutInt"){
		opt.ioutInt = std::stod(elements.at(1));
	}
	else if(elements.at(0) == "iout"){
		opt.iout = std::stoi(elements.at(1));
	}
}

options optionParser(char * filename){
	//first create the file object
	std::ifstream file;
	file.open(filename);
	//create the default option list
	options opts;
	if (file.is_open()) { 
		std::string sa;
		// Read data from the file object and put it into a string
		
		while (getline(file, sa)) { 
			// Print the data of the string.
			parseLine(opts, sa);
		}
		// Close the file object.
		file.close(); 
	}
	
	file.close();
	return opts;
}
