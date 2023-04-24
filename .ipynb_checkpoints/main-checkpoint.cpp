#include <iostream>
#include <chrono>
#include <ctime>
#include "include/particle.h"
#include "include/options.h"
#include "include/double3.h"
#include "include/optionsParser.h"
#include "include/simulation.h"

using namespace std;
using namespace std::chrono;


options parseUserInput(int argc, char * argv[], char** outputName);

int main(int argc, char* argv[]){
	char * outputName;
	options opt = parseUserInput(argc, argv, &outputName);
	int totalTime = 3600; //total time allowed in seconds
	unsigned int seed = 0;
	auto start = high_resolution_clock::now();
	mainAnalysis(opt, totalTime, outputName, seed);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop-start).count();
	std::cout<<duration<<std::endl;
	return 0;
}

options parseUserInput(int argc, char *argv[], char** outputName){
	//assume the first input is the file name
	if(argc < 3){
		std::cout<<"invalid input options"<<std::endl;
		std::cout<<"Expects ./main parameterFile outputFile\n"<<std::endl;
		exit(-1);
	}
	char *filename = argv[1];
	options opt = optionParser(filename);
	*outputName = argv[2];
	return opt;
}
