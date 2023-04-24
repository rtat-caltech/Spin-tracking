#ifndef __OPTIONSPARSER_H_DEFINED__
#define __OPTIONSPARSER_H_DEFINED__

#include "options.h"
#include<sstream>
#include <string>

void parseLine(options& opt, std::string s);

options optionParser(char * filename);

#endif
