#pragma once

#include "options.h"
#include<sstream>
#include <string>

void parseLine(options& opt, std::string s);

options optionParser(char * filename);