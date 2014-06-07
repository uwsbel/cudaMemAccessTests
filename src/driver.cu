#include <stdio.h>
#include <string.h>
#include <iostream>
#include "oddsEnds.h"

using namespace std;


int main(int argc, char **argv)
{
    const string BASIC("basic");
    const string PINNED("pinned");
    const string ZEROCOPY("zero-copy");
    const string UVA("UVA");
    const string UNIFMEM("UM");

    if( argc==1 ) {
        cout << "Run using one of the following arguments:\n";
        cout << "basic" << endl;
        cout << "pinned" << endl;
        cout << "zero-copy" << endl;
        cout << "UVA" << endl;
        cout << "UM" << endl;

        cout << "\nRunning now some default test..." << endl;
        return default_test();
    }

    if( !BASIC.compare(argv[1]) )
        return test_basic();
    else if( !PINNED.compare(argv[1]) )
        return test_pinned();
    else if( !ZEROCOPY.compare(argv[1]) )
        return test_zerocopy();
    else if( !UVA.compare(argv[1]) )
        return test_UVA();
    else if( !UNIFMEM.compare(argv[1]) )
        return test_uniformMem();
}
