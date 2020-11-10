# DEPRECATED: Our next project of XCS library is available
Newer project => [m4saka/xcspp](https://github.com/m4saka/xcspp)
- Changes from xxr:
    - Much simpler code (w/o complex C++ template)
    - Not using non-thread-safe random utility
    - Using exception instead of C-style assertion

--------

# **xxr** : XCS &amp; XCSR Application / C++ Library
Learning classifier systems (XCS & XCSR) written in C++14

- Full implementation of XCS and XCSR (XCS for real values)
    - Built-in benchmark problems
        - Multiplexer problem (XCS)
        - Even-parity problem (XCS)
        - Majority-on problem (XCS)
        - Block world (maze/woods) problem (XCS)
        - Real-multiplexer problem (XCSR)
        - Checkerboard problem (XCSR)
        - Function problem (XCSR)
    - CSV dataset (XCS & XCSR)
- Also can be used as a header-only library

# Compilation
```
$ git clone https://github.com/m4saka/xxr.git && cd xxr
$ git submodule update -i
$ make
```

# Usage

## 6-bit Multiplexer Problem
```
$ ./xcs --mux=6
```

## 6-bit Real Multiplexer Problem (Center-Spread Representation)
```
$ ./xcsr --mux=6 --repr=csr
```

## 6-bit Real Multiplexer Problem (Ordered Bound Representation)
```
$ ./xcsr --mux=6 --repr=obr
```

## 6-bit Real Multiplexer Problem (Unordered Bound Representation)
```
$ ./xcsr --mux=6 --repr=ubr
```

## Checkerboard Problem (n=3, n_d=3, CSR)
```
$ ./xcsr --chk=3 --chk-div=3 --repr=csr
```

## From CSV
```
$ ./xcs --csv=dataset.csv --action=0,1
```

## From CSV (with another test dataset)
```
$ ./xcs --csv=dataset.csv --csv-eval=dataset_test.csv --action=0,1
```

## For the details:
```
$ ./xcs --help
```
```
$ ./xcsr --help
```

# Use as a C++ library
```cpp
#include <iostream>
#include <xxr/xcs.hpp>

using namespace xxr;

int main()
{
    // XCS parameters
    auto constants = XCSConstants();
    constants.n = 1200; // N (max number of classifiers)

    // XCS classifier system
    auto xcs = XCS<>({ 0, 1 } /* Available actions */, constants);

    // Learning iterations
    while (/* Termination criteria */)
    {
        int action = xcs.explore(/* Input (std::vector<int>) */);

        // ... (Evaluate the action here!)

        xcs.reward(/* Reward (double) */);
    }

    // Show acquired rules
    xcs.dumpPopulation(std::cout);

    // Use "xcs.exploit(input)" to get the most desirable action
    // chosen by using acquired rules

    return 0;
}
```

# Reference

- S. W. Wilson, "Classifier Fitness Based on Accuracy," Evolutionary Computation, Vol.3, No.2, pp.149-175, June 1995. https://doi.org/10.1162/evco.1995.3.2.149
    - The original paper of XCS.
- M. V. Butz, S. W. Wilson, "An algorithmic description of XCS," S. Soft Computing, Volume 6, Issue 3-4, pp 144-153, June 2002. https://doi.org/10.1007/s005000100111
    - The tutorial for the XCS implementation.
- S. W. Wilson, "Get Real! XCS with Continuous-Valued Inputs," Learning Classifier Systems, IWLCS 1999, Lecture Notes in Computer Science, vol 1813, pp 209-219, July 2000. https://doi.org/10.1007/3-540-45027-0_11
    - The original paper of XCSR.
    - The Center-Spread representation is proposed in this paper.
- C. Stone, L. Bull, "For Real! XCS with Continuous-Valued Inputs," Evolutionary Computation, Vol. 11, No.3, pp.299-336, Fall 2003. https://doi.org/10.1162/106365603322365315
    - The Ordered Bound Representation (which is equivalent of LU representation) is proposed based on the the interval-based integer representation in XCSI [Wilson, 2001].
    - The Unordered Bound Representation (UBR) is proposed in this paper.
- A. Wada, K. Takadama, K. Shimohara, O. Katai, "Analyzing Parameter Sensitivity and Classifier Representations for Real-Valued XCS," Learning Classifier Systems. IWLCS 2003, IWLCS 2004, IWLCS 2005, Lecture Notes in Computer Science, vol 4399, pp 1-16, 2007. https://doi.org/10.1007/978-3-540-71231-2_1
    - The Ordered Bound Representation (called "LU representation" in this paper) is rediscovered based on the interval-based integer representation in XCSI [Wilson, 2001].
    - The Center-Spread representation (CSR) and the Lower-Upper representation (OBR) in this program are based on this paper.
