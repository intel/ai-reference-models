#include <omp.h>
#include <vector>
#include <iostream>

namespace kmp {
//
// Wrapper around kmp_affinity_mask_t
// And help API
// Notes:
//   Use bind function will effectively remove OpenMP 4.0 affinity
//   Better disable OpenMP 4.0 affinity using environment var
//
class KMPAffinityMask {
public:
  KMPAffinityMask() {
    kmp_create_affinity_mask(&mask_);
  }
  ~KMPAffinityMask() {
    kmp_destroy_affinity_mask(&mask_);
  }

  KMPAffinityMask& bind() {
    kmp_set_affinity(&mask_);
    return *this;
  }

  KMPAffinityMask& addCore(int core) {
    kmp_set_affinity_mask_proc (core, &mask_);
    return *this;
  }

  void getAffinity() {
    kmp_get_affinity(&mask_);
  }

  void removeCore(int core) {
    kmp_unset_affinity_mask_proc (core, &mask_);
  }

  static int getMaxProc () {
    auto nMaxProc = kmp_get_affinity_max_proc();
    if (nMaxProc == 0) {
      std::cout<<"Warning: Can't get affinity max proc info. Probably running on OS that doesn't support Affinity Interface, Ex. MacOS"<<std::endl;
      // The omp_get_num_procs routine returns the number of processors
      // available to the device.
      nMaxProc = omp_get_num_procs();
    }
    return nMaxProc;
  }

  bool isSetCore(int core) {
    return kmp_get_affinity_mask_proc (core, &mask_);
  }
private:
  kmp_affinity_mask_t mask_;
};

class KMPLauncher {
public:
  KMPLauncher() : places_(), affinityChanged_(true)  {}
  ~KMPLauncher() = default;

  KMPLauncher& setAffinityPlaces(std::vector<int> newPlaces);
  std::vector<int> getAffinityPlaces() const;

  void pinThreads();
  void pinRoot(int place);

  static int getMaxProc();
  static size_t getNumOfThreads();
  static void setNumOfThreads(int nCores);

private:

  std::vector<int> places_;
  bool affinityChanged_;
};

}
