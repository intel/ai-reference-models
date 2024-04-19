#include "kmp_launcher.hpp"

namespace kmp {
void KMPLauncher::pinRoot(int place) {
  KMPAffinityMask mask;
  mask.addCore(place).bind();
  // std::cout<<"Pin root at: "<<place<<std::endl;
}

int KMPLauncher::getMaxProc() {
  return KMPAffinityMask::getMaxProc();
}

void KMPLauncher::pinThreads() {
  if (!affinityChanged_ && places_.size() > 0)
    return;

  // for (auto place : places_)
  //     std::cout<<"Pin threads at :"<<place<<std::endl;
  setNumOfThreads(places_.size());

# pragma omp parallel
  {
    KMPAffinityMask mask;
    mask.addCore(places_[omp_get_thread_num()]).bind();
  }

  affinityChanged_ = false;
}

KMPLauncher& KMPLauncher::setAffinityPlaces(std::vector<int> newPlaces) {
  // std::sort(newPlaces.begin(), newPlaces.end(), std::less<int>());

  if (places_ != newPlaces) {
    places_ = std::move(newPlaces);
    affinityChanged_ = true;
  }

  return *this;
}

std::vector<int> KMPLauncher::getAffinityPlaces() const {
  return places_;
}

size_t KMPLauncher::getNumOfThreads() {
  return omp_get_thread_num();
}

void KMPLauncher::setNumOfThreads(int nCores) {
  omp_set_num_threads(nCores);
}

}
