#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
//#include "thread_bind.h"
#include "kmp_launcher.hpp"

namespace py = pybind11;

namespace binder {

  void bind_thread(int startIndex, int len) {
    kmp::KMPLauncher thCtrl;
    std::vector<int> places(len);
    for (int i = 0; i < len; ++i) {
        places[i] = i + startIndex;
    }
    thCtrl.setAffinityPlaces(places).pinThreads();
  }

  void set_thread_affinity(const std::vector<int> &core_list) {
    kmp::KMPLauncher thCtrl;
    thCtrl.setAffinityPlaces(core_list).pinThreads();
  }

}


PYBIND11_MODULE(thread_binder, m){
  m.doc() = "Core binder for python process.";
  m.def("bind_thread", &binder::bind_thread);
  m.def("set_worker_affinity", &binder::set_thread_affinity);
}

