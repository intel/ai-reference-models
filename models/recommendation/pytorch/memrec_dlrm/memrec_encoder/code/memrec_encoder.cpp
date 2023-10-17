// Copyright (c) 2023 Intel Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <stdlib.h> 
#include <map>
#include <ctime>
#include <Python.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include  "wyhash.h"

#if defined(_MSC_VER)
typedef signed __int8 int8_t;
typedef signed __int32 int32_t;
typedef signed __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
// Other compilers
#else    // defined(_MSC_VER)
#include <stdint.h>
#endif // !defined(_MSC_VER)
uint32_t seeds[20] = {7186053, 6387688, 5937987, 2014911, 8420444, 7735429, 7189516, 6746271, 6122425, 3363721, 1344626, 3883505, 357814, 7959252, 5356618, 6020406, 3523941, 3551544, 8555633, 9130236};
#define SETITEM(ii, jj) (*(npy_int8*)((PyArray_DATA(output_array) +               \
                                      (ii) * PyArray_STRIDES(output_array)[0] +   \
                                      (jj) * PyArray_STRIDES(output_array)[1])))

#define SETITEM_(ii, jj) (*(npy_int32*)((PyArray_DATA(output_array) +               \
                                      (ii) * PyArray_STRIDES(output_array)[0] +   \
                                      (jj) * PyArray_STRIDES(output_array)[1])))

#define SETITEM_TE(ii, jj) (*(npy_int32*)((PyArray_DATA(output_array_te) +               \
                                      (ii) * PyArray_STRIDES(output_array_te)[0] +   \
                                      (jj) * PyArray_STRIDES(output_array_te)[1])))

#define SETITEM_WE(ii, jj) (*(npy_int32*)((PyArray_DATA(output_array_we) +               \
                                      (ii) * PyArray_STRIDES(output_array_we)[0] +   \
                                      (jj) * PyArray_STRIDES(output_array_we)[1])))

#define SETITEM__(ii, jj) (*(npy_float32*)((PyArray_DATA(output_array) +               \
                                      (ii) * PyArray_STRIDES(output_array)[0] +   \
                                      (jj) * PyArray_STRIDES(output_array)[1])))

static PyObject *
wyh_hash_array_sparse(PyObject *self, PyObject *args)
{

    int hash_tech_id;
    PyArrayObject *input_list;
    PyArrayObject *output_array;
    
    int target_str_len, d, num_features, k, kw, num_ones_te, num_ones_we;

    if (!PyArg_ParseTuple
    (args, "O!iiiiii", 
     &PyArray_Type, &input_list,
     &d, &target_str_len, &num_features, &k, &kw, &hash_tech_id))
    {
        return NULL;
    }

    Py_ssize_t nrows, ncols, nseeds_te, nseeds_we;
    nrows = PyArray_SHAPE(input_list)[0];
    ncols = PyArray_SHAPE(input_list)[1];
    num_ones_te = num_features*k;
    num_ones_we = num_features*kw;
    
    npy_intp dims_[1] = { nrows*(num_ones_te+num_ones_we)};
    output_array = (PyArrayObject*) PyArray_ZEROS(1, dims_, NPY_INT, 0);

    uint32_t result[1];
    uint64_t _wyp[4];
    int new_jj_te=0;
    int new_jj_we=0;

    uint32_t seed = seeds[0];				
    Py_ssize_t ix[2];
    //make_secret(seed,_wyp);
    std::clock_t time_4 = std::clock();

    for (int jj=0; jj<ncols; jj++)
    {
        bool use_hot_hashes = true;
        int top_n_hashes = 10; //power of 2
        uint64_t hot_hashes[top_n_hashes];
        uint64_t h;

        for (int i=0; i<top_n_hashes; i++)
        {            
            uint16_t token_int=jj*430000000 + i; 
            hot_hashes[i]=wyhash(&token_int, target_str_len, seed, _wyp);
        }
        for (int ii=0; ii<nrows; ii++)
        {            
            uint16_t *item = (uint16_t *) PyArray_GETPTR2(input_list, ii, jj);
            if ((*item < top_n_hashes) && use_hot_hashes)
            {
              h = hot_hashes[*item];
            }
            else
            {
              uint16_t token_int = jj*430000000 + *item; 
              h=wyhash(&token_int, target_str_len, seed, _wyp);
            }

            
            uint32_t msw, lsw;
            msw = (h & 0xFFFFFFFF00000000) >> 32;
            lsw = h & 0x00000000FFFFFFFF;
            ix[0] = (msw) % d;
            ix[1] = (lsw) % d; //d vs its value 

            uint32_t *item_address_0 = (uint32_t *) PyArray_GETPTR1(output_array, ii*k*ncols+jj);
				    uint32_t *item_address_1 = (uint32_t *) PyArray_GETPTR1(output_array, nrows*num_ones_te+ii*kw*ncols+jj);
            *item_address_0 = ix[0];
				    *item_address_1 = ix[1];

            int num_more_hashes = k+kw-2;
            uint64_t more_hashes[num_more_hashes];
            
            if (num_more_hashes)
            {
              //create more hashes
              int more_hashes_idx = 0;
              for (int m=0; m<ceil((float)num_more_hashes/2); m++)
              {
                uint16_t token_int = jj*430000000 + *item; 
                h=wyhash(&token_int, target_str_len, seeds[m+1], _wyp);
                msw = (h & 0xFFFFFFFF00000000) >> 32;
                more_hashes[more_hashes_idx] = (msw) % d;
                more_hashes_idx++;
                lsw = h & 0x00000000FFFFFFFF;
                if (more_hashes_idx < num_more_hashes) more_hashes[more_hashes_idx] = (lsw) % d;
                more_hashes_idx++;
              }
              //store generated hashes at appropriate places in the output array
              int store_idx = 0;
              for (int this_k=1; this_k<k; this_k++) //this_k = 0 has already been processed
              {
                uint32_t *store_here = (uint32_t *) PyArray_GETPTR1(output_array, ii*k*ncols + this_k*ncols + jj);
                *store_here = more_hashes[store_idx];
                store_idx++;
              }
              for (int this_kw=1; this_kw<kw; this_kw++) //this_k = 0 has already been processed
              {
                uint32_t *store_here = (uint32_t *) PyArray_GETPTR1(output_array, nrows*num_ones_te + ii*kw*ncols + this_kw*ncols + jj);
                *store_here = more_hashes[store_idx];
                store_idx++;
              }
            }
            
        }
    }

    return PyArray_Return(output_array);
}


struct module_state {
  PyObject *error;
};
    
#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyMethodDef MEMethods[] = {
    {"wyh_hash_array_sparse", (PyCFunction)wyh_hash_array_sparse, METH_VARARGS},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int me_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int me_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef memrec_encodermodule = {
    PyModuleDef_HEAD_INIT,
    "memrec_encoder",
    "memrec_encoder is the implementation of memrec embedding encoder using WY hash",
    sizeof(struct module_state),
    MEMethods,
    NULL,
    me_traverse,
    me_clear,
    NULL
};

#define INITERROR return NULL

extern "C" {
PyMODINIT_FUNC
PyInit_memrec_encoder(void)

#else // PY_MAJOR_VERSION >= 3
#define INITERROR return

extern "C" {
void
initmemrec_encoder(void)
#endif // PY_MAJOR_VERSION >= 3

{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&memrec_encodermodule);
    import_array();
#else
    PyObject *module = Py_InitModule("memrec_encoder", MEMethods);
#endif

    if (module == NULL)
        INITERROR;

    PyModule_AddStringConstant(module, "__version__", "3.0.0");

    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException((char *) "memrec_encoder.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
} // extern "C"