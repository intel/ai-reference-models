#!/bin/bash

function print_vars {
  for VAR in ${!CCL*} ${!I_MPI*} ${!i_mpi*} ${!PSM3*} ${!FI_*} ${!KMP_*} ${!OMP_*} ${!ATL_*} LD_PRELOAD LD_LIBRARY_PATH ${!DLRM_*} ${!PYTORCH_*} ${!PCL_*} ${!LIBXSMM_*} ${!EMULATE_*} DATALOADER_WORKER_COUNT VIRTUAL_ENV ${!ARGS_*} $@ ; do
    if ! test -z ${!VAR} ; then
       echo "Using $VAR=${!VAR}"
    fi
  done
}

SINGLE_SOCKET_ONLY=0

while (( "$#" )); do
  case "$1" in
    -n|-np)
      ARGS_NTASKS=$2
      shift 2
      ;;
    -ppn)
      ARGS_PPN=$2
      shift 2
      ;;
    -f)
      ARGS_HOSTFILE=$2
      shift 2
      ;;
    -sso)
      SINGLE_SOCKET_ONLY=1
      shift
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      break
      ;;
  esac
done

if ! test -z $SLURM_JOB_ID ; then
  PREFIX="srun -n 1 -N 1 "
else
  PREFIX=
fi

if ! test -z $ARGS_HOSTFILE ; then
  if ! test -f $ARGS_HOSTFILE ; then
    echo "Hostfile $ARGS_HOSTFILE does not exist!" ; exit 1
  else
    OPT_HOSTFILE="-f $ARGS_HOSTFILE"
    PREFIX="mpiexec.hydra -np 1 -ppn 1 -f $ARGS_HOSTFILE"
  fi
fi

CORES_PER_SOCKET=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
NUM_SOCKETS=`$PREFIX lscpu | grep "Socket(s):" | awk '{print $NF}'`
NUM_NUMA_NODES=`$PREFIX lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
THREADS_PER_CORE=`$PREFIX lscpu | grep "Thread(s) per core:" | awk '{print $NF}'`

NNODES=1
NP=1
if [ $SINGLE_SOCKET_ONLY -eq 1 ] ; then
PPN=1
else
PPN=$NUM_NUMA_NODES
fi

if ! test -z $SLURM_NNODES ; then NNODES=$SLURM_NNODES ; fi
if ! test -z $SLURM_NTASKS ; then NP=$SLURM_NTASKS ; fi
if ! test -z $SLURM_NNODES && ! test -z $SLURM_NTASKS ; then PPN=$(( SLURM_NTASKS / SLURM_NNODES )) ; fi
if ! test -z $ARGS_NTASKS ; then NP=$ARGS_NTASKS ; fi
if ! test -z $ARGS_HOSTFILE ; then
  NNODES=`cat $ARGS_HOSTFILE | sort -u | wc -l`
fi

if ! test -z $ARGS_PPN ; then
  PPN=$ARGS_PPN
fi
REAL_NNODES=$(( (NP + PPN - 1) / PPN ))
if [[ $REAL_NNODES -lt $NNODES ]] ; then NNODES=$REAL_NNODES ; fi

if [ $(( NP % NNODES )) -ne 0 ] ; then
  echo "Number of tasks ($NP) not multiple of number of nodes ($NNODES), exiting..."
  exit 1
fi

PPN=$(( NP / NNODES ))

echo "Running $NP tasks on $NNODES nodes with ppn=$PPN"


OPT_PPN="-ppn $PPN "

HT_WORKER_OFFSET=$(( CORES_PER_SOCKET * NUM_SOCKETS ))
if [ $SINGLE_SOCKET_ONLY -eq 1 ] ; then
  NUM_THREADS=$(( CORES_PER_SOCKET / PPN ))
else
  NUM_THREADS=$(( CORES_PER_SOCKET * NUM_SOCKETS / PPN ))
fi

if [ "x${DATALOADER_WORKER_COUNT}" == "x" ] ; then
DATALOADER_WORKER_COUNT=0
fi

if [ $NP == 1 ] ; then
export CCL_WORKER_COUNT=0
else
if [ "x${CCL_WORKER_COUNT}" == "x" ] ; then
export CCL_WORKER_COUNT=1
fi
fi
CCL_WORKER_AFFINITY=""
PYTORCH_MPI_THREAD_AFFINITY=""

#NUM_RESV_THREADS=$(( CCL_WORKER_COUNT + DATALOADER_WORKER_COUNT ))
NUM_RESV_THREADS=$(( DATALOADER_WORKER_COUNT ))
NUM_WORKER_THREADS=$(( NUM_THREADS - NUM_RESV_THREADS ))
USE_BC=1
if ! which bc >& /dev/null ; then USE_BC=0 ; fi
for I in 0 1 2 3 ; do
SHFT=$(( NUM_RESV_THREADS + I ))
if [ $USE_BC -eq 1 ] ; then
PROC_MASK_STR[$I]=`BC_LINE_LENGTH=0 bc <<<"obase=16;(2^${NUM_WORKER_THREADS} - 1)*(2^${SHFT} )"`
else
PROC_MASK=$(( ( ( 1 << NUM_WORKER_THREADS ) - 1 ) << SHFT ))
PROC_MASK_STR[$I]=`printf "%X" $PROC_MASK`
fi
#echo "PROC_MASK_STR $I = ${PROC_MASK_STR[$I]}"
done
MASKS=( )
for(( I=0; I < PPN ; I++)) ; do
  SHFT=$(( I * NUM_THREADS ))
  IND=$(( SHFT % 4 ))
  if [ $SHFT -lt 4 ] ; then
  ZEROS=""
  else
  ZEROS=`printf "%0*X" $(( SHFT / 4 ))`
  fi
  SMASK=${PROC_MASK_STR[$IND]}${ZEROS}
  MASKS[$I]="0x$SMASK"
  for((P=0;P < CCL_WORKER_COUNT ; P++)); do CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $(( HT_WORKER_OFFSET + I * NUM_THREADS + P ))" ; done
  PYTORCH_MPI_THREAD_AFFINITY="${PYTORCH_MPI_THREAD_AFFINITY} $(( HT_WORKER_OFFSET + I * NUM_THREADS ))"
done
export I_MPI_PIN_DOMAIN=[`echo ${MASKS[@]} | tr " " ","`]
export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`
#export OMP_NUM_THREADS=$(( NUM_THREADS - CCL_WORKER_COUNT - DATALOADER_WORKER_COUNT ))
export OMP_NUM_THREADS=$(( NUM_THREADS - DATALOADER_WORKER_COUNT ))
export PYTORCH_MPI_THREAD_AFFINITY=`echo ${PYTORCH_MPI_THREAD_AFFINITY} | tr " " ","`

which python icc gcc mpicc mpiexec.hydra 2> /dev/null

echo "#### INITIAL ENV ####"
print_vars
echo "#### INITIAL ENV ####"

echo "PyTorch version: `python -c "import torch; print(torch.__version__)" 2> /dev/null`"

if ! test -z $SLURM_JOB_ID ; then
srun hostname | sort -u
fi

export MASTER_ADDR=`$PREFIX hostname`
export MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR"

CMD=$1
shift
ARGS="$@"

MPIEXE_ARGS="-np $NP $OPT_PPN $OPT_HOSTFILE -l -genv I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN -genv CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY -genv CCL_WORKER_COUNT=$CCL_WORKER_COUNT -genv OMP_NUM_THREADS=$OMP_NUM_THREADS "

#echo "Running mpiexec.hydra ${MPIEXE_ARGS} $CMD $@"
eval set -- "${MPIEXE_ARGS} hostname"
mpiexec.hydra $@ | sort
eval set -- "${MPIEXE_ARGS} $CMD $ARGS"
echo "Running mpiexec.hydra $@"
echo "Start Time:  `date`"
SECONDS=0
#mpiexec.hydra ${MPIEXE_ARGS} ${CMD} $@
mpiexec.hydra $@
echo "End Time:    `date`"
duration=$SECONDS
echo "Total Time: $(($duration / 60)) min and $(($duration % 60)) sec"

