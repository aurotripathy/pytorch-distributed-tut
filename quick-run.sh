export NCCL_P2P_LEVEL=4
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_MIN_NRINGS=4
export NCCL_DEBUG=INFO

python3.6 mnist-distributed.py
