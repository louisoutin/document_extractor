if nvidia-smi 2>&1 | grep -q "NVIDIA-SMI has failed"; then
    echo "Installing projects using torch CPU"
    poetry install --extras=cpu
else
    echo "Installing projects using torch CUDA"
    poetry install --extras=cuda --with cuda
fi