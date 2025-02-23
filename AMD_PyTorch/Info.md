# PyTorch

For PyTorch a docker is available from AMD. Contained with the folder is a script which will launch the docker which mounts the projects folder.

## AMD Sources

https://www.reddit.com/r/ROCm/comments/1ep4cru/rocm_613_complete_install_instructions_from_wsl/

WSL ubuntu 22.04:

```
 wsl --install -d Ubuntu-22.04

then after install do this inside the distribution in which you can get to in cmd using command:

wsl

then to just update the install of ubuntu to the newest version for its components do those two commands:

sudo apt-get update

sudo apt-get upgrade

then to install the drivers and install rocm do this:

sudo apt update

wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb

sudo apt install ./amdgpu-install_6.1.60103-1_all.deb

amdgpu-install -y --usecase=wsl,rocm --no-dkms 
```

Docker:

```
sudo apt install docker.iode
sudo docker pull rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2
sudo docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 8G --device=/dev/dxg -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so -v /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1 rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2
```

pip:

```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torch-2.3.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torchvision-0.18.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/pytorch_triton_rocm-2.3.0%2Brocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl
pip3 uninstall torch torchvision pytorch-triton-rocm
pip3 install torch-2.3.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl torchvision-0.18.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-2.3.0+rocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl

location=`pip show torch | grep Location | awk -F ": " '{print $2}'`
cd ${location}/torch/lib/
rm libhsa-runtime64.so*
cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so
```

\[]{https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#pytorch-docker-support}
