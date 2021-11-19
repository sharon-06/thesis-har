ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# build mmaction2 for all common cuda architectures
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nano p7zip-full imagemagick unrar wget unzip

# Install mmcv-full
RUN pip install mmcv-full==1.3.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# Install MMAction2
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
WORKDIR /mmaction2
RUN mkdir -p /mmaction2/data
# during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .

COPY annotations/kinetics_400_annotations.txt data/annotations/
COPY annotations/tanz_annotations.txt data/annotations/
COPY scripts/process_automation.py demo/
COPY scripts/long_video_demo.py demo/
COPY scripts/i3d_r50_video_32x2x1_tanz_baseline.py configs/recognition/i3d/
RUN mkdir checkpoints && wget -O checkpoints/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth
COPY checkpoints/i3d_60_epoch.pth checkpoints/
