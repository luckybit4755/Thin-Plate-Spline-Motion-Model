#!/bin/bash	
#############################################################################
# 
# This script is based on the collab and youtube video about using a driver
# video to animate an image.
# 
# It needs a source image (source.png) and an video (driving.mp4) to create
# a 256x256 output video (result.mp4) and and upsized version that is 
# 512x512 (upsized-result.mp4)
# 
# It uses pyenv and pythong 3.9.16. There is a *lot* off juking around with
# wget and other sorts of stuff... I've tried to harden it slightly, but ymmv
# 
# This script is designed to allow you to continue where you left off. If 
# you want to rerun, either remove the bit it already created or use a new 
# filename.
# 
#############################################################################

#############################################################################
# sources and resources
#
# https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model
# https://www.youtube.com/watch?v=XjObqq6we4U
# https://huggingface.co/spaces/CVPR/Image-Animation-using-Thin-Plate-Spline-Motion-Model

#############################################################################
# or from https://drive.google.com/drive/folders/1pNDo1ODQIb5HVObRtCmubqJikmR7VVLT?usp=sharing

export VOX_PATH="https://cloud.tsinghua.edu.cn/seafhttp/files/f5d7905f-5e92-435d-be40-565d5fdb0596/vox.pth.tar"

#############################################################################
# video2x bits

export V2X_URL="https://github.com/k4yt3x/video2x/archive/refs/tags/4.7.0.tar.gz"

export V2X_DRIVER_URI="https://github.com/nihui"
export V2X_DRIVER_URL="${V2X_DRIVER_URI}/srmd-ncnn-vulkan/releases/download/20200818/srmd-ncnn-vulkan-20200818-linux.zip"
export V2X_DRIVER_URL="${V2X_DRIVER_URI}/waifu2x-ncnn-vulkan/releases/download/20200818/waifu2x-ncnn-vulkan-20200818-linux.zip"
export V2X_DRIVER_URL="${V2X_DRIVER_URI}/realsr-ncnn-vulkan/releases/download/20200818/realsr-ncnn-vulkan-20200818-linux.zip"

export V2X_YAML_URL="https://raw.githubusercontent.com/lenardcarroll/video2x.yaml/main/video2x.yaml"

#############################################################################
# either gpu or cpu
export RUN="gpu"

#############################################################################

_run_me_main() {
	#############################################################################
	# passing these in bash is exhausting...

	export SOURCE="${1-source.png}" ; shift
	export DRIVING="${1-driving.mp4}" ; shift
	export RESULT="${1-result.mp4}" ; shift

	local r=$( echo ${RESULT} | sed 's,\.mp4,,' )

	export UPSIZED="upsized-${RESULT}"
	export FRAMES="frames-${r}"
	export FIXED="fixed-${r}"
	export RECOMBINED="recombined-${RESULT}"

	#############################################################################
	# run the steps from the collab... takes a minute...

	# steps 1 and 2
	_run_me_pyfixes || return 11
	_run_me_checkpoints || return 22
	_run_me_verify_inputs ${SOURCE} ${DRIVING} || return 33

	# step 3
	_run_me_generate ${SOURCE} ${DRIVING} ${RESULT} || return 44

	# step 4
	_run_me_upsize ${RESULT} || return 55

	# step 5
	_run_me_frames ${RESULT} || return 66

	# step 6
	_run_me_gfpgan ${RESULT} || return 77

	# step 7
	_run_me_recombine ${RESULT}  || return 88

	echo wow...
}

#############################################################################
# step 3

_run_me_generate() {
	if [ -f "${RESULT}" ] ; then
		echo ${RESULT} exists, not recreating it...
		return 0
	fi

	local extra=""
	if [ "cpu" = "${RUN}" ] ; then
		extra="--cpu"
	fi
	
	# for me, cpu took around 8m41.492s and gpu: 1m53.580s
	time python demo.py ${extra} \
		--config        config/vox-256.yaml \
		--checkpoint    checkpoints/vox.pth.tar \
		--source_image  ${SOURCE} \
		--driving_video ${DRIVING} \
		--result_video  ${RESULT}
	return ${?}
}

_run_me_pyfixes() {
	pyenv local 3.9.16 || return 1

	if [ -d ${HOME}/.pyenv/versions/3.9.16/lib/python3.9/site-packages/torch-1.13.1+cu117.dist-info ] ; then
		return
	fi

	cat requirements.txt \
	| sed 's,^torch==.*,# weak sauce .... torch==1.13.1+cu117,' \
	| sed 's,^torchvision==.*,# so lame... torchvision==0.13.1+cu117,' \
	| sed ';s,^Pillow.*,Pillow!=8.3,' \
	> requirements-jr.txt || return 2

	# from https://pytorch.org/get-started/previous-versions/
	pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
		--extra-index-url https://download.pytorch.org/whl/cu117

	pip install -r requirements-jr.txt || return 3
	pip install tensorboard face_alignment imageio_ffmpeg || return 4
}

_run_me_checkpoints() {
	mkdir -p checkpoints || return 1
	if [ ! -f checkpoints/vox.pth.tar ] ; then
		cd checkpoints  || return 2
		wget ${VOX_PATH} || return 3
		cd .. || return 4
	fi
}
	
_run_me_verify_inputs() {
	local ok=0
	local expected
	for expected in ${*} ; do
		if [ ! -f ${expected} ] ; then
			echo "missing input file: ${expected}"
			let ok=${ok}+1
		fi
	done

	_run_me_verify_versions || return 77

	return ${ok}
}

_run_me_verify_versions() {
	echo --------------------------------------------------------------------------------------
cat << EOM  | python || return 1
import torch
print(f"Torch Version: {torch.__version__}, CUDA Version:  {torch.version.cuda}, CUDNN Version: {torch.backends.cudnn.version()}, CUDA Available: {torch.cuda.is_available()}")
EOM
	echo --------------------------------------------------------------------------------------
}

#############################################################################
# step 4

_run_me_upsize() {
	if [ ! -f "${RESULT}" ] ; then
		return 1
	fi

	if [ -f ${UPSIZED} ] ; then
		echo ${UPSIZED} already exists
		return 0
	fi

	_run_me_upsize_packages
	_run_me_upsize_get_video2x
	_run_me_upsize_get_video2x_driver
	#############################################################################
	local driver=$( basename ${V2X_DRIVER_URL} | sed 's,-[0-9].*,,;s,-,_,g' )
	python video2x-4.7.0/src/video2x.py \
		-i ${RESULT} \
		-o ${UPSIZED} \
		-d ${driver} \
		-h 512 -w 512 || return 15
}

_run_me_upsize_packages() {
	local wanted="ffmpeg libmagic1 python3-yaml libvulkan-dev"
	local needed=""
	for package in ${wanted} ; do
		if [ 0 = 1 ] ; then
			needed="${needed}${package} "
		fi
	done

	if [ "" != "${needed}" ] ; then
		echo apt install ${needed} || return 2
		sudo apt install ${needed} || return 2
	fi

	pip install --user youtube-dl || return 3
	pip install -U PyYAML || return 4
}

_run_me_upsize_get_video2x() {
	local v2x_tgz=$( basename ${V2X_URL} )
	local v2x_dir=$( echo ${v2x_tgz} | sed 's,\.tar.\gz,,' )

	if [ -d ${v2x_dir} ] ; then 
		return 0
	fi

	if [ ! -f ${v2x_tgz} ] ; then
		wget ${V2X_URL} || return 5
	fi

	tar -xvf ${v2x_tgz} || return 6
	cd ${v2x_dir}/src || return 7
	pip install -r requirements.txt || return 7

	# have to fix the paths

	local root=${PWD}
	if [ ! -f og-video2x.yaml ] ; then
		mv -i video2x.yaml og-video2x.yaml 
	fi

	curl ${V2X_YAML_URL} | sed "s,/content/,${root}/,g" > video2x.yaml

	cd ../.. || return 11
}

_run_me_upsize_get_video2x_driver() {
	local url=${V2X_DRIVER_URL}
	local zip=$( basename ${url} )
	local dir=$( echo ${zip} | sed 's,\.zip$,,' );
	if [ ! -d ${dir} ] ; then
		wget ${url} || return 12
		unzip ${zip} || return 13
		rm -f ${zip} || return 14
	fi
	local unv=$( echo ${dir} | sed 's,-[0-9].*,,' )
	if [ -h ${unv} ] ; then
		ln -s ${dir} ${unv}
	fi
}

#############################################################################
# step 5

_run_me_frames() {
	if [ -d ${FRAMES} ] ;then
		echo ${FRAMES} already exists
		return 0
	fi

	mkdir -p ${FRAMES} || return 1
	ffmpeg -i ${UPSIZED} ${FRAMES}/out-%04d.png
}

#############################################################################
# step 6

_run_me_gfpgan() {
	if [ -d ${FIXED} ] ; then
		echo ${FIXED} already exists
		return 0
	fi

	# Install basicsr  - https://github.com/xinntao/BasicSR
	# Install facexlib - https://github.com/xinntao/facexlib
	pip install basicsr facexlib || return 1

	##>>>		mkdir -p /usr/local/lib/python3.7/dist-packages/facexlib/weights  # for pre-trained models

	if [ ! -d GFPGAN ] ; then
		git clone https://github.com/TencentARC/GFPGAN.git || return 2
		cd GFPGAN || return 3
		pip install -r requirements.txt || return 4
		python setup.py develop || return 5
		pip install realesrgan || return 6
		wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth -P experiments/pretrained_models || return 7
		cd .. || return 8
	fi

	cd GFPGAN || return 9

	python inference_gfpgan.py -i ../${FRAMES} -o ../${FIXED} --aligned || return 10
	cd .. || return 11
	##>>>		!rm fixed.zip
	##>>>		!zip -r fixed.zip fixed/restored_faces
}

#############################################################################
# step 7

_run_me_recombine() {
	if [ -f ${RECOMBINED} ] ; then
		echo ${RECOMBINED} already exists
		return 0
	fi

	ffmpeg \
		-framerate 20 \
		-pattern_type glob \
		-i ${FIXED}'/restored_faces/*.png' \
  		-c:v libx264 \
		-pix_fmt yuv420p \
		${RECOMBINED}

	return ${?}
}

#############################################################################
# the end... or is it the beginning?!
_run_me_main ${*}
