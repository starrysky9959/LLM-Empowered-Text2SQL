#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/bingxing2/apps/anaconda/2021.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/bingxing2/apps/anaconda/2021.11/etc/profile.d/conda.sh" ]; then
        . "/home/bingxing2/apps/anaconda/2021.11/etc/profile.d/conda.sh"
    else
        export PATH="/home/bingxing2/apps/anaconda/2021.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# load environemnt
module load compilers/gcc/9.3.0 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x anaconda/2021.11
conda activate deepseek
export PYTHONUNBUFFERED=1
python main1.py
# python chroma_demo.py
