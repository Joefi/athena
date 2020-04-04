# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

source tools/env.sh

stage=3
stop_stage=100
horovod_cmd="horovodrun -np 2 -H localhost:2"
horovod_prefix="horovod_"
dataset_dir=/data/thchs30/data_thchs30

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Creating csv"
    mkdir -p examples/asr/data
    python examples/asr/thchs30/local/prepare_data.py \
        $dataset_dir examples/asr/thchs30/data || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn
    echo "Computing cmvn"
    cat examples/asr/thchs30/data/train.csv > examples/asr/thchs30/data/all.csv
    tail -n +2 examples/asr/thchs30/data/dev.csv >> examples/asr/thchs30/data/all.csv
    tail -n +2 examples/asr/thchs30/data/test.csv >> examples/asr/thchs30/data/all.csv
    python athena/cmvn_main.py \
        examples/asr/thchs30/configs/mpc.json examples/asr/thchs30/data/all.csv || exit 1
fi

#if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#    # pretrain stage
#    echo "Pretraining"
#    $horovod_cmd python athena/${horovod_prefix}main.py \
#        examples/asr/aishell/configs/mpc.json || exit 1
#fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # finetuning stage
    echo "Fine-tuning"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/thchs30/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # prepare language model
    echo "Training language model ..."
    tail -n +2 examples/asr/thchs30/data/train.csv |\
        cut -f 3 > examples/asr/thchs30/data/text
    tail -n +2 examples/asr/thchs30/data/train.csv |\
        awk '{print $3"\t"$3}' > examples/asr/thchs30/data/train.trans.csv
    tail -n +2 examples/asr/thchs30/data/dev.csv |\
        awk '{print $3"\t"$3}' > examples/asr/thchs30/data/dev.trans.csv
    tail -n +2 examples/asr/thchs30/data/test.csv |\
        awk '{print $3"\t"$3}' > examples/asr/thchs30/data/test.trans.csv
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/thchs30/configs/rnnlm.json || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # decoding stage
    echo "Running decode ..."
    python athena/decode_main.py \
        examples/asr/thchs30/configs/mtl_transformer_sp.json || exit 1
fi
