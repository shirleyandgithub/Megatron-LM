
# 基于Megatron-LM的实战准备：<br>
git clone https://github.com/NVIDIA/Megatron-LM.git<br>
pip install regex transformers pybind11<br>
mkdir -p /root/data/gpt2_local<br><br>

for i in {1..10000}; do echo "{\"text\": \"Megatron-LM is the godfather of large language models.\"}" >> /root/data/raw.jsonl; done<br><br>

# 下载GPT-2 Tokenizer的字典和拼词规则<br>
wget -O /root/data/gpt2_local/vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json<br>
wget -O /root/data/gpt2_local/merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt<br><br>
echo '{"model_type": "gpt2"}' > /root/data/gpt2_local/tokenizer_config.json<br>
echo '{"model_type": "gpt2"}' > /root/data/gpt2_local/config.json<br><br>

python tools/preprocess_data.py \<br>
--input /root/data/raw.jsonl \<br>
--output-prefix /root/data/my-gpt2 \<br>
--tokenizer-type HuggingFace Tokenizer \<br>
--tokenizer-model /root/data/gpt2_local \<br>
--append-eod \<br>
--workers 4 \<br>
--json-keys text<br><br>

# 文件说明<br>
raw.jsonl # 原文件<br>
preprocess.py # 数据格式转换成Megatron所需<br>
最终在/root/data/转化出.bin和.idx<br><br><br><br>



通信墙:cd /root/Megatron-LM/; bash ./run_step1_tp.sh<br>
 每一步耗时0.6秒(elapsed time per iteration (ms): 631.8);<br>
 但是模型很小(Hidden Size = 1024)，如果不做切分，用4090单卡每一步约10ms~50ms;现在做切分(TP=2)，却更慢(631.8ms)，说明大部分时间都花在两张卡通过PCIe总线互通上了;<br>
 NVLink可以解决此问题;<br><br>

气泡实验:cd /root/Megatron-LM; bash ./run_step2_pp_bubble.sh<br>
对比“通信墙”来分析，通信墙是TP=2、Batch=8、每步631.8，每样本平均耗时631.8/8≈79ms/sample;<br>
 气泡测试分析，PP=2、Batch=1、每步119.3(elapsed time per iteration (ms): 119.3);<br>
 虽然用2张卡算同样的模型，但是在PP=2(Batch=1)模式下，每样本处理时间从79ms增到了119ms，约50%的性能损耗被气泡吃掉;<br><br>

1F1B:cd /root/Megatron-LM; bash ./run_step3_pp_1f1b.sh<br>
对比“气泡实验”单步耗时119.3ms，1F1B的单步耗时只有51.1ms(elapsed time per iteration
(ms): 6546.3/Batch is 8)，速度提升了2.3倍;<br>
