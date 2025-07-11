import time
import argparse
import mindspore as ms
from mindnlp.configs import use_pyboost
from mindnlp.configs import cpu_boost, set_cpu_boost
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from mindnlp.core.nn.functional import linear
#################################################### PLEASE READ ME FIRST ############################################
# key api tho boost on cpu, it will download some 'so' files where the Torch cpu operators are used and integrated to
# mindspore if they don't exist on your machine, so the first time you  run this script, it will be slow,
# but from the second time, it will be fast.
set_cpu_boost(True)
print("Use pyboost:", use_pyboost())
print(f"Cpu boost:", cpu_boost())

parser = argparse.ArgumentParser(
    description="cpu boost使用示例",
)
parser.add_argument("--dtype", type=str, default="fp32", help="数据类型") # 当前只支持fp32和bf16，对齐torch2.1+cpu

args = parser.parse_args()
if args.dtype == "fp16":
    raise ValueError("Unsupported dtype: {}".format(args.dtype))
elif args.dtype == "fp32":
    ms_dtype = ms.float32
elif args.dtype == "bf16":
    ms_dtype = ms.bfloat16
else:
    raise ValueError("Unsupported dtype: {}".format(args.dtype))

model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, ms_dtype=ms_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "请介绍一下华为"
messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="ms")


def gen(model, model_inputs):
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=10,
        # max_new_tokens=512,
        do_sample=False,
    )
    end_time = time.time()
    print(f"推理时间: {end_time-start_time:.2f} 秒")
    return generated_ids

def decode_ids(generated_ids):
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

def no_profile():
    generated_ids = gen(model, model_inputs)
    decode_ids(generated_ids)

no_profile()

# python cpu_boost_demo.py  --dtype fp32