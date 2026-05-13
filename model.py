from helper import *


config = GPTConfig(dim=768, vocab_size=50257, ctx_window=1024, num_heads=12, num_blocks=12)
model = GPT(config)

from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


print(model_hf.state_dict)
print("#####################")
print(model.state_dict)
print("#####################")
print("#####################")
print("#####################")
print("#####################")
print("#####################")

x = torch.randint(0, 50257, (2, 16))  # batch=2, seq=16

tok = model.wte(x)                          # (2, 16, 768)
pos = model.wpe(torch.arange(16))           # (16, 768)
x_emb = tok + pos                           # (2, 16, 768)

out = model.blocks[0](x_emb)
mout = model(x)
print(mout.shape)
print(out.shape)     