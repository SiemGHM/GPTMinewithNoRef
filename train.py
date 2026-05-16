import torch
import torch.optim
from transformers import AutoTokenizer
from helper import *



batch_size = 16
    
class DataLoader:
    def __init__(self, file, block_size):
        self.block_size = block_size
        with open(file, "r") as f:
            self.lines = f.readlines()
            
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.encoded_lines = []
        for line in self.lines:
            self.encoded_lines.append(self.tokenizer.encode(line))

        # print(lines[:10])
        # print(encoded_lines[:10])
        self.text_copra = []
        for i in self.encoded_lines:
            self.text_copra+=i
        self.len_data = len(self.text_copra)
        self.text_copra = torch.tensor(self.text_copra, dtype=torch.long)
                
    def get_batch(self, batch_size, index):
        offset = (batch_size * index)
        assert offset + batch_size + self.block_size <= len(self.text_copra), "out of self.text_copra" 
        x = torch.stack([self.text_copra[offset + i : offset + i + self.block_size] for i in range(batch_size)])
        y = torch.stack([self.text_copra[offset + i + 1 : offset + i + self.block_size + 1] for i in range(batch_size)])
        return x, y
        
        
# training loop

NUM_EPOCHS = 3

dim = 128
vocab_size = 50257  # if using tiktoken
n_heads = 4
n_layers = 4
block_size = 64


gpt_config = GPTConfig(dim, vocab_size, block_size, n_heads, n_layers)

model = GPT(gpt_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay= 0.1)

data_loader = DataLoader("input.txt", block_size)
print("wagwanita")
for i in range(NUM_EPOCHS):
    for ind in range((data_loader.len_data)//batch_size):
        
        X, y = data_loader.get_batch(batch_size, ind)
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ind % 10 == 0:
            print(f"step {ind}, loss: {loss.item():.4f}")
    

