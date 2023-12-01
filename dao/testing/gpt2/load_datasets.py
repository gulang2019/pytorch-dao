import json 
from torch.utils.data import Dataset
    
def format(data_point):
    return f"instruction {data_point['instruction']}\n\ninput {data_point['input']}\n\noutput {data_point['output']}\n\n"

class InstructionTuningDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open('/home/siyuanch/ssd/workspace/GPT-4-LLM/data/alpaca_gpt4_data.json') as f:
            raw_data = json.load(f)
        self.data = [format(x) for x in raw_data]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], [] 