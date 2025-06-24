import json

data = []
with open('/mnt/vast/home/sanjana/dpo_data/dpojpi_chatml_no_names_llama33i_resample.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# data now contains all JSON objects from the file
import pdb; pdb.set_trace()
