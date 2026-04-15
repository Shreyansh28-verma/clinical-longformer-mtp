import re

# Update train_teacher_longformer.py
with open("c:/Users/shrey/Downloads/dataset/dataset/train_teacher_longformer.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace(
    "train_data, val_data = train_test_split(reasonings_data, test_size=0.2, random_state=42)",
    "train_data, val_data = train_test_split(reasonings_data, test_size=0.2, random_state=42)\n    train_data = train_data[:4]\n    val_data = val_data[:4]"
)
text = text.replace(
    "EPOCHS = 10",
    "EPOCHS = 1"
)

with open("c:/Users/shrey/Downloads/dataset/dataset/train_teacher_longformer.py", "w", encoding="utf-8") as f:
    f.write(text)

# Update inference_teacher_longformer.py
with open("c:/Users/shrey/Downloads/dataset/dataset/inference_teacher_longformer.py", "r", encoding="utf-8") as f:
    text_inf = f.read()

text_inf = text_inf.replace(
    "data = [json.loads(line) for line in f]",
    "data = [json.loads(line) for line in f][:2]"
)

with open("c:/Users/shrey/Downloads/dataset/dataset/inference_teacher_longformer.py", "w", encoding="utf-8") as f:
    f.write(text_inf)
