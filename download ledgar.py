from datasets import load_dataset

dataset = load_dataset("coastalcph/lex_glue", "ledgar")

print(dataset)
print(dataset["train"][0])