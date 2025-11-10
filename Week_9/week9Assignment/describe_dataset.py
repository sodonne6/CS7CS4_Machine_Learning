def describe_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    length = len(text)
    print(f"{path}:")
    print(f"  Number of characters (length): {length}")
    print(f"  Vocabulary size (unique chars): {vocab_size}")
    print(f"  Example snippet: {repr(text[:200])}")
    print()
    return text, chars

train_text, train_chars = describe_dataset("input_childSpeech_trainingSet.txt")
test_text, test_chars = describe_dataset("input_childSpeech_testSet.txt")
shk_text, shk_chars = describe_dataset("input_shakespeare.txt")

#main to run the fucntin
if __name__ == "__main__":
    pass

def main():
    describe_dataset("input_childSpeech_trainingSet.txt")
    describe_dataset("input_childSpeech_testSet.txt")
    describe_dataset("input_shakespeare.txt")