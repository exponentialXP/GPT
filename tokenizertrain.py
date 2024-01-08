import sentencepiece 
if __name__ == '__main__':

    state = 'TEST' # TRAIN to train tokenizer and download dataset, TEST to test tokenizer, DOWNLOAD to just download the dataset and save it to a txt file, LOAD to not download but just train the tokenizer

    if state == 'TRAIN' or state == 'DOWNLOAD':
        from datasets import load_dataset 
        examples_to_train = 100_000 # 1 example is about 1000 tokens for openwebtext
        dataset = load_dataset('openwebtext', cache_dir='./cache')
        from tqdm import tqdm

        load_txt_file = False
        file_path = 'owt_temp.txt'

        if load_txt_file == False:
            pbar = tqdm(total=examples_to_train, desc="Processing text")
            for i, example in enumerate(dataset['train']):
                example_text = example['text']

                if i > examples_to_train:
                    break

                pbar.update(1)

                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(example_text)
                
                del example_text

    if state == 'TRAIN' or state == 'LOAD':
        input_file = 'owt_temp.txt'
        tokenizer_save_path = 'tokenizer'
        vocab_size = 32_000
        special_tokens = '<|endoftext|>,<|$USER|>,<|$ASSISTANT|>' # Seperate each special token with a comma

        sentencepiece.SentencePieceTrainer.train(
            model_type='BPE',
            train_extremely_large_corpus=True,
            input=input_file,
            model_prefix=tokenizer_save_path,
            vocab_size=vocab_size,
            user_defined_symbols=special_tokens
        )

    elif state == 'TEST':
        tokenizer_load_path = 'tokenizer.model'
        p = sentencepiece.SentencePieceProcessor(model_file=tokenizer_load_path)
        sentence = 'Harry went to the market'
        tokens = p.Encode(sentence)
        print(tokens)
        rev_sentence = p.Decode(tokens)
        print(rev_sentence)
