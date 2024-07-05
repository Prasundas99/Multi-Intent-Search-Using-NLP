import sentencepiece as spm

def train_bpe_model(corpus):
    # Write the corpus to a temporary text file
    with open('corpus.txt', 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line + '\n')

    # Train the SentencePiece model
    spm.SentencePieceTrainer.train(input='corpus.txt', model_prefix='bpe', vocab_size=5000)

    return spm.SentencePieceProcessor()

def load_bpe_model():
    sp = spm.SentencePieceProcessor()
    sp.load('bpe.model')
    return sp
