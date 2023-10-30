from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.lm import KneserNeyInterpolated, Laplace, StupidBackoff

raw_text = """
Michael Jackson, 31, was innocent, his mother said.
Michael is innocent, Katherine Jackson, said in her statement.
Malia Jackson's mother, Katherine, said Wednesday that Michael was innocent and is charged with murder.
"""

tokenized = [word_tokenize(sent) for sent in sent_tokenize(raw_text)]

train, vocab = padded_everygram_pipeline(2, tokenized)
lm = KneserNeyInterpolated(order=2)
lm.fit(train, vocab)

# Compute entropy and perplexity of these ngrams
bigrams = [("Katherine", "innocent"), ("Michael", "statement")]
print(bigrams)
print(lm.entropy(bigrams))
print(lm.perplexity(bigrams), "\n")

# Even with bigrams that don't occur, we just get higher entropies & perplexities,
# but no infinity
bigrams = [("That", "is"), ("large", "text")]
print(bigrams)
print(lm.entropy(bigrams))
print(lm.perplexity(bigrams), "\n")

# Only if *none* of the words in any of the bigrams exist, do we get infinity
bigrams = [("That", "could"), ("large", "sentences")]
print(bigrams)
print(lm.entropy(bigrams))
print(lm.perplexity(bigrams), "\n")

# For unigrams, as long as all of the words exist, we get normal outputs
unigrams = [("that",), ("long",), ("should",)]
print(unigrams)
print(lm.entropy(unigrams))
print(lm.perplexity(unigrams), "\n")

# However, if we use a unigram that does not exist, the entropy/perplexity does become infinity
unigrams = [("That",), ("long",), ("should",)]
print(unigrams)
print(lm.entropy(unigrams))
print(lm.perplexity(unigrams))