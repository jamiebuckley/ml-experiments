# Alice in Wonderland Language Models

This contains a simple RNN and Transformer model trained on Alice in Wonderland.
It does not achieve very good results, for a variety of reasons:
* The dataset is very small
* So are the models, although the limitation is more at the dataset side
* Lewis Carrol has a unique writing style and therefore this is a very difficult prediction problem

So why bother?  For one - I'd be fascinated to have seen what Lewis Carrol's opinions would have been on modern NLP technology.

But also, the model does successfully learn _something_, from the RNN:

> Prompt: oh, you can't help that; we're all

> Prompt tokens: ['▁o', 'h', ',', '▁you', '▁c', 'an', "'t", '▁he', 'l', 'p', '▁t', 'hat', ';', '▁w', 'e', "'", 're', '▁a', 'll']

> Untrained sample:  oh, you can't help that; we're allingliliingliliarg ogxi youghe,[lice ggonon,stonhat m n h quononononon[lilililiingliliarli)g cggxing't to g ggonhe! nan[ing hi h h hi you, you.] h o o og n pheliceorhatst qustghe quon w o o o o o o o

> Trained sample:  oh, you can't help that; we're all the room rooms at the room rooms at the room rooms at the room rooms at the room rooms at the room rooms at the room rooms at the

I guess we're all the rooms.  

With a higher temperature:

> Trained sample:  oh, you can't help that; we're all the gryphon and shouldn't everybody looks at the room right us!   alice  i don't know what is asked.   alice  i don't know what is all the room rooms at

> Trained sample:  oh, you can't help that; we're all  " "yes, you know," said the dormouse, "that's very curious to the processions of the trees of the slately for a minute: the queen's cropsing to the sea, and the queen's crolled it?" said the dormouse, "forrow, and the queen's very curious tone. "ho i wish i wish i wish i wish

So from a prompt, the trained sample has at least started to combine tokens semi-reasonably, and occasionally use them in
a correct sequence, i.e. it is starting to capture _some_ of the structure of English.

In one example, it achieves an entropy of around 2.3, versus around 4.6 for the untrained model with a vocabulary size of 100 
This means it is going from 'choosing' from around 100 tokens to around 10 tokens at each step, making predictions like this:

| Token | ID | Probability |
|-------|----|-----------:|
| '▁"' | 66 | 0.2991 |
| 'our' | 225 | 0.1164 |
| 'id' | 81 | 0.0769 |
| '▁' | 53 | 0.0287 |
| ''t' | 143 | 0.0247 |
| '1' | 11 | 0.0245 |
| 'ought' | 232 | 0.0239 |
| '▁and' | 75 | 0.0213 |
| '*' | 6 | 0.0175 |
| 'x' | 48 | 0.0168 |

This is primarily an experiment into how to try to make a simple model, with bad data as good as it can possibly get with nothing but dataset cleaning and hyperparameter tuning

For example:
* Cleaning the data to make it as standard as possible also helped a lot, lowercasing, removing strange punctuation, extra whitespace etc.
* Tweaks to sequence length, hidden state & embedding size, vocab size all hit a similar floor

RNN Results:

| Vocab Size | Entropy  | Embedding Size | Hidden Size | Seq Length | Epochs | Learning Rate | Best Epoch | Best Val Loss | Comparable Perplexity Score |
|------------|----------|----------------|-------------|------------|--------|---------------|------------|---------------|-----------------------------|
| 100        | 4.6052   | 50             | 64          | 5          | 20     | 0.001         | 18         | 2.4741        | 0.12                        |
| 100        | 4.6052   | 50             | 128         | 5          | 20     | 0.001         | 10         | 2.3925        | 0.11                        |
| 100        | 4.6052   | 50             | 128         | 10         | 20     | 0.001         | 9          | 2.4248        | 0.11                        |
| 100        | 4.6052   | 100            | 128         | 10         | 20     | 0.001         | 8          | 2.3720        | 0.11                        |
| 500        | 6.2146   | 100            | 128         | 5          | 20     | 0.001         | 8          | 3.9356        | 0.10                        |
| 500        | 6.2146   | 50             | 64          | 5          | 20     | 0.001         | 16         | 4.0450        | 0.11                        |
| 50         | 3.9120   | 50             | 64          | 5          | 20     | 0.001         | 20         | 1.8218        | 0.12                        |
| 200        | 5.2983   | 50             | 64          | 5          | 20     | 0.001         | 18         | 3.1335        | 0.11                        |

A transformer model only slightly improved the results, achieving an entropy of around 4.2 at vocab 500, and 2.26 at vocab 100.

