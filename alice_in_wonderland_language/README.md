# Alice in Wonderland Language Models

This contains a simple word-level RNN trained on Alice in Wonderland.
It does not achieve very good results, for a variety of reasons:
* The dataset is very small
* So is the model
* RNNs are not the best architecture for natural language in general
* Lewis Carrol has a unique writing style

However, the model does successfully learn _something_:

> Untrained sample:  let's go to the you he towsear wh g? haj fpli pa bqu!hareenesv]'sw,[ ill gll'sj t!((zow p' nzl youv alice izden's: oen b( nb; d! [.] quinsh youv [ p bhats youu ba?l,[toyod g:z of n the and:rli

> Trained sample:  let's go to the room!   alice  i don't tell is the doors at least as if you don't tell me the stupose wid you playing at the queen off you don't tell the ready everybody excame.   alice

So why bother? I think as a mathematician and a linguist he would have found language models interesting, or offensive perhaps

It's also primarily an experiment into how to try to make a bad model, with bad data as good as it can possibly get.

For example:
* It achieves far better results with a small vocabulary.
* It did better with a very small training rate
* Cleaning the data to make it as standard as possible helped a lot
* Increasing sequence length did not help particularly