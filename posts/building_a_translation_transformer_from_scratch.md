# Building a Translation Transformer from Scratch

This writeup is specifically about how I went about training a 30 million parameter transformer on a translation task between English and French. It does get technical, and most of the assumptions I make are based on hypothesis I tested while training. If you want to know more about how the transformer itself works, I'll be writing a separate writeup on that — you can also look at my repo where I rebuild the actual transformer architecture from the *Attention Is All You Need* paper. That's the same architecture I used here.

---

## Why is translation hard?

Translation is just the act of taking some meaning in one language and representing the same meaning in another. What makes it hard comes from a few factors.

The first is data inefficiency — for a lot of languages, you simply don't have enough data to train models powerful enough to do translation well. The second, and more interesting one, is that languages that are very dissimilar don't share the same linguistic properties, and that makes direct translation extremely difficult.

A good example is English to Zulu. English follows a subject-verb-object order — you have spaces  between the subject, verb, and object to stress that they are different things. Southern African languages tend to group what would be separate sentence components in English into one continuous expression, almost said in one breath e.g siyafunda in english is we are learning . A model trying to do direct translation between those two languages has to actually learn that these linguistic properties are fundamentally different, which is very hard to do.

That's why I chose English and French — they share similar linguistic properties, which made this more of a proof of concept: how well can a model translate between two languages that are already structurally similar?

---

## Data

The first question when training any model is data. Specifically — what type of data do I want, and how sure am I that it will lead to generalised predictions? Generalised in the sense that the model understands the language broadly, not just a specific subset of it.

If you train purely on political data, the model gets very good at political language but falls apart on a normal everyday sentence. This is the garbage in garbage out problem in machine learning — the model fits its weight distribution onto whatever abstract properties it has to learn from the data. Train it on political language and it prunes itself to understand political language. It will never tune itself to understand a formal greeting or an idiom because it has never had to optimise for that.

So I needed data that sounded like everyday conversations anybody could have in both English and French. The dataset I chose was **Opus100** — sentence pairs between English and French drawn from a wide range of books, including novels and more technical texts. This gives the model a chance to learn general language while also picking up some more technical terms.

I trained on **800,000 sentence pairs**, which comes out to around **30-50 million tokens** across both languages. The size decision came down to a balance between having enough data for generalisation and my compute constraints.

---

## Preprocessing

Having the data isn't enough — you have to make sure it's clean, otherwise the model learns noise.

The specific constraints I put on the data were:

- **Duplicates** — removed entirely
- **Length constraints** — very short sentence pairs with a large max length act more as noise than signal, so we enforced a hard constraint on minimum length
- **Formatting errors** — used the `ftfy` library to normalise or remove malformed chunks of text
- **Time data** — removed any remnants of time expressed in words, since these can be incoherent across languages
- **Redundant translations** — removed pairs where English was translated to English, or French to French, rather than across languages

After all of that we still ended up with around 800,000 clean pairs.

---

## Tokenisation

The model doesn't deal with words — it works with numbers. So you need some representation of those words, and that means making a big decision upfront: do you use a shared vocabulary across both languages, or do you train separate tokenisers for each?

Training separate tokenisers prevents leakage — whatever tokens are produced for French can never be tokens generated from English, so the translation is always purely in the target language. The model is forced to rely purely on the attention layers to bridge the two languages rather than having any shared vocabulary to lean on. The modern approach is a shared embedding, but I chose disjoint embeddings specifically to see how well the model works when it has never shared vocabulary between English and French.

For the tokeniser itself I used **SentencePiece** with the **unigram algorithm** — chosen because it doesn't take too much compute or time to train. I set a vocabulary size of **32,000** and added two special tokens: a start-of-sequence token and an end-of-sequence token. Their job is to give the model grounding on when a sentence starts and ends, which during inference lets you stop generation at the end token rather than letting the model generate infinitely.

---

## Training approach — teacher forcing

I used a **teacher forcing** approach to train the model. In teacher forcing, you give the model both the language it needs to encode and the language it needs to decode, and let it use both to predict the next probable token. This is good for training — it gives better performance and more stable learning. The caveat is that the model becomes dependent on always having some input from the previous language to give back an output. On out-of-distribution data that is very left field, the model tends to struggle to give coherent translations. That's a known limitation of teacher forcing.

---

## Model architecture

The model is a 30 million parameter encoder-decoder transformer with:

- **4 attention heads**
- **8 layers**
- **256 dimensions**

I could have used a larger dimension size, which would have given the model more capacity to learn representations. But besides compute constraints, there's an interesting property of smaller models worth noting: smaller models are kind of like read-and-write memory models. Any small deviation from the distribution of whatever the weights have learnt causes the model to deviate way more easily than a larger model would. Bigger models have enough capacity to cling on to certain aspects of a good prediction and learn more representations — you could argue they have room to "cheat" a bit. A smaller model is forced to get really good at next token prediction in a more general way, because as soon as predictions that are out of its learnt pattern appear, it has to readjust and learn more general patterns rather than relying on memorised capacity.

---

## Training run

Training was monitored with **Weights & Biases**. We used a **warmup scheduler with Adam** — the warmup allows the model to start with a higher learning rate so it can quickly settle into more stable gradients within the loss landscape. Once the gradients have been decreasing for a specific amount of time, the learning rate starts decreasing to keep the model within that locality — ideally a global minimum, but a very good local minimum is also acceptable.

Batch size was chosen based purely on compute constraints — as high as we could go before hitting GPU out-of-memory errors. Larger batch sizes give more stable gradient estimates but mean fewer updates per epoch, so the model needs to train longer to learn general patterns. A smaller batch would have been noisier but updated more frequently.

We trained for **23 epochs** in total.

![Full training run — step, learning rate, and loss](/assets/full_traning_run_translation.png)

The loss drops significantly with every new iteration as the model sees the same input repeatedly, settling from around 10 down toward 2. this is from the first half run as it took 2 days to train the model fully

---

## Evaluation

After training, we ran an inference check — the model was giving coherent French on shorter sentences at specific checkpoints, and by the final run it was producing sensible translations consistently.

For formal evaluation we used the **BLEU score**, which borrows from the n-gram approach to language modelling — specifically, how well does the model's output match the actual expected next tokens from the test set.

if you want tro test it check the deep learning from scratch repo i do give guidance on how to run the inference check and blue score check

![Loss curve — full training run](/assets/loss_transaltion_image.png)


The model achieved a **BLEU score of 21.28** on the test set. That means it can produce coherent French that sounds sensible, but it's not super accurate. For very long sentences it tends to lose the thread of what it's talking about — a long-term dependency issue that comes down to model capacity and data constraints.

---

## Fine-tuning — catastrophic forgetting and the fix

The next question was how the model performs when fine-tuned on a specific domain. I used a small subset of medical translation data — around 1,000 pairs — and ran a continued pretraining run on it.

The result was bad. The BLEU score plummeted to around **9**. The model could no longer give coherent sentences and was mainly pattern matching. Give it a medical term and it got that right, but everything else fell apart. This is **catastrophic forgetting** — small models are particularly vulnerable to it because any change to the data distribution can cause the model to overwrite what it previously learned.

The fix was **mixed fine-tuning**: combining a relevant subset of the original Opus100 data with the medical fine-tuning data. The old data acts as an anchor — the model still has to do predictions for the general translation task at the same time as learning the medical terms. This prevents it from forgetting what it already knew.

The result: BLEU score recovered to **19**. The model retained coherent general translation while becoming noticeably better at medical terminology. If you try the demo and toggle between the base model and the medical fine-tune on a medical sentence, you can see the difference in output directly.

---

## Closing thoughts

This project was fundamentally a way to understand how powerful the transformer actually is. We went from RNNs and LSTMs — which always felt like a band-aid for language modelling, where even with the ability to read and write memory by importance you were always constrained by a fixed context size and fighting long-term dependency — to an attention mechanism that handles it cleanly.

The combined effect of the transformer with its attention mechanism for language processing is, I'd argue, one of the greatest advances in modern history. What started as a paper about making translation better went on to become the foundation of one of the most transformative technologies ever built.

The next writeup covers the decoder-only coding model — and it raises some interesting questions about what happens when you move away from translation toward a model trying to do more logic applied  specific  tasks.
