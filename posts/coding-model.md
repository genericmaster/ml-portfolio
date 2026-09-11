# Pretraining a Coding Model, FIM, and What the Model Actually Learned

Decoder models are everywhere right now. ChatGPT, Claude, Gemini, the small local models you can run on your laptop — they're all variations of the same core idea. So instead of trying to build another general language model, I wanted to see what it actually takes to build one for a very specific task: coding. Not just slap a model together and call it done, but actually understand where things break, why they break, and what you have to do differently when the task is code versus natural language.

This writeup covers the full journey — data, training, failure, a different approach, and what I actually learned at the end of it.

---

## The Data

The first real decision was which language to train on. The more languages you add, the more you're asking the model to share its capacity across all of them. Given the compute I had available, spreading across multiple languages would mean the model never gets deep enough into any one of them. So I picked Python — most readily available, most data, and the easiest to get clean production examples of.

The dataset was **CodeParrot**, specifically the cleaned split. Production code, real functions, already reasonably clean. The preprocessing checks were straightforward: fixed size filtering, length checks on functions, deduplication, and the usual data hygiene. After preprocessing the total token count landed somewhere around **1.4 to 1.7 billion tokens** — a rough estimate based on how the data was chunked, not an exact count from a token counter.

Two things stood out while digging through the data that ended up mattering a lot later:

1. The functions were long. Production code tends to have complex logic and long function bodies.
2. A large chunk of the data started with **docstrings** before any actual code. That second point came back to bite me during inference in a way I didn't expect at the time.

---

## Tokenization and Chunking

Tokenization used SentencePiece with a unigram algorithm. For special tokens, the decision at this stage was simple: just an end-of-text token. This is a decoder model trained for next token prediction — you don't need a start token, you just need the model to know when to stop.

That reasoning turned out to be wrong, but more on that shortly.

Because the dataset was too large to load all at once, it was streamed in chunks during training. Every function was chunked down to **512 tokens** — the max context length the model supports. If a function was longer, it got split. If shorter, it got padded.

---

## The Model

A decoder-only transformer, built from scratch. The architecture:

- **70M parameters**
- **512 embedding dimensions**
- **8 attention heads**
- **12 layers**

For training: Adam optimizer, a learning rate scheduler, one epoch over the data. One epoch sounds light but when you're training on ~1.5 billion tokens, one pass is enough to give the model a solid exposure to the distribution. The assumption was the model would develop a reasonable grasp of Python structure from that single pass.

Most of those assumptions turned out to be wrong.

---

## Training Run

![Training run overview](images/training_run_coding_model.png)

The loss curve dropped fast early on, then flattened out. The model hit its floor well before the end of training.

![Loss curve detail](images/loss_coding_model.png)

That early floor makes sense in hindsight. Code is a formal language — there's no such thing as a "close enough" prediction. In natural language, if the model predicts a slightly different word, the sentence can still be coherent. In code, one wrong token breaks the whole function. A 70M parameter model just doesn't have the capacity to fully encode that kind of strict logical structure. Research backs this up — models under 100M parameters tend to be pattern matchers at best, and even at 100M they're not really reasoning about code. So the loss floor wasn't a training failure, it was a capacity ceiling.

The new question became: if this model is always going to be a pattern matcher, can I at least get the pattern matching to be useful? Can it complete a function in a way that looks plausible?

---

## What Actually Came Out of Training

The model could produce code that referenced what it was given. So if you gave it the start of a function, the output would be related to that function. That part worked.

Two things that didn't:

- **It devolved into nonsense** — after a few plausible tokens, it would start looping or producing garbage.
- **It never stopped** — set the max generation length to 500 tokens, you get 500 tokens. Set it to 1000, you get 1000. The end-of-text token never fired. Ever. The model completely ignored it.

That second point was the more confusing one. The EOT token was in the training data. It appeared at the end of every chunk. So why was the model treating it like it wasn't there?

---

## Why the Model Ignored the End Token

The issue was that EOT appeared at **exactly position 512 in every single training chunk**. Always. No variation.

In natural language training, the end token appears at different positions — short sentences end early, long ones end late. The model learns that the end token means *stop*, because it sees it fire at all different points in a sequence.

In my training setup, EOT was always glued to position 512. So the model never learned what it meant. It just learned that position 512 exists, and since the token at position 512 didn't carry a learnable signal (it was always the same, always at the same spot), the model effectively tuned it out. Change the max length, and it just generates to the new limit — the token itself meant nothing.

This was the core problem, and it pointed directly to what needed to change.

---

## Fill in the Middle (FIM)

The fix came from an OpenAI paper on **Fill in the Middle** — a training technique specifically designed for coding models. 
fun fact it was also used to train codex at the time .
Instead of only predicting the next token from left to right, you restructure the training data so the model has to predict a missing middle section given the start and the end.

The format looks like this:

```
<PREFIX> head of the function
<SUFFIX> return statement / closing logic  
<MIDDLE> ← model has to predict this
```

Four special tokens instead of one: prefix, suffix, middle, and end-of-text. And crucially, the middle section is **randomly placed** — different lengths, different positions, different amounts of context on either side.

I mixed this with the standard autoregressive approach, randomizing which format each training example used. The randomization was intentional — you don't want the model to see 50% AR then 50% FIM in a predictable pattern and start memorizing the structure. The format itself had to be something the model couldn't anticipate.

---

## What Changed After FIM

The model started stopping.

Not perfectly — not always at the right place — but the EOT token started actually firing during generation. It would stop somewhere in the 495–512 token range, which matched the training chunk size. That was the signal: the model had actually learned that EOT means stop, because now it had been seeing EOT at variable positions throughout training. The token had meaning now.

The model still couldn't write coherent multi-step logic, but it could now produce a plausible-looking function and stop. That's  a win in my books .

---

## Continued Pretraining: Teaching It to Stop Sooner

The next problem was that the model only knew how to stop around 500 tokens. Short functions would still get padded out with garbage until it hit that range. So the next step was continued pretraining on a dataset that mixed in shorter Python functions — same objective, just a different data distribution to shift the stopping behavior earlier.

This introduced a new balancing act:

- Too much short-function data → the model stops after 1–2 tokens. Outputs nothing useful.
- Too much long-function data → back to the autoregressive loop, stopping at 400–500 every time.

The learning rate mattered a lot here. A high learning rate meant the model snapped to the new distribution too fast — suddenly it was just producing one-liners. A very low learning rate let it shift slowly. The end result was a model that could produce shorter, more contained functions but would still occasionally drift into a long loop.

---

## The Docstring Problem

The other thing that surfaced during inference was the docstring bias. Because a large portion of the training data started with docstrings before any code, the model naturally favored producing docstrings over actual function bodies. It had seen more docstrings than code — so docstrings were the more probable output.

Temperature and top-p sampling helped here. By making the model pick less probable tokens, you could push it away from docstrings and toward actual code. But that's not really a fix — it's just adjusting inference to work around a data problem. The root cause was always the distribution. If the data is mostly docstrings, the model produces mostly docstrings. The inference parameters just let you trade "give me the most probable thing" for "give me something less obvious," which happened to be more useful in this case.

---

## The Real Takeaway

The gap between "model that produces text" and "model that produces useful code" is mostly a data and scale problem. 

Getting the model to stop required understanding why EOT wasn't firing — which required actually running experiments, not just reading theory. The FIM paper gave the direction, but working through *why* the autoregressive approach was failing first made the solution make sense rather than just being a thing to copy.

The docstring bias, the loss floor, the stopping problem — all of it traces back to data decisions made before training even started. Garbage in, garbage out isn't just a cliché. One data distribution choice went from "model that produces functions" to "model that produces docstrings 80% of the time."

The scale reality is also just humbling. A 70M parameter model trained on 1.5 billion tokens with the right techniques can produce plausible-looking Python. Getting it to actually reason about code would take orders of magnitude more compute and data than I had access to,it's just the reality of what it takes to build something genuinely useful in this space.

---

## What's Next

A demo exists — I'll update the hosting link here once it's in a better place. The next writeup in this series is going to go through building the transformer from scratch: starting from a recurrent neural network, moving to sequence-to-sequence, then attention, then the full transformer. It'll be a slow build but the goal is to actually show the whole progression rather than jumping straight to the finished architecture.
