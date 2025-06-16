# Basic Learnings

<details>
   <summary><strong>AI for Everyone Course - by Andrew Ng on Coursera</strong></summary>

## 20/5/25
- ANI, Gen AI and AGI
- Supervised Learning
- How do LLMs work?
- Importance of Cleaning up data before feeding it to system
- ML v/s DS
- What is Deep Learning / Neural Networks?

## 21/5/25
- Starting an AI project: Workflows for ML and DS projects
- Brainstorming Framework: How can businesses use AI to be more efficient
- Build v/s Buy
- Working with an AI team
- Various Libraris/Tools: PyTorch, TensorFlow, HuggingFace, Paddle Paddle, Scikit-Learn, R, Research Pubilication on arxiv, Repos: Github
- Building AI in your company: Case Studies for Smart Speackers and Self-Driving Cars
- Different Roles for AI: Software Engineer, ML Engineer, ML Researcher, Data Scientist, Data Engineer, AI Product Manager
- AlexNet and its papers

## 22/5/25
- Execute Pilot Projcts: more important for initial projects to succeed rather than be most valuable
- Show traction within 6-12 months
- Who is CAIO: Chief AI Officer looks upon the in-house AI team which develops solutions for other business units
- Providing AI training for executives, senior business leaders, leaders of divitions and trainees too is very important
- Better Product -> More Users -> More Data -> Better Product and the cycle continues
- Don't be too optimistic or pessimistic about AI. It can't solve everything? At the same time, it can create great impact for very specific applications
- Get some friends to learn about AI
- Start brainstorming projects with them: No project is too small to start
- Areas of Impact: Computer Vision, NLP, Speech, Generative AI, Robotics, General ML, Unsupervised Learning, Transfer Learning, Reinforcement Learning, GAN, Knowledge graphs etc.
- Limitations of AI: Biases, performance issues, adversarial attacks, deepfakes etc.
</details>

<details>
  <summary><strong>How I use LLMs by Andrej Karpathy on YT</strong></summary>

  # [How I use LLMs -by Andrej Karpathy](https://www.youtube.com/watch?v=EWvNQjAaOHw)
  ## 23/5/25
  - [Mind Map for LLMs](https://github.com/qwerty-arun/Machine-Learning/blob/main/Resources/LLMs.svg)
  - User input -> LLM like ChatGPT (Generative Pre-trained Transformer) -> Output
  - Basically LLMs predict the next works in a sentence as we type.
  - How user input is divided into tokens? Use [Tiktokenizer](https://tiktokenizer.vercel.app/) to actually what's happening under the hood.
  - What is context window? It is like the working memory.
  - LLMs are usually out of date by a few months.
  - For every 1TB data trained on LLM, there will be trillions of parameters that can be fine tuned.
  - What is pre-training and what is post-training?

  ### Extra
  - I learnt about OpenAI's API keys. I will use them sooner of later.

  ## 24/5/25
  - In reinforcement learning, model discovers thinking strategies that leads to good outcomes.
  - Research Paper: Incentivizing Reasoning Capability in LLMs via Reinforcement learning.
  - All models of GPT like o1, o3-mini, o3-mini-high, 03-pro etc are "thinking" models.
  - If you want to do more complex tasks in math and coding, try "reasoning / thinking" models.
  - How does internet search work? This tool has the power to insert tokens into our context window.
  - Models can switch anytime to "web search" even if you don't specify it.
  - "Deep Research" is a combination of Internet Search and Thinking.
  - Try asking different models aobut recent news.

  ## 25/5/25
  ### Uploading documents feature
  - It may discard images
  - If present, it will not be well understood
  - Under the hood:
   ```mermaid
flowchart LR;
A[PDF]-->B[Text/Tokens];
B-->C[Context Window];
   ```
  - Use LLMs to read books faster and clearer
  ### Python Interpreter
  - For calculating big multiplications, GPT uses a Python interpreter, write code, calculates the answer, converts it to text and puts it in the context window.
  - Some other LLMs may not use code, it can directly do it using its brain. But, it may be wrong!

    ### Advanced Data Analysis
    - Ask GPT OpenAI valuation throughout the years in the form of table, then ask it to create graphs.
    - Also, this can be wrong! You need to dig a little bit.
    - But, it is actually powerful.
    ### Claud: Artifacts, Apps and Diagrams
    - Create 20 flashcards from a text
    - Tell it to create an quiz app for the flashcards. It basically adds UI to it
    - You can create mindmaps if you are a visual learner. It uses the Mermaid library. Same thing which I used above.
    ## Cursor - AI
    - You just have to give some prompts. The rest, it will take care. The composer will generate the code.
    - In chat, you can ask to explain a specific chunk of code to be explained.
    ## Audio Input/Ouput
    - You can either record your audio which gets converted to text which is then fed into the context window
    - Using Advanced Audio Mode: We can speak live and we give answers instantly
    - So, there are audio tokens getting exchanged
    - NotebookLM: You can add files, text, websites in the context window, then you can create a podcast out of it.
    ## Image
    - GPT splits the images into small patches, then the patches are added in series into the context window.
    - Output: DALL-E like models
    ## Video
    - You can "live call" with GPT and ask questions.
    - There are LLMs which generate videos. Eg: Sora
    ## GPT Memory
    - Saves memory about you from chat to chat
    - It can be invoked by us or it can happen automatically
    - "Can you please remember this".
    - You can customize GPT
    ## Custom GPT
    - You can create translators specifically for translation
    - You can ask it to give output in the form which you like by telling it before hand
    - You need not give the format again and again

    # Extra
    ## Natural Language Processing (NLP)
    - Natural Language Processing is a broader field focused on enabling computers to understand, interpret and generate human language.
    - Uses Sentiment Analysis, Named Entity Recognition and Machine Translation.

    ## Large Language Models (LLM)
    - Powerful subset of NLP.
    - Characterized by their massive size, training and ability to perform wide range of tasks.
    - Eg: Llama, GPT etc.
</details>

<details>
   <summary><strong>Python</strong></summary>

   # 26/5/25
   - Learnt basic arithmetic
   - Learnt basic syntax
   - True, False, bool(), None, is, ==, difference between `is` and `==`, chainin relational operators
   - Strings and their basics, f-strings
   ## How to actually learn python fast?
   - Week 1-2: Master the basics
   - Week 3-4: Write at least 30-40 programs
   - Never fall into "tutorial hell". When watching 70-80% of content, you actually only retain 10-20%.
   - Focus on 80-90% doing and only 20% tutorials.
   - Pick a Niche: Web Dev (Django, Flask, Fast API), Game Dev (Pygame), Data Analysis (Pandas, Numpy), Machine Learning (PyTorch, Tensorflow), Working with AI agents (LangChain, LangGraph), Automation scripts for daily tasks and Hardware Projects (Raspberry Pi).
   - After picking your niche, pick a project to work on.
   - Finish the damn project! Even if it is bad!
   - Advanced Python: List comprehensions, Generator expressions, Context managers, Dictionary and Set Operations, Decorators, Type Hints
   - Version Control your code properly
   - Deploy your projects: Web apps (Heroku, Railway or Render), Data Projects (Google Colab, Kaggle), Utilities (Docker)

   # 27/5/25
   - Wrote some basic programs: Odd/Even, Leap Year, Sum of Digits, Armstrong no., Prime no.s in a range, Print triangle patterns, Number Reversal, Fibonacci sequence, Remove duplicate characters in a string
   - Learnt about string functions in python
   - Learnt some tips and tricks along the way

   # 28/5/25
   - Wrote Some more programs: calculator, number guessing, password strength checker, capitalize first letter of each word in a sentence, word frequency in a sentence etc.
   - Learnt new concepts: functions, some cool operations on lists and even strings
   - There are crazy functions out there!!
   ## General
   - Applied for "AI essentials" course on Coursera. Seems worth it.
   - Revised some previous AI learnings, python syntax etc.

   ## 31/5/25
   - Wrote around 6-7 programs
   - Transpose problem, flatten lists, divide the list into chunks, merge sorted lists into one, anagram strings, removing palindrome strings from sentence, reverse each word in strings.
   - Learnt about tuples and dictionaries and basic operations
   - Searched about projects which I can do using python. There are so many options!
</details>

<details>
   <summary><strong>Deep Dive into LLMs like ChatGPT - by Andrej Karpathy</strong></summary>

   # [Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI?list=PLv6a69CxXDO_adRH9DQdvgjvAI_b8MdhQ)

   ## 30/5/25
   - [FineWeb Dataset](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
   ### Pre-Training
   - Download and preprocess the internet
   - Filtering Mechanisms in place: URL filtering (remove adult content), Text Extraction (just raw HTML), Language Filterng (More than 65% English webpages) and other custom filters.
   - After filtering you end up with TBs worth of text from billions of webpages. This is the starting point for training the model.
   ### Tokenization
   - How to feed it to Neural Network? You need to arrange the text as a 1D sequence of characters.
   - Then apply utf-8 code on the text.
   - We don't just two symbols (0 and 1) and extremely long sequences. Instead, we want more symbols and shorter sequences.
   - If we group 8 bits as 1 byte? We have 8 times smaller sequence but we now have 256 possible symbols. We call this "vocabulary". We need to even smaller sequences.
   - We can go even further. It turns out `116 32` is most recurring byte pair. So, we group it now. This is called "Bit Pair Encoding Algorithm".
   - [Tiktokenizer](https://tiktokenizer.vercel.app/). Choose cl100k_base model. Type some words to see the tokens. Play around with capitalization. It is case sensitive.
   ### Neural Network Training
   - We take a window of tokens: say 8000 tokens. The length of the window is our choice.
   - Then we try to predict the next token in the sequence. We assign probabilites for next token. Since GPT's vocabulary is 100,277 tokens, we assign those many probabilities.
   - If we already know what's coming next and the NN gives a lower probability for that token, we can fine tune the model mathematically in such a way that the correct answer gets higher probability and all others lower probabilities.
   ### Neural Network Internals
   - You have a window of size: 8000 and on the other hand, you have a billion parameters (randomly set at the start)
   - We feed them into a giant mathematical expression and it emits probabilities
   - [LLMs Visualization](https://bbycroft.net/llm)
   - You can see various weights/parameters. They are just mathematical functions too, they don't have any memory. Its just vector products, matrix multiplications etc.
   - We have to fine tune these weights to get the desired output.
   ### Extra
   - Downloaded "Attention is all you need" paper.
   - Will read it some time.

   ## 31/5/25
   ### Inference
   - To generate data, just predict one token at a time.
   - Start with a token, feed it to a Neural Network, then we sample the tokens of highest probabilities from the probability vector.
   ```mermaid
    graph LR
    A[91] -->|Neural Network| B(( Probablities ))
    B -->|sample| C[860]


    D[91 and 860]--> |Neural Network| E(( Probablities ))
    E-->|Sample| F[287]


    N[91, 860 and 287]--> |Neural Network| G(( Probablities ))
    G-->|Sample| H[11579]
   
    I[91, 860, 287 and 11579]--> |Neural Network| J(( Probablities ))
    J-->|Sample| K[13659]
   ```
   - As we can see, it is not the right answer. It is 3962.
   ### Reproducing OpenAI's GPT-2
   - GPT-2 was published in 2019
   - Paper: "Language Models are Unsupervised Multitask Learners"
   - 1.6 Billion parameters
   - Maximum context length of 1024 tokens
   - Trained on about 100 billion tokens
   - [Reproducing GPT-2](https://github.com/karpathy/llm.c/discussions/677)
   - Before the cost of training was about $800, now you can do it within $100. The reason is that datasets have become much better due to filtering mechanisms.
   - You can't train the model on your laptop! You need a GPU. If you don't have one, you can rent one on [Lambda](lamdalabs.com)
   - Search "Biggest LLM base models"
   - Llama 3 (2024): 405 billion parameters trained on 15 trillion tokens
   - [Paper: "The Llama 3 Herd of Models"](https://arxiv.org/pdf/2407.21783)
   - [Hyperbolic](app.hyperbolic.xyz)
   - How is a base model different from an assistant? To a base model, you can't ask a question and expect a reply. Your prompt will be tokenized and fed to the Neural Network. Then, it is just autocompleting the next tokens. What you get is just a recollection of its past "memory" that it was previously trained on.
   - These models are very good at memorization. Ask it a sentence on wikipedia and it will give you the rest. But, eventually, it will deviate.
   - Prime the model about stuff from the future (a date after its knowledge cutoff) and see what happens.
     
   ## 1/6/25
   - The models have "in-context" learning abilities. By learning about the context, they can answer more questions about what comes next.
   - Suppose, I give it 9 pairs of English words with its translation in Hindi and for the 10th pair, I just give it the English word. By recognizing the pattern in its input, it gives me the translation in Hindi for the 10th word.
   - If you want a base model to work as an assistant, you can give it a two-person (human and chatbot) conversation turn by turn. And at the end, just give a prompt: a question. Now, it will take on the role of an assistant and then it will answer your question.
   ### The "psychology" of a base model
   - It is a token-level internet document simulator
   - It is a probabilistic - you're going to get something else each time you run
   - It "dreams" internet documents
   - It can also recite some training documents verbatim from memory ("regulation")
   - The parameters of the model are kind of like a lossy zip file of the internet -> a lot of useful world knowledge is stored in the parameters of the network
   - You can already use it for applications (e.g. translation) by being clever with your prompts
   ### Post-training
   - We are now using a new data set of conversations
   - This training period will be much shorter: in the matter of hours
   - The pre-training stage will be around 3 months of time
   - The data sets are made manually and fed into the same Neural Network
   - Again, it is time to visualize `gpt-4o`on [Tiktokenizer](tiktokenizer.vercel.app)
   ```
   User: What is 2+2?
   Assistant: 2+2=4
   User: What if it was *?
   Assistant: 2*2=4, same as 2+2!
   ```
   - The tokens are as follows:
   ```
   <|im_start|>user<|im_sep|>What is 2+2?<|im_end|><|im_start|>assistant<|im_sep|>2+2=4<|im_end|><|im_start|>user<|im_sep|>What if it was *?<|im_end|><|im_start|>assistant<|im_sep|>2*2=4, same as 2+2!<|im_end|><|im_start|>assistant<|im_sep|>
   ```
   - `im_start` stands for "imaginary monologue start", then it is `im_sep` and at last it is `im_end`. All of these are new and special.
   - [2022 Paper](https://arxiv.org/abs/2203.02155): "Training language models to follow instructions with human feedback" or ***InstructGPT***
   - [Hugging Face Interface Playground](https://huggingface.co/spaces/huggingface/inference-playground)
   - Rules that AI organizations should follow for a chatbot to be: respectful, truthful, harmless, and helpful assistant etc.
   - The assistant will take on the persona according to the dataset it is trained on. So, it is very important what kind of data we are dealing with.
   - Now companies don't write conversations from scratch, they just use other LLMs which do it with ease. For example: UltraChat
   - Asking a LLM is basically like asking an human labeler. A chatbot is like a simulation of a human labeler.
   - When you ask a question, there is no infinite intelligence there. What you are getting in response is a statistical simulation of a labeler that was hired by OpenAI.

   ## 2/6/25
   ### Hallucinations
   - You feed conversations like: User: "Who is `person name` ?" and Assistant: `person name` is a ..., and then ask completely random name, it will hallucinate.
   - But the style will match with the training set it was given.
   - How do we know what a model knows and what it doesn't know? We can probe it.
   - [Meta Paper: The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783) tells us how it dealt with hallucinations. Refer to "facuality" section.
   - Prompt: A paragraph from an article. Ask the LLM to generate 3 factual questions based on it and also generate the correct answer.
   - Compare it with other LLMs answer (acting as a judge).
   - If there is a hallucination, then take the same question, add it to the dataset, the correct answer will now be: "I don't know".
   - Repeat this for a number of questions.
   - If the model doesn't know, allow the model to search. Introduce a search token: `search_start` for example.
   - When the model sees this, it will stop generating and goes to "search" the internet. It retrieves the text, copy paste into the context window.
   - `Vague Recollection`: Knowledge in the parameters (something you read a month ago)
   - `Working Memory`: Knowledge in the tokens of the context window
   ### Knowledge of Self
   - What model are you? Who built you? are not sensical questions.
   - If model is not trained on such questions, it will give rubbish.
   - Invisible tokens are there to remind models about its identity.
   ### Models need tokens to think
   - Human: "I buy 3 apples and 2 oranges. Each orange costs $2. The total cost of all the fruit is #13. What is the cost of apples?
   - Assistant Answer-1: The answer is $3. This is because 2 oranges at $2 are $4 total. So the 3 apples cost $9, and therefore each apple is 9/3 = $3.
   - Assistant Answer-2: The total cost of oranges is $4. 13-4 = 9, the cost of the 3 apples is $9. 9/3 = 3, so each apple costs $3. The answer is $3.
   - For nano-gpt, there is fixed amount of computation happening for each token. We need to distribut our reasoning and our computation across many tokens because every single token is spending a finite amount of computation on it.
   - The first answer is worse. In the first sentence itself, it has the answer and it will be stored in the context window. The later sentences justify that answer. There is no computation that is happening. If you train the model like this, what you are doing is making the model to basically guess the answer in a single token because of the finite amount of computation that can happen per token.
   - Therefore, answer-2 is much better because the computation and reasoning is spread across tokens.
   - You can force a model to produce an answer in a single token by literally asking for it. It will do it but the answer will be wrong.
   - Models can "mentally think" or can use "code". It so happens that the intermediate steps can actually go wrong in the mentally thinking case. So you can ask the model to use code to verify the answer.

   ## 6/6/25
   ### Models need tokens to think (Contd.)
   - Models are very bad at counting. This also forces models to give answers in a single token.
   - Tell it count the dots in `................................................................................` for example. Then compare it with answer from code. The code answer will be correct.
   ### Models are not good with spelling
   - Remember they see tokens (text chunks), not individual letter.
   - Ask it to print every nth character from a string. For example, "Arithmetic" and every 3rd letter.
   ### Random Stuff
   - `What is bigger 9.11 or 9.9?` would previously result in 9.11 as the answer, but now it's correct.
   - It seems like it is related to the verses on the Bible. The verse 9.11 would come after 9.99 and so on.

   ### Reinforcement Learning and Supervised Finetuning (SFT model)
   - Think of this as a textbook that you were reading at school. Similarly, the models need to go to "school". First learn theory, then solved problems. Then test yourself by solving unsolved problems. Then when you make error, you learn by reinforcement. Same thing goes with models.
   - Exposition $\Leftrightarrow$ pretraining (background knowledge)
   - Worked Problem $\Leftrightarrow$ supervised finetuning (problem + demonstrated solution, for imitation)
   - Practice Problems $\Leftrightarrow$ Reinforcement Learning (prompts ot practice, trial and error until you reach the correct answer)
   - Ask a question many times. Get like 20 solutions, only 5 of them might be right. So take the top solution (each right and short), train on it. Repeat many, many times.
   - The model's parameter will get adjusted to this type of behaviour for that kind of questions.
   - [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning talks aobut RL models](https://arxiv.org/pdf/2501.12948) talks about how it performed reinforcement learning.
   - How models are fine-tuned to get more accurate.
   - Model was trying different ways to think through the same problem. This in turn increased the response length. But on the other hand, it increased its accuracy.
   - [together.ai](https://api.together.xyz/signin?redirectUrl=%2Fplayground%2Fv2%2Fchat) playground hosts many different models.
   - AI studio by google. Try it out.
   - [AlphaGo](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf). See how it became a very good player at go.
   - There are thousands of matches that it learns from. Basically you are imitating human players, therefore there is a certain limit of ELO rating after which you can't cross. This is the case for Suupervised Learning.
   - But, in the case of Reinforcement Learning, it gets even better.
   - Search about `move 37 alphago`, an extremely rare move that no human would play. But it turns out that it was a great move!

   ### Reinforcement Learning in un-verifiable domains (RLHF: Reinforcement Learning from Human Feedback)
   - [Human Preferences by OpenAI](https://arxiv.org/pdf/1909.08593)
   - Make a model to write a joke. It turns out they are bad at it. Also, how to rate these jokes?
   - Run RL as usual, 1000 prompts of 1000 rollouts. That is worse.
   - Or create a neural net simulator (see LLM chart in Resources folder) of human simulators and let it rate the jokes.
   #### Advantages of RLHF
   - We can run RL, in arbitrary domains! Even for the unverifiable ones.
   - There is something called the discriminator - generator gap.
   - Write a story v/s Which of these 5 poems if best? Turns out, it is much easier to discriminate than to generate.
   #### Downside
   - We are doing RL with respect to a lossy simulation of humans. Can be misleading after many iterations.
   - Results become non-sensical and also the reware system might rate it high.
   - This happens when RL runs for too long. It finds a way to get really high scores with nonsensical results.
   - Crop it after a certain time!
   - Keeping track of LLMs: [lmarena.ai](lmarena.ai)
   - [AI News](news.smol.ai)
   - Where to find them: proprietary models (on respective websites of the LLM providers)
   - Open weights models (DeepSeek, Llama): an inference provider
   - We can run them locally! [LMStudio](lmstudio.ai) 
</details>

<details>
   <summary><strong>Numpy</strong></summary>

   # [Numpy documentation](https://numpy.org/doc/stable/user/basics.creation.html)
   ## 15/6/25
   ### Numpy Basics
   - Array Creation
   - Array properties: Shape, Dimension, Size, dtype etc.
   - Special arrays: zeros, ones, eye, arange, linspace etc.
   - Randome arrays
   - Array operations: +, *, **, np.dot(a,b)
   - Indexing and Slicing
   - Reshaping and Transposing
   - Boolean Indexing
   - Aggregation Functions
   ## 16/6/25
   ### Beginner Practice Programs
   - Create a 1D array of numbers from 0 to 9.
   - Create a 3x3 matrix with values ranging from 0 to 8.
   - Create a 5x5 identity matrix.
   - Generate a random number between 0 and 1.
   - Create a 10-element array of zeros and set the fifth value to 1.
   ### Intermediate Practice Programs
   - Create a 10x10 array with random values and find the minimum and maximum values.
   - Normalize a 5x5 random matrix (values between 0 and 1).
   - Multiply two matrices (2x3) and (3x4).
   - Replace all odd numbers in an array with -1.
   - Create a checkerboard 8x8 matrix using slicing.
   ### Advanced Programs
   - Find the most frequent value in an array.
   - Write a function that computes the moving average over a 1D array.
   - Rotate a matrix by 90 degrees.
   - Solve a linear system Ax = b using np.linalg.solve().
   - Create a Mandelbrot set using NumPy and matplotlib.
</details>
