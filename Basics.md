# All Learnings of May 2025

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
  <summary><strong>How I use LLMs by Andrew Karpathy on YT</strong></summary>

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
   - First Stage:
   ```mermaid
graph LR
    A[91] -->|Neural Network| B(( Probablities ))
    B -->|sample| C[860]
   ```
   - Second Stage:
   ```mermaid
   graph LR
   A[91 and 860]--> |Neural Network| B(( Probablities ))
   B-->|Sample| C[287]
   ```
   - Third Stage:
   ```mermaid
   graph LR
   A[91, 860 and 287]--> |Neural Network| B(( Probablities ))
   B-->|Sample| C[11579]
   ```
   - Fourth Stage:
   ```mermaid
   graph LR
   A[91, 860, 287 and 11579]--> |Neural Network| B(( Probablities ))
   B-->|Sample| C[13659]
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
   - 
</details>
