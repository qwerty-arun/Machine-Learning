  # [How I use LLMs - by Andrej Karpathy](https://www.youtube.com/watch?v=EWvNQjAaOHw)
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
