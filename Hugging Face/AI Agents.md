# [AI Agents](https://huggingface.co/learn/agents-course/unit0/introduction)

## Course Structure
- Understanding Agents
- Role of LLMs in Agents
- Tools and Actions
- Agentic Workflow
- Code your first agent
- Publish it on HF spaces

# What is an Agent?
- An agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions to fulfill tasks.
- User Input -> Reason -> Plan -> Execute with tools

## Two main parts of an Agent

### 1) The Brain (AI Model)
- The thinking happens here.
- The model handles reasoning and planning.
- It decides which **actions** to take based on situation.

## 2) The Body (Tools and Capabilities)
- Everything the Agent is equipped to do.
- The scope of possible actions depends on what the agent has been equipped with.

## Types of Agents
- Simple Processor: Agent o/p has no impact on program flow
- Router: Determines basic control flow
- Tool Caller: Determines function execution
- Multistep agent: Controls iteration and program continuation
- Multi-agent: One agentic workflow can start another agentic workflow
- Code Agent: LLM acts in code, can define its own tools / start other agents

## Extras
- AI agents are programs where LLM outputs control the workflow
- Most common AI models are: Text as input -> Text as ouput
- The design of tools is very Important and has a great impact on the quality of your agent
- Actions != Tools
- Actions use multiple tools for execting a task

### Personal Virtue Assistants
- Siri, Alexa
- Take user queries, analyze context, retrieve info from databases, provide responses or initiate actions

### Customer Service Chatbots
- Troubleshoot, open issues, or complete transactions

### AI Non-Playable Characters in a video game
- More lifelike characters

# Large Language Models (LLMs)
- Type of AI model that excels at understanding and generating human language
- Most of them are built on Transformer Architecture based on "Attention" Algorithm
- [Attention is all you need](https://arxiv.org/pdf/1706.03762)

## Types of Transformers
### 1) Encoders
- Text as input
- Outputs a dense representation of text
- Classify text, sematic search, named entity recognition
- Millions of parameter
- Ex: BERT from Google

### Decoders
- Focuses on generating new tokens to complete a squence, one token at a time.
- Llama
- Chatbots, Text Generator, Code Generator
- Billions of Parameters

### Seq2Seq (enc-dec)
- Combines encoder and decoder
- Encoder processes input sequences to a context representation
- Decoder generates an output sequence
- Translation, summarization and paraphrasing
- Millions of parameters

#### Principle of LLMs: Predict the next token, given a sequence of previous tokens
- Token is a unit of information: sub-word units
- Each LLM has special tokens specific to the model.
- The LLM uses tokens to open and close the structured components of its generation.

## Token Prediction
- LLMs are auto-regressive. Output from one pass becomes input for the next one.
- LLM will decode text until it reaches EOS (end of sequence)
- Once input text is tokenized, the model computes a representation of sequence that captures info about meaning and position of each token in input sequence.
- This representation goes into model, probabilities are assigned to each token in its vocabulary.
- Highest -> choost it.

## Beam Search Visualizer
- Explores multiple candidate sequences to find the one with maximum total score even if some individual tokens have lower scores
### Parameters
- Sentence to decode from (input)
- Number of steps
- Number of beams
- Length of penalty. `> 0.0` will promote long sequences and `< 0.0` will promote shorter ones.
- Number of return sequences -> to be returned at end of generation should be <= num_beams

## Attention is all you need
- Key aspect of Transformer architecture is Attention. While predicting next word, not every word in a sentence is equally important.
- Ex: "The <ins>capital</ins> of <ins>France</ins> is ...". `capital` and 'France` hold the most meaning.
- Context Length: Max number of tokens the LLM can process, and the max attention span it has.

## How can you use LLMs?
- Run locally
- Use a cloud/API (HF Serverless interface API)

## Messages and Special Tokens
- Exchanging messages.

### System Messages
- Define how the model should behave.
- `You are a professional customer service agent. Always be polite, clear and helpful.`
- Also gives info about the available tools, provides instructions to model on how to format the actions to take and includes guidelines on how the thought process should be segmented.
- Chat Templates help maintain context by preserving conversation history, storing previous exchanges between user and assistant.
- Concatenate all the messages in conversation and pass it on to LLM as single-stand-alone sequence.

### Chat Templates
- Essential for structuring conversations between language models and users.

### Base Model
- Trained on raw text data to predict the next token.

### Instruct Model
- Fine-tuned specifically to follow instructions and engage in conversations.
- To make a base model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand.
