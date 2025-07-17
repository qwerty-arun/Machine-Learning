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
