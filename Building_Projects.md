<details>
   <summary><strong>Let's build GPT: from scratch, in code, spelled out - by Andrej Karpathy</strong></summary>
  
# Pre-requisits
- Good understanding of python
- Hands-on experience with pytorch
- Study LLMs and NN (entire playlist)
- Colab and Jupyter Notebook experience
- Read `Attention is all you need` paper 

# Resources 
## [Let's build GPT: from scratch, in code, spelled out -by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLv6a69CxXDO_adRH9DQdvgjvAI_b8MdhQ&index=2)
## [Google Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=nql_1ER53oCf)
## [Pytorch](https://pytorch.org/)
# 15/6/25
- Download Shakespeare Dataset
- Print how many characters in the dataset and the length of dataset
- Sort the characters and make them unique
- Encode and Decode (char to int mapping and vice versa). Building a very simple tokenizer
- Google uses `SentencePiece` and OpenAI uses `tiktoken`
- Split the data into training and validation sets (90% + 10%)
- Feed the data in chunks of block size = 8
- Make a batch size of 4. 4 chunks of data is fed in parallel to the GPUs.
</details>
