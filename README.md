[# LLM base AI Agent
Please replace API key with yours  
You can get Grog API key from [this site](https://console.groq.com/playground)

## Ollama
install Ollama from [this site](https://ollama.com)  
If you want to use Ollava, run the code below,

```
ollama pull llava
```

## How to run
Select a model which you want to use from the options below:
1. simpleSFM
2. llmagent
3. multimodal

Run the code with the selected model like this:
```python
poetry run python src/main.py multimodal
```

## File Explanation
- simpleSFM.py 
  - Implements a simple Social Force Model.
- llmagent.py
  - Text-only LLM-based AI Agent.
- main.py
  - Entry point for running the simulations.
- multimodal/
  - Contains code for the multimodal model.


## Reference
1. [今こそ学ぶLLMベースのAIエージェント入門](https://speakerdeck.com/os1ma/imakosoxue-bullmbesunoaiezientoru-men-ji-ben-de-nasikumi-slash-kai-fa-turu-slash-you-ming-naossyalun-wen-noshao-jie)
2. [A Survey on Large Language Model based autonomous Agents](https://arxiv.org/abs/2308.11432) ]
3. [ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
4. [LLaVa](https://ollama.com/blog/vision-models)