# Notes

Using ChatGPT with YOUR OWN Data ([LangChain](https://github.com/langchain-ai/langchain) OpenAI API)

To help you ship LangChain apps to production faster, check out LangSmith. LangSmith is a unified developer platform for building, testing, and monitoring LLM applications. Fill out this form to speak with our sales team.




Modify `constants.py.default` to use your own [OpenAI API key](https://platform.openai.com/account/api-keys), and rename it to `constants.py`.

Place your own data into `data/data.txt`.


## Example usage
Test reading `data/data.txt` file.
```
> python chatgpt.py "what is my dog's name"
Your dog's name is Sunny.
```

Test reading `data/cat.pdf` file.
```
> python chatgpt.py "what is my cat's name"
Your cat's name is Muffy.
```

[Ref](https://www.youtube.com/watch?v=9AXP7tCI9PI&ab_channel=TechLead)