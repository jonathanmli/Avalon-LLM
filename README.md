# Communication Between Generative Agents

The following project explores communication between generative AI (large language models in particular) in natural social settings. 

ChatGPT took the world by storm when it came out, demonstrating impressive language and reasoning capabilities. However, one of the major challenges facing broad usage of these generative agents is that to elicit good answers one needs to provide good prompts. This has led to a flurry of new companies that specialize in prompt engineering, mostly through human prompt engineers. 

We hope to help solve this challenge by automating the prompt engineering process using LLMs. Moreover, through obversing how generative agents interact, we hope to ellucidate how LLMs reason and collaborate with other LLMs and with humans. 

## Literature

The literature is relatively new in this area, and there is a lot of enthusiasm for more research. 

### Generative collaborative agents

- [CAMEL](https://arxiv.org/pdf/2303.17760.pdf) Introduces a framework where two LLMs interact with each other to complete a task
- [Generative Society](https://arxiv.org/abs/2304.03442) Simulates LLMs in a fictional virtual village
- [Socially Aligned LLMs in Society](https://arxiv.org/pdf/2305.16960.pdf) Aligning LLMs through social interactions

### Playing games with LLMS

- [Repeated Gameplay with LLMs](https://arxiv.org/pdf/2305.16867.pdf) Let GPTs play repeated games with each other

### AI Multi-agent Gameplay

- [Diplomacy](https://arxiv.org/pdf/2010.02923.pdf) Presents an algorithm for effective play of diplomacy agents human and AI opponents
- [Avalon Assasin](https://arxiv.org/pdf/2209.09331.pdf) Training an AI to assassinate in Avalon

### Training LLMs with HF

- [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) Model for aligning AI outputs with human feedback
- [Self-Instruct](https://arxiv.org/pdf/2212.10560.pdf) Aligning LLMS using self generated instructions
- [Negotiation and Self-Play](https://arxiv.org/pdf/2305.10142.pdf) Can LLMS improve each other through a negotiation game?

### Hidden Identity Game Strategy

- [Percival Strategy](http://www.cs.cmu.edu/~ynakamur/fun/avalonstats.pdf) Analysis of strategy for percival

## Data

- [AvalonLogs](https://github.com/WhoaWhoa/avalonlogs) Logs of quantitative (voting, successes, etc.) data for human players for around 12k games
- [ProAvalon](https://www.proavalon.com/statistics) Logs can be availiable on request

## Questions
- What other alternatives to RLHF exist currently?