<h1>RadABench (continuing ...)</h1>

<img src="./assets/RadA.png" alt="logo" style="height:60px; vertical-align: middle;margin-bottom: -10px;"> 
The official code inplementation for "Can Modern LLMs Act as Agent Cores in Radiology Environments?"

[Paper (arXiv)](https://arxiv.org/pdf/2412.09529)

[RadABench-Data (huggingface)](https://huggingface.co/datasets/QiaoyuZheng/RadABench-Data)

This paper aims to investigate the pre-requisite question for building concrete radiology agents which is, **"Can modern LLMs act as agent cores in radiology environments?"**

To investigate it, we introduce **RadABench** with three-fold contributions: 
1. We present **RadABench-Data**, a comprehensive synthetic evaluation dataset for LLM-based agents, generated from an extensive taxonomy encompassing 6 anatomies, 5 imaging modalities, 10 tool categories, and 11 radiology tasks. 
2. We propose **RadABench-EvalPlat**, a novel evaluation platform for agents featuring a prompt-driven workflow and the capability to simulate a wide range of radiology toolsets.
3. We assess the performance of 7 leading LLMs on our benchmark from 5 perspectives with multiple metrics. 


<div>	
    <Center>
    <img src="/assets/AgentWorkflow.png"
         alt="CoRe模型图片缺失"
         style="80%"/>
    <br>	
    </Center>
</div>

## Our Conclusion
- **Challenges in understanding complex external tools:** LLMs struggle to interpret and apply instructions involving long and detailed contextual descriptions. This limitation is particularly problematic in radiology, where tool instructions and diagnostic criteria often require sustained coherence and nuanced understanding.
- **Inefficiencies in synthesizing multi-round information**: Performance degrades markedly as response rounds increase. This limits the models’ ability for iterative diagnostic processes and longitudinal patient monitoring, both of which demand consistent tracking and integration of information.
- **LLMs are prone to significant "tool incomplete hallucinations"**: When working with external tools, LLMs often generate erroneous or incomplete outputs—referred to as "hallucinations"—especially when these tools are not fully integrated or accessible. These hallucinations can mislead clinicians or undermine the trustworthiness of the AI system in clinical decision-making, a crucial concern in high-stakes medical environments.
- **LLMs struggle with organizing strict IO formats for successive tools**: LLMs often fail to precisely follow complex instructions, especially those that require systematic organization of IO to link different tools. In radiology, where diagnostic workflows may involve multiple stages, for example, imaging analysis, report generation, and treatment recommendation, LLMs are unable to strictly organize tasks and link tools in a coherent manner. 
- **LLMs often fail to select the most appropriate tools based on their performance**: A key aspect of effective agent systems in radiology is the ability to evaluate and select the best tools based on objective performance metrics. Existing LLMs often make suboptimal choice, that can compromise diagnostic accuracy and overall system performance.
- **Closed-source LLMs remain superior than open-source alternatives**: In our evaluations, closed-source LLMs consistently demonstrated better performance than their open-source counterparts. This may be due to proprietary optimizations, access to higher-quality training data, or more advanced model architectures, all of which contribute to a higher level of reliability and clinical utility.
