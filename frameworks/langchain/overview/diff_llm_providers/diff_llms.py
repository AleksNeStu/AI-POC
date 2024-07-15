import os

from langchain.llms import Cohere
# from langchain_cohere.llms import Cohere
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from common.cfg import *

# https://python.langchain.com/v0.2/docs/integrations/providers/

chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo')
# https://platform.openai.com/docs/deprecations
gpt3 = OpenAI(model_name='gpt-3.5-turbo-instruct')
# https://python.langchain.com/v0.2/docs/integrations/providers/cohere/
cohere = Cohere(model='command-xlarge')
# https://python.langchain.com/v0.2/docs/integrations/providers/serpapi/

# https://huggingface.co/bigscience/bloom-1b7
bloom = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={
        "temperature": 0.1, "max_length": 64,
        "do_sample": True
    }
)

human_msg_txt = 'What is a AI applications developer roadmap in 2024?'
human_msg = HumanMessage(content=human_msg_txt)

def ask_qns():
    try:
        chatgpt_res = chatgpt([human_msg])
        chatgpt_res_act = ('''
        content='As of 2024, an AI applications developer roadmap may look something like this:\n\n1. Understanding the 
        latest trends and advancements in AI technology, such as deep learning, natural language processing, 
        computer vision, and reinforcement learning.\n\n2. Mastering programming languages commonly used in AI 
        development, such as Python, Java, and C++.\n\n3. Learning how to effectively use popular AI frameworks and 
        libraries, such as TensorFlow, PyTorch, and scikit-learn.\n\n4. Developing a strong foundation in data science 
        and machine learning concepts, including data preprocessing, feature engineering, model evaluation, 
        and optimization.\n\n5. Building expertise in developing AI applications for specific industries and use cases, 
        such as healthcare, finance, autonomous vehicles, and robotics.\n\n6. Collaborating with cross-functional teams 
        to design, develop, and deploy AI solutions that meet business requirements and deliver value to end users.\n\n7. 
        Staying updated on ethical and regulatory considerations in AI development, including privacy, security, bias, 
        and fairness.\n\n8. Participating in AI competitions, hackathons, and conferences to network with peers, 
        showcase skills, and stay at the forefront of the field.\n\n9. Continuing education through online courses, 
        workshops, and certifications to expand knowledge and skill set in AI development.\n\n10. Exploring opportunities 
        for specialization in niche areas of AI, such as AI ethics, explainable AI, AI for social good, 
        and AI governance.' response_metadata={'token_usage': {'completion_tokens': 284, 'prompt_tokens': 19, 
        'total_tokens': 303}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 
        'logprobs': None} id='run-bcba435c-e761-4f2e-bb2c-e8c61244d0e3-0' usage_metadata={'input_tokens': 19, 
        'output_tokens': 284, 'total_tokens': 303}
        ''')
        gpt3_res = gpt3(human_msg_txt)
        gpt3_res_act = ('''
    
    In 2024, an AI applications developer roadmap may include the following:
    
    1. Strong foundation in computer science: As AI technology advances, a strong understanding of computer science 
    principles will become even more important for AI developers. This includes knowledge of programming languages, 
    data structures, algorithms, and software development methodologies.
    
    2. Advanced knowledge of AI and machine learning: AI developers will need to have a deep understanding of AI and 
    machine learning techniques, such as neural networks, natural language processing, and deep learning. They will also 
    need to keep up with the latest advancements in these fields.
    
    3. Specialization in a specific industry or domain: In order to create effective AI applications, developers will 
    need to have a strong understanding of the industry or domain they are working in. This could include healthcare, 
    finance, transportation, or any other industry that can benefit from AI technology.
    
    4. Experience with big data and cloud computing: AI applications often require large amounts of data to train and 
    improve their performance. Developers will need to have knowledge of big data technologies and cloud computing 
    platforms to manage and process this data effectively.
    
    5. Understanding of ethical and legal implications: As AI technology becomes more advanced and integrated into 
    various industries, developers will need to consider the ethical and legal implications of their applications. This
        ''')
        cohere_res = cohere(human_msg_txt)
        cohere_res_act = ('''
         As an AI chatbot, I cannot generate a roadmap for AI applications development, however, here is a general 
         outline you can use as a starting point: 
    
    1. Acquire knowledge - Familiarize yourself with the foundational concepts of AI, including machine learning 
    algorithms, natural language processing, computer vision, and data analytics, to make you become capable of working 
    within these technologies. 
    
    2. Hands-on Practice - Consistently engage in AI projects allowing you to demonstrate your capabilities and deepen 
    your understanding of AI tools and techniques, boosting your experience and showcasing your problem-solving abilities. 
    
    3. Business Acumen - Study business and entrepreneurship to become adept at identifying problems, determining when AI 
    can provide solutions, and articulating these solutions to stakeholders. 
    
    4. Stay Current - Continue to upgrade your knowledge and skills by monitoring the AI industry, including the latest 
    tools, techniques, and best practices, as well as emergent trends and technological disruptions. 
    
    5. Portfolio and Networking - Document your projects and their results in a portfolio highlighting your AI expertise 
    and your abilities to solve real-world problems. Also, build connections within the AI community by participating in 
    networks, forums, and conferences, which can reveal potential job opportunities or collaborators for projects. 
    
    6
        ''')
        bloom_res = bloom(human_msg_txt)
        bloom_res_act = ('''
        What is a AI applications developer roadmap in 2024? What are the key challenges and opportunities for AI 
        applications developers in the coming years? What are the key trends and challenges for AI applications 
        developers in the coming years? What are the key trends and challenges for AI applications developers in the 
        coming years? What are the key trends
        ''')
    except Exception as ex:
        print(ex)
        # raise ex

if __name__ == '__main__':
    ask_qns()
