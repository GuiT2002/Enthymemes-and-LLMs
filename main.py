import re
from together import Together
import warnings
import openai
from groq import Groq

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Client
#client = Together(api_key='')
#client = openai.OpenAI()
client = openai.OpenAI(api_key="", base_url="https://api.deepseek.com")
#client = Groq(api_key='')



args_classes = ['as4ABC', 'as4DStd', 'as4E2C', 'as4NH', 'as4PFAO', 'as4VoVC', 'asMJAdPA', 'classification', 'necessary_condition', 'position_to_know']
models_list = ['deepseek-reasoner', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct', 'deepseek-chat']

# Model and argumentation scheme
#model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
#arg_s = args_classes[0]
class_list = ['1', '2', '3', '4']
#model = models_list[2]
class_ = class_list[3]



# Prompt template
test_template = """[INST]
You will receive an argument in natural language which is an enthymeme, where conclusion or premises might be missing.
You must translate the given argument to a computational form, represented in the context below, by reasoning through the present argumentative sentences and then inferring the context of the rest of the argumentation scheme.
The context contains an example in natural form, stating all the premises from the argument, and its computational representation for a variety of numbers of argumentation schemes.
You must correctly identify to which argumentation scheme the presented argument belongs. The text belongs to only one of the argumentation schemes provided in the examples.
FULLY Instantiate ALL the variables and predicates from the argumentation scheme given the natural language enthymeme provided below by inferring the missing components. 
The final computational representation of the argument MUST BE EXACTLY in the following format and not have any kind of irrelevant text after: "Final answer: <computational argument>".

Context: [ {context} ]

Text: {question}

[/INST]
"""

def get_final_answer(llm_response: str) -> str:
    marker = "Final answer:"
    index = llm_response.find(marker)
    
    if index == -1:
        # "Final answer: " not found; return the original text
        return llm_response
    
    # Move the start to the end of the marker
    start_idx = index + len(marker)
    
    # Return the text after "Final answer: "
    return llm_response[start_idx:].strip()


def generate(query: str, context: str) -> str:
    prompt = test_template.format(context=context, question=query)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens= 8000,
        temperature=0.0
    )
    return response

def apply_filter(text: str) -> str:
    return re.sub(r'("[^"]*")|.', lambda m: m.group(1) or m.group(0).lower().replace('\\', '').replace(' ', '').replace('-', '_'), text)

def testing_input(queries: list, context: str, model: str, class_):
    for query in queries:
        print("\033[32mQ:", query, "\033[0m\n")
        response = generate(query, context)
        final_response = response.choices[0].message.content
        print('A:', final_response, '\n')
        #filtered_output = apply_filter(response.strip())
        
        with open(f'deepseek-reasoner/class_{class_}/{arg_s}_tests.txt', 'a', errors='ignore') as file:
            file.write(get_final_answer(final_response) + "\n")

        if model == "deepseek-reasoner":
            reasoning_response = response.choices[0].message.reasoning_content
            with open(f'{model.replace("/", "-")}/class_{class_}/{arg_s}_tests_reasoning.txt', 'a', errors='ignore') as f:
                f.write(reasoning_response + "\n\n\n")




# Load context examples
with open(f'examples/all_examples.txt', encoding='utf-8', errors='replace') as f:
    context_examples = f.read()


arg_s = args_classes[3]
for model in models_list:
    # Load test cases
    with open(f'tests/class_{class_}/{arg_s}.txt', encoding='utf-8', errors='replace') as f:
        test_cases = [line.strip() for line in f.readlines() if line.strip()]

        testing_input(test_cases, context_examples, model, class_)
