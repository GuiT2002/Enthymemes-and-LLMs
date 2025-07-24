from together import Together
import warnings
import openai
from groq import Groq


warnings.filterwarnings("ignore", category=DeprecationWarning)

# TogetherAI client
client_gq = Groq(api_key='')
#client = openai.OpenAI()
client_ds = openai.OpenAI(api_key="", base_url="https://api.deepseek.com")


args_classes = ['as4ABC', 'as4DStd', 'as4E2C', 'as4NH', 'as4PFAO', 'as4VoVC', 'asMJAdPA', 'classification', 'necessary_condition', 'position_to_know']
models_list = ['deepseek-reasoner', 'deepseek-chat', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8']
tests_models = ['deepseek-ai-DeepSeek-V3', 'deepseek-reasoner', 'meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8']


# Model and argumentation scheme
class_list = ['1', '2', '3', '4']
class_ = class_list[0]
test_model = tests_models[0]

with open(f'{test_model}/class_{class_}/all_tests.txt', 'r', encoding='utf-8', errors='replace') as f:
    tests_list = [line.strip() for line in f.readlines() if line.strip()]

with open('all_answers.txt', 'r', errors='replace', encoding='utf-8') as p:
    answers_list = [line.strip() for line in p.readlines() if line.strip()]

with open(f'tests/class_{class_}/all_tests.txt', 'r', encoding='utf-8', errors='replace') as g:
    nl_tests_list = [line.strip() for line in g.readlines() if line.strip()]


def filtering_output(text):
    keyword = "Semantics:"
    index = text.find(keyword)
    if index != -1:
        return text[index:]
    return ""

# Evaluation prompt
evaluation_prompt = """[INST]

You will be the evaluator of an enthymeme reconstruction task. Another LLM received the task of receiving an argument with premises and/or conclusion missing (an enthymeme) and reconstruct the missing components by inference, on a specific computational format.
A brief explanation of the prompt that the LLM received: The task is to receive a natural-language enthymeme with missing premises or conclusion, identify its argumentation scheme from the provided examples, infer and fully instantiate all variables and predicates to reconstruct the complete argument in computational form.

Now, you must analyze the output of this task, evaluating two specific points: the semantics of the reconstruction; and the proper reconstruction of the missing components.
You must give, for each criteria, one score ranging from 1 to 5 and an explanation, where each score to each criteria corresponds to one specific justification for the enthymeme reconstruction task, comparing the model's output with the expected answer modelled:

Semantics: [Score 1: Variables are incorrectly instantiated, violating intended meaning or type.
Score 2: Several variables have unclear or loosely related semantics.
Score 3: Most variables are semantically correct, with some ambiguity.
Score 4: Variables are mostly well-defined with minor semantic issues.
Score 5: All variables are precisely and unambiguously instantiated.]

Correctness/Completeness: [Score 1: Components are mostly incorrect or missing, with major inconsistencies.
Score 2: Several components are inaccurate or incomplete.
Score 3: Key elements are captured, but some are missing or distorted.
Score 4: Components are mostly accurate with only minor deviations.
Score 5: Components are reconstructed with high precision and fidelity.]



Your response must follow rigorously the specific format and not add anything else:
"Semantics: <score>. Justification: <the justification for the score on this specific test>
Correctness/Completeness: <score>. Justification: <the justification for the score on this specific test>".

Natural language enthymeme: [{enthymeme}]

Model's output: [{output}]

Expected output: [{expected}]


[/INST]
"""



i = 0
for test in tests_list:

    print(f'\033[92mTest: {test}\n\n\033[0m')

    evaluation1 = filtering_output(client_ds.chat.completions.create(model=models_list[0],
                                                max_tokens=8000,
                                                messages=[{"role": "user", "content": evaluation_prompt.format(enthymeme=nl_tests_list[i],
                                                                                                               output=test,
                                                                                                               expected=answers_list[i])}],
                                                temperature=0.0).choices[0].message.content.strip())

    print('Evaluation 1: ' + evaluation1 + '\n\n')

    evaluation2 = filtering_output(client_ds.chat.completions.create(model=models_list[1],
                                                max_tokens=8000,
                                                messages=[{"role": "user", "content": evaluation_prompt.format(enthymeme=nl_tests_list[i],
                                                                                                               output=test,
                                                                                                               expected=answers_list[i])}],
                                                temperature=0.0).choices[0].message.content.strip())

    print('Evaluation 2: ' + evaluation2 + '\n\n')

    evaluation3 = filtering_output(client_gq.chat.completions.create(model=models_list[2],
                                                max_tokens=8000,
                                                messages=[{"role": "user", "content": evaluation_prompt.format(enthymeme=nl_tests_list[i],
                                                                                                               output=test,
                                                                                                               expected=answers_list[i])}],
                                                temperature=0.0).choices[0].message.content.strip())
    
    print('Evaluation 3: ' + evaluation3 + '\n\n')

    evaluation1 = evaluation1.replace("Correctness/Completeness", "\tCorrectness/Completeness").replace('\n', '').replace('\r', '')
    evaluation2 = evaluation2.replace("Correctness/Completeness", "\tCorrectness/Completeness").replace('\n', '').replace('\r', '')
    evaluation3 = evaluation3.replace("Correctness/Completeness", "\tCorrectness/Completeness").replace('\n', '').replace('\r', '')


    
    with open(f'evaluations/{models_list[0]}/{test_model}/evaluations_class_{class_}.txt', 'a', encoding='utf-8', errors='replace') as t:
        t.write(evaluation1 + '\n')
    
    with open(f'evaluations/{models_list[1]}/{test_model}/evaluations_class_{class_}.txt', 'a', encoding='utf-8', errors='replace') as t:
        t.write(evaluation2 + '\n')

    with open(f'evaluations/{tests_models[2]}/{test_model}/evaluations_class_{class_}.txt', 'a', encoding='utf-8', errors='replace') as t:
        t.write(evaluation3 + '\n')

    i += 1
    

