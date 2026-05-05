from langchain_core.prompts import PromptTemplate

# Required inputs: input_variables, template
## Method 1
prompt_template1 = PromptTemplate(
    template = "You are {role}, and your name is {name}",
    input_variables=["role", "name"]
)

## Method 2
prompt_template2 = PromptTemplate.from_template(
    template = "You are {role}, and your name is {name}"
)

prompt = prompt_template1.format(role="an AI expert", name="xXx")

## Method 3
prompt_template3 = PromptTemplate(
    template = "You are {role}, and your name is {name}",
    input_variables=["role", "name"],
    partial_variables={"role": "an AI expert"}
)
prompt = prompt_template3.format(name="xXx")

## Method 4
prompt_template4 = PromptTemplate(
    template = "You are {role}, and your name is {name}",
    input_variables=["role", "name"]
)
# partial() doesn't modify the original template, and creates a new template instead.
template = prompt_template4.partial(role="an AI expert")
prompt = template.format(name="xXx")

prompt_template5 = PromptTemplate(
    template = "You are {role}, and your name is {name}",
    input_variables=["role", "name"]
).partial(role="an AI expert")
prompt = prompt_template5.format(name="xXx")

print(prompt)

# ---------------------------------------- #
# Combined prompt templates
# ---------------------------------------- #
template1 = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + 
    PromptTemplate.from_template(" in {language}")
)

prompt = template1.format(topic="sports", language="Chinese")
print(prompt)

# --------------------------------------------------------- #
# Assign values to variables using format() and invoke()
#
# format(): input is variable, output is str
# invoke(): input is a dict, output is langchain_core.prompt_values.StringPromptValue
# --------------------------------------------------------- #
template1 = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + 
    PromptTemplate.from_template(" in {language}")
)

prompt = template1.invoke(input={"topic": "sports", "language": "Chinese"})
print(prompt)

# --------------------------------------------------------- #
# --------------------------------------------------------- #

# --------------------------------------------------------- #
# ChatPromptTemplate
# invoke() \ format() \ format_messages() \ format_prompt()
# --------------------------------------------------------- #
from langchain_core.prompts import ChatPromptTemplate

chat_prompt_tempalte = ChatPromptTemplate(
    messages=[
        ("system", "You are an AI assistant, your name is {name}"),
        ("human", "My question is {question}")
    ],
    input_variables = ["name", "question"] # this is optional
)

prompt = chat_prompt_tempalte.invoke(
    input={"name": "xXx", "question": "1+2*3=?"}
)
print(prompt)
# messages=[SystemMessage(content='You are an AI assistant, your name is xXx', additional_kwargs={}, response_metadata={}), HumanMessage(content='My question is 1+2*3=?', additional_kwargs={}, response_metadata={})]

chat_prompt_tempalte = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are an AI assistant, your name is {name}"),
        ("human", "My question is {question}")
    ]
)

prompt = chat_prompt_tempalte.invoke(
    input={"name": "xXx", "question": "1+2*3=?"}
)
print(prompt)
# messages=[SystemMessage(content='You are an AI assistant, your name is xXx', additional_kwargs={}, response_metadata={}), HumanMessage(content='My question is 1+2*3=?', additional_kwargs={}, response_metadata={})]
