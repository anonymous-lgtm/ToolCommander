import json
import os
import re

# ToolLLaMa System Prompt Template
FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:\n{TOOLS_LIST}"""

# ToolLLaMa Finish Function Template
FINISH_FUNC = {
    "name": "Finish",
    "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
    "parameters": {
        "type": "object",
        "properties": {
            "return_type": {
                "type": "string",
                "enum": ["give_answer", "give_up_and_restart"],
            },
            "final_answer": {
                "type": "string",
                "description": 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"',
            },
        },
        "required": ["return_type"],
    },
}


pool = {}


def load_tool_json(tool_root_dir, category, tool_name):
    category = standardize_category(category)
    tool_name = standardize(tool_name)
    key = os.path.join(tool_root_dir, category, tool_name + ".json")
    if key in pool:
        return pool[key]
    else:
        value = json.load(
            open(os.path.join(tool_root_dir, category, tool_name + ".json"), "r")
        )
        pool[key] = value
    return value


def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name


def convert_tool_json_to_corpus(doc):
    return (
        (doc.get("category_name", "") or "")
        + ", "
        + (doc.get("tool_name", "") or "")
        + ", "
        + (doc.get("api_name", "") or "")
        + ", "
        + (doc.get("api_description", "") or "")
        + ", required_params: "
        + json.dumps(doc.get("required_parameters", ""))
        + ", optional_params: "
        + json.dumps(doc.get("optional_parameters", ""))
        + ", return_schema: "
        + json.dumps(doc.get("template_response", ""))
    )


def process_retrieval_document(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        corpus_str = convert_tool_json_to_corpus(doc)
        ir_corpus[row.docid] = corpus_str
        corpus2tool[corpus_str] = (
            doc["category_name"],
            doc["tool_name"],
            doc["api_name"],
        )
    return ir_corpus, corpus2tool


def build_tool_description(data_dict):
    origin_tool_names = [
        standardize(cont["tool_name"]) for cont in data_dict["api_list"]
    ]
    tool_descriptions = [
        [cont["standard_tool_name"], cont["description"]] for cont in origin_tool_names
    ]
    return tool_descriptions


def api_json_to_openai_json(api_json, standard_tool_name):
    description_max_length = 256
    templete = {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "optional": [],
        },
    }

    map_type = {"NUMBER": "integer", "STRING": "string", "BOOLEAN": "boolean"}

    pure_api_name = change_name(standardize(api_json["api_name"]))
    templete["name"] = pure_api_name + f"_for_{standard_tool_name}"
    templete["name"] = templete["name"][-64:]

    templete["description"] = (
        f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
    )

    if api_json["api_description"].strip() != "":
        tuncated_description = api_json["api_description"].replace(
            api_json["api_name"], templete["name"]
        )[:description_max_length]
        templete["description"] = (
            templete["description"]
            + f'The description of this function is: "{tuncated_description}"'
        )
    if (
        "required_parameters" in api_json.keys()
        and len(api_json["required_parameters"]) > 0
    ):
        for para in api_json["required_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"
            prompt = {
                "type": param_type,
                "description": para["description"][:description_max_length],
            }

            if "default" in para.keys():
                default_value = para["default"]
            else:
                default_value = ""
            if len(str(default_value)) != 0:
                prompt = {
                    "type": param_type,
                    "description": para["description"][:description_max_length],
                    "example_value": default_value,
                }
            else:
                prompt = {
                    "type": param_type,
                    "description": para["description"][:description_max_length],
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["required"].append(name)
    if (
        "optional_parameters" in api_json.keys()
        and len(api_json["optional_parameters"]) > 0
    ):
        for para in api_json["optional_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"

            default_value = para["default"]
            if len(str(default_value)) != 0:
                prompt = {
                    "type": param_type,
                    "description": para["description"][:description_max_length],
                    "example_value": default_value,
                }
            else:
                prompt = {
                    "type": param_type,
                    "description": para["description"][:description_max_length],
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["optional"].append(name)

    return templete, api_json["category_name"], pure_api_name
