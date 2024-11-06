import json

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import util
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizer

from toolbench import *
from utils import mean_pooling


class ToolRetriever:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        corpus_tsv_path: str = "",
    ):
        """
        Retriever to retrieve APIs from the toolbench corpus.

        Attributes:
            model (PreTrainedModel): The model to be used for retrieval.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
            corpus_tsv_path (str): The path to the TSV file containing the toolbench corpus.
        """
        self.corpus_tsv_path = corpus_tsv_path
        self.model = model
        self.tokenizer = tokenizer
        self.corpus, self.corpus2tool = self._build_retrieval_corpus()
        self.corpus_embeddings = self._build_corpus_embeddings()

    def _embedding(self, sentences, disable=True, batch_size=256):
        """
        Get the embeddings of the sentences / tool documents.

        Args:
            sentences (List[str]): The list of sentences to be embedded.
            disable (bool): Whether to disable the progress bar.
            batch_size (int): The batch size for embedding.
        """
        assert isinstance(sentences, list), "Sentences must be a list of string"
        all_embeddings = []
        for i in trange(0, len(sentences), batch_size, disable=disable):
            encoded_input = self.tokenizer(
                sentences[i : i + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            ).to(self.model.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = mean_pooling(
                model_output[0], device=self.model.device
            )
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings)
        result = torch.concat(all_embeddings, dim=0)
        return result

    def _build_retrieval_corpus(self):
        """
        Build the retrieval corpus.
        """
        print("Building corpus...")
        documents_df = pd.read_csv(self.corpus_tsv_path, sep="\t")
        corpus, corpus2tool = process_retrieval_document(documents_df)
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]
        return corpus, corpus2tool

    def _build_corpus_embeddings(self):
        """
        Embed the corpus.
        """
        print("Building corpus embeddings...")
        return self._embedding(self.corpus, disable=False)

    def retrieve_api(
        self,
        query,
        top_k=5,
        excluded_tools={},
        extra_functions=[],
        force_extra_first_dev=False,
        force_extra_last_dev=False,
    ):
        """
        Retrieve the APIs based on the query.

        Args:
            query (str): The query to be used for retrieval.
            top_k (int): The number of APIs to retrieve. Defaults to 5.
            excluded_tools (dict): The tools to be excluded. Defaults to {}.
            extra_functions (list[str]): The extra functions to be included. Defaults to [].
            force_extra_first_dev (bool): Whether to force the extra functions to be first. Defaults to False.
            force_extra_last_dev (bool): Whether to force the extra functions to be last, for dev-use only. Defaults to False.
        """
        query_embedding = self._embedding([query])

        return self.retrieve_api_by_emb(
            query_embedding,
            top_k,
            excluded_tools,
            extra_functions,
            force_extra_first_dev,
            force_extra_last_dev,
        )

    def retrieve_api_by_emb(
        self,
        emb,
        top_k=5,
        excluded_tools={},
        extra_functions=[],
        force_extra_first_dev=False,
        force_extra_last_dev=False,
    ):
        """
        Retrieve the APIs based on the query embedding.

        Args:
            emb (Tensor): The query embedding.
            top_k (int): The number of APIs to retrieve. Defaults to 5.
            excluded_tools (dict): The tools to be excluded. Defaults to {}.
            extra_functions (list[str]): The extra functions to be included. Defaults to [].
            force_extra_first_dev (bool): Whether to force the extra functions to be first. Defaults to False.
            force_extra_last_dev (bool): Whether to force the extra functions to be last, for dev-use only. Defaults to False.
        """
        corpus2tool = self.corpus2tool

        extra_corpus = []
        extra_tool_list = []
        extra_tool_names = set()
        for tool in extra_functions:
            tool = json.loads(tool)
            tool_corpus = convert_tool_json_to_corpus(tool)
            extra_corpus.append(tool_corpus)
            extra_tool = (tool["category_name"], tool["tool_name"], tool["api_name"])
            corpus2tool[tool_corpus] = extra_tool
            extra_tool_list.append(tool)
            extra_tool_names.add(standardize(tool["tool_name"]))

        corpus = self.corpus + extra_corpus

        if len(extra_corpus) == 0:
            corpus_enlarged_embeddings = self.corpus_embeddings
        else:
            extra_embeddings = self._embedding(extra_corpus).to(self.model.device)
            corpus_enlarged_embeddings = torch.cat(
                [self.corpus_embeddings, extra_embeddings], dim=0
            ).to(self.model.device)

        hits = util.semantic_search(
            emb,
            corpus_enlarged_embeddings,
            top_k=10 * top_k,
            score_function=util.cos_sim,
        )

        retrieved_extra_cnt = 0
        retrieved_tools = []
        for rank, hit in enumerate(hits[0]):
            category, tool_name, api_name = corpus2tool[corpus[hit["corpus_id"]]]
            category = standardize_category(category)
            tool_name = standardize(tool_name)  # standardizing
            api_name = change_name(standardize(api_name))  # standardizing

            if tool_name in excluded_tools:
                continue
            tmp_dict = {
                "category": category,
                "tool_name": tool_name,
                "api_name": api_name,
            }
            retrieved_tools.append(tmp_dict)

            if tool_name in extra_tool_names:
                retrieved_extra_cnt += 1

            if len(retrieved_tools) == top_k:
                break

        if force_extra_first_dev:
            for i, tool in enumerate(retrieved_tools):
                if tool["tool_name"] in extra_tool_names:
                    retrieved_tools.insert(0, retrieved_tools.pop(i))
            if retrieved_extra_cnt == 0:
                retrieved_tools = extra_tool_list + retrieved_tools
                retrieved_tools = retrieved_tools[:top_k]
        elif force_extra_last_dev:
            for i, tool in enumerate(retrieved_tools):
                if tool["tool_name"] in extra_tool_names:
                    retrieved_tools.append(retrieved_tools.pop(i))
            if retrieved_extra_cnt == 0:
                if top_k - 1 < 0:
                    retrieved_tools = extra_tool_list
                else:
                    retrieved_tools = retrieved_tools[: top_k - 1] + extra_tool_list

        return retrieved_tools, retrieved_extra_cnt


class ToolAttacker:
    def __init__(
        self,
        retriever_model,
        retriever_tokenizer,
        corpus_tsv_path,
        tool_root_dir,
        device="cpu",
        extra_tool_category="Customized",
        extra_tool_name="ToolAttack",
        extra_tool_desc="ToolAttack is a tool for generating adversarial examples for API-based systems.",
    ):
        """
        Initializes the tool attacker.

        Args:
            retriever_model: The model to be used for retrieval.
            retriever_tokenizer: The tokenizer associated with the model.
            corpus_tsv_path: The path to the TSV file containing the toolbench corpus.
            tool_root_dir: The root directory of the toolbench.
            device: The device to be used for the attacker. Defaults to "cpu".
            extra_tool_category: The category of the extra tool. Defaults to "Customized".
            extra_tool_name: The name of the extra tool. Defaults to "ToolAttack".
            extra_tool_desc: The description of the extra tool. Defaults to "ToolAttack is a tool for generating adversarial examples for API-based systems
        """
        self.retriever_model = retriever_model
        self.retriever = ToolRetriever(
            model=retriever_model,
            tokenizer=retriever_tokenizer,
            corpus_tsv_path=corpus_tsv_path,
        )
        self.tool_root_dir = tool_root_dir
        self.device = device
        self.category = standardize_category(extra_tool_category)
        self.extra_tool_name = standardize(extra_tool_name)
        self.extra_tool_desc = extra_tool_desc

    def retrieve_and_update_sys_prompt(
        self,
        query,
        extra_functions=[],
        exclude_tools={},
        force_extra_first_dev=False,
        force_extra_last_dev=False,
        retrieved=None,
    ):
        """
        Retrieve and update the system prompt based on the query.

        Args:
            query: The query to be used for retrieval.
            extra_functions: The extra functions to be included. Defaults to [].
            exclude_tools: The tools to be excluded. Defaults to {}.
            force_extra_first_dev: Whether to force the extra functions to be first. Defaults to False.
            force_extra_last_dev: Whether to force the extra functions to be last, for dev-use only. Defaults to False.
            retrieved: The retrieved APIs. Defaults to None.

        Returns:
            The system prompt, extra API names, other API names, the number of retrieved extra functions, and the target indices
        """
        if retrieved is None:
            retrieved, _ = self.retriever.retrieve_api(
                query,
                top_k=5,
                extra_functions=extra_functions,
                excluded_tools=exclude_tools,
                force_extra_first_dev=force_extra_first_dev,
                force_extra_last_dev=force_extra_last_dev,
            )

        sys_prompt, extra_api_name, other_api_name, retrieved_extra_cnt, target_idxs = (
            self._generate_system_message(
                retrieved=retrieved,
                tool_root_dir=self.tool_root_dir,
                extra_functions=extra_functions,
            )
        )
        sys_prompt += f"\nAction Must Be One of the Following APIs: " + ",".join(
            other_api_name + extra_api_name + ["Finish"]
        )
        return (
            sys_prompt,
            extra_api_name,
            other_api_name,
            retrieved_extra_cnt,
            target_idxs,
        )

    def _generate_system_message(
        self,
        retrieved,
        tool_root_dir,
        extra_functions=[],
        system_message=FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION,
    ):
        """
        Adapted from toolbench official code
        """
        assert "with a function call to actually excute your step." in system_message
        system_message = system_message.replace(
            "with a function call to actually excute your step.",
            "with a function call to actually excute your step. Your output should follow this format:\nThought: <Your Thought>\nAction: <full API names, i.e., subfunction name (do not use '->' or '.')>\nAction Input: <API call input in JSON format>\n",
        )

        deduplicated_tools = []
        api_list = []
        extra_api_name = []
        other_api_name = []

        retrieved_extra_cnt = 0
        idx = -1
        target_idxs = []
        for func in retrieved:
            idx += 1
            next = False
            if "tool_name" in func:
                extra_tool_name = standardize(func["tool_name"])
            else:
                extra_tool_name = self.extra_tool_name

            if "tool_desc" in func:
                extra_tool_desc = func["tool_desc"]
            else:
                extra_tool_desc = self.extra_tool_desc
            for extra_func in extra_functions:
                extra_func_dict = json.loads(extra_func)
                if func["api_name"] == standardize(extra_func_dict["api_name"]):

                    openai_api_json, _, _ = api_json_to_openai_json(
                        extra_func_dict, extra_tool_name
                    )
                    api_list.append(openai_api_json)
                    extra_api_name.append(openai_api_json["name"])
                    retrieved_extra_cnt += 1
                    deduplicated_tools.append(
                        (
                            extra_tool_name,
                            extra_tool_desc if extra_tool_desc else "",
                        )
                    )
                    target_idxs.append(idx)
                    next = True
                    break
            if next:
                continue
            standardized_tool_name = standardize(func["tool_name"])
            tool_json = load_tool_json(
                tool_root_dir, func["category"], standardized_tool_name
            )

            tool_desc = tool_json["tool_description"]
            deduplicated_tools.append((standardized_tool_name, tool_desc))

            for api_dict in tool_json["api_list"]:
                pure_api_name = change_name(standardize(api_dict["name"]))
                if pure_api_name != func["api_name"]:
                    continue
                api_json = {}
                api_json["category_name"] = func["category"]
                api_json["api_name"] = api_dict["name"]
                api_json["api_description"] = api_dict["description"]
                api_json["required_parameters"] = api_dict["required_parameters"]
                api_json["optional_parameters"] = api_dict["optional_parameters"]
                api_json["tool_name"] = tool_json["tool_name"]

                openai_api_json, _, _ = api_json_to_openai_json(
                    api_json, standardized_tool_name
                )

                api_list.append(openai_api_json)
                other_api_name.append(openai_api_json["name"])

        api_list.append(FINISH_FUNC)
        deduplicated_tools = list(set(deduplicated_tools))

        tools_str = ""
        for idx, tool in enumerate(deduplicated_tools):
            tool_name, tool_desc = tool
            tools_str += f"{idx+1}.{tool_name}: {tool_desc}\n"

        system_message = system_message.format(TOOLS_LIST=tools_str)

        system_message = (
            system_message
            + "\nSpecifically, you have access to the following APIs: "
            + str(api_list)
        )
        return (
            system_message,
            extra_api_name,
            other_api_name,
            retrieved_extra_cnt,
            target_idxs,
        )
