import json
import re
import sys

from typing import List, Dict
from chainlite import chain, get_logger, llm_generation_chain, pprint_chain
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnablePassthrough


from spinach_agent.parser_state import (
    Action,
    PartToWholeParserState,
    SparqlQuery,
    state_to_dict,
    state_to_string,
)
from spinach_agent.parser_utils import (
    BaseParser,
    execute_sparql_object,
    get_prune_edges_chain,
    parse_string_to_json,
    sparql_string_to_sparql_object,
)
from wikidata_utils import (
    SparqlExecutionStatus,
    get_outgoing_edges,
    search_span,
)
from wikidata_utils import get_property_examples

logger = get_logger(__name__)

wikidata_data_type_to_description_map = {
    # "string": "",
    "external-id": "Represents an identifier used in an external system.",
    "url": "Represents a link to an external resource or site. Can be converted to string using STR()",
    "commonsMedia": "References to files (like images, audio) on Wikimedia Commons.",
    "geo-shape": "Reference to map data files on Wikimedia Commons.",
    "tabular-data": "Reference to tabular data files on Wikimedia Commons.",
    "math": "Stores and displays formatted mathematical formulas.",
    "musical-notation": "Stores and displays musical scores as generated images in .png format.",
}


def format_wikidata_search_result(arr):
    ret = []
    for item in arr:
        label = item.get(
            "label", ""
        )  # Get the label or an empty string if not available
        id = item.get("id", "")  # Get the id or an empty string if not available
        description = item.get(
            "description", ""
        )  # Get the description or an empty string if not available

        display_string = f"{label} ({id}): {description}"
        if "datatype" in item:
            data_type = item["datatype"]
            if data_type in wikidata_data_type_to_description_map:
                data_type += f" ({wikidata_data_type_to_description_map[data_type]})"
            display_string += f"\tData Type: {data_type}"

        ret.append(display_string)
    return "\n".join(ret)


@chain
async def json_to_string(j: dict) -> str:
    return json.dumps(j, indent=2, ensure_ascii=False)


@chain
async def json_to_action(action_dict: dict) -> Action:
    thought = action_dict["thought"]
    action_name = action_dict["action_name"]
    action_argument = action_dict["action_argument"]

    if action_name == "execute_sparql":
        assert action_argument, action_dict

    return Action(
        thought=thought,
        action_name=action_name,
        action_argument=action_argument,
    )


# async def sparql_debugger_router(state):
#     # last action was an execute_sparql()
#     current_action = PartToWholeParser.get_current_action(state)
#     assert current_action.action_name == "execute_sparql"
#     sparql_to_debug = state["generated_sparqls"][-1]
#     sparqls_to_help_debug = []
#     # TODO feed in previous observations of PIDs and QIDs and entity pages as well
#     if not sparql_to_debug.has_results():
#         print("sparql_to_debug = ", sparql_to_debug)
#         for _ in range(15):
#             o = await PartToWholeParser.debug_sparql_chain.ainvoke(
#                 {
#                     "sparql_to_debug": sparql_to_debug,
#                     "sparqls_to_help_debug": sparqls_to_help_debug,
#                 }
#             )
#             print("sparql to help debug =  ", o)
#             print("execution result:", o.execution_result)
#             sys.stdout.flush()
#             sparqls_to_help_debug.append(o)
#         exit()
#     return "controller"

def display_actions(actions):
    # truncate actions
    actions = actions[-10:]
    
    action_history = []
    for i, a in enumerate(actions):
        include_observation = True
        if i < len(actions) - 2 and a.action_name in [
            "get_wikidata_entry",
            "search_wikidata",
        ]:
            include_observation = False
        if a.action_name == "stop":
            include_observation = False
        action_history.append(a.to_jinja_string(include_observation))
    return action_history


class PartToWholeParser(BaseParser):
    engine = None  # Store engine as class variable for use in static methods
    
    @classmethod
    def initialize(
        cls,
        engine: str,
    ):
        cls.engine = engine  # Store engine for recursive query splitting
        
        @chain
        async def initialize_state(input):
            return PartToWholeParserState(
                question=input["question"],
                conversation_history=input["conversation_history"],
                engine=engine,
                action_counter=0,
                actions=[],
                response=""
                # generated_sparqls=[],
            )

        # build the graph
        graph = StateGraph(PartToWholeParserState)
        graph.add_node("start", lambda x: {})
        graph.add_node("controller", PartToWholeParser.controller)
        graph.add_node("execute_sparql", PartToWholeParser.execute_sparql)
        graph.add_node("get_wikidata_entry", PartToWholeParser.get_wikidata_entry)
        graph.add_node(
            "search_wikidata",
            PartToWholeParser.search_wikidata,
        )
        graph.add_node("get_property_examples", PartToWholeParser.get_property_examples)
        graph.add_node("reporter", PartToWholeParser.reporter)
        graph.add_node("stop", PartToWholeParser.stop)

        graph.set_entry_point("start")

        graph.add_edge("start", "controller")
        graph.add_conditional_edges(
            "controller",
            PartToWholeParser.router,  # the function that will determine which node is called next.
        )
        for n in [
            "execute_sparql",
            "get_wikidata_entry",
            "search_wikidata",
            "get_property_examples",
        ]:
            graph.add_edge(n, "controller")
        graph.add_edge("stop", "reporter")
        graph.add_edge("reporter", END)

        # graph.add_edge("execute_sparql", sparql_debugger_router)

        cls.controller_chain = (
            {
                "input": llm_generation_chain(
                    template_file="controller.prompt",
                    engine=engine,
                    max_tokens=700,
                    temperature=1.0,
                    top_p=0.9,
                    # stop_tokens=["Observation:"],
                    keep_indentation=True,
                )
            }
            | llm_generation_chain(
                template_file="format_actions.prompt",
                engine=engine,
                max_tokens=700,
                keep_indentation=True,
                output_json=True
            )
            | parse_string_to_json
            | json_to_action
        )

        cls.sparql_chain = sparql_string_to_sparql_object | execute_sparql_object
        cls.prune_edges_chain = get_prune_edges_chain(engine=engine) | json_to_string
        cls.reporter_chain = llm_generation_chain(
            template_file="conversation_reporter.prompt",
            engine=engine,
            max_tokens=2000,
            temperature=0.0,
            keep_indentation=True,
        )
        cls.language_detection_chain = llm_generation_chain(
                    template_file="language_detection.prompt",
                    engine=engine,
                    max_tokens=50,
                    temperature=0.0,
                    top_p=0.9,
                    keep_indentation=True,
                )

        cls.is_query_simple_chain = llm_generation_chain(
            template_file="is_query_simple.prompt",
            engine=engine,
            max_tokens=50,
            temperature=0.0,
            keep_indentation=True,
        )
        
        cls.decompose_query_chain = (
            llm_generation_chain(
                template_file="decompose_query.prompt",
                engine=engine,
                max_tokens=1000,
                temperature=0.0,
                keep_indentation=True,
                output_json=True
            )
            | parse_string_to_json
        )

        # cls.debug_sparql_chain = (
        #     llm_generation_chain(
        #         template_file="sparql_debugger.prompt", engine=engine, max_tokens=1000
        #     )
        #     | extract_code_block_from_output.bind(code_block="sparql")
        #     | cls.sparql_chain
        # )

        compiled_graph = graph.compile()
        cls.runnable = initialize_state | compiled_graph
        logger.info("Finished initializing the graph.")
        # compiled_graph.get_graph().print_ascii()
        # sys.stdout.flush()

    @staticmethod
    def get_current_action(state):
        return state["actions"][-1]
    
    @staticmethod
    async def is_query_simple(sparql: str, question: str) -> bool:
        """
        Determine if a SPARQL query is simple enough to execute directly.
        
        Args:
            sparql: SPARQL query string
            question: Original question
            
        Returns:
            True if query is simple, False if it needs decomposition
        """
        if not PartToWholeParser.engine:
            # If engine not initialized, default to simple (no recursive splitting)
            return True
            
        try:
            response = await PartToWholeParser.is_query_simple_chain.ainvoke({
                "sparql": sparql,
                "question": question
            })
            return response.strip().upper() == 'SIMPLE'
        except Exception as e:
            logger.warning(f"Error checking query simplicity: {e}. Defaulting to simple.")
            return True
    
    @staticmethod
    async def decompose_query(sparql: str, question: str) -> dict:
        """
        Decompose a complex SPARQL query into simpler sub-queries.
        
        Args:
            sparql: Complex SPARQL query
            question: Original question
            
        Returns:
            Dictionary with 'sub_query', 'final_query', and 'explanation'
        """
        if not PartToWholeParser.engine:
            raise ValueError("Parser not initialized. Call initialize() first.")
            
        try:
            decomposition = await PartToWholeParser.decompose_query_chain.ainvoke({
                "sparql": sparql,
                "question": question
            })
            return decomposition
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            raise ValueError(f"Failed to decompose query: {e}")
    
    @staticmethod
    async def solve_query_recursively(sparql: str, question: str, depth: int = 0, max_depth: int = 5) -> SparqlQuery:
        """
        Recursively solve a SPARQL query by splitting it into simpler sub-queries if needed.
        
        Args:
            sparql: SPARQL query to solve
            question: Original question
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            SparqlQuery object with execution results
        """
        # TODO: This is optional, we can remove it if we want or tinker with max depth.
        if depth >= max_depth:
            logger.warning(f"Max recursion depth ({max_depth}) reached. Executing query directly.")
            sparql_obj = PartToWholeParser.sparql_chain.invoke(sparql)
            return sparql_obj
        
        is_simple = await PartToWholeParser.is_query_simple(sparql, question)
        
        if is_simple:
            # Execute the query directly
            logger.info(f"{'  ' * depth}Query is simple (depth {depth}), executing directly...")
            sparql_obj = PartToWholeParser.sparql_chain.invoke(sparql)
            return sparql_obj
        else:
            # Decompose the query
            logger.info(f"{'  ' * depth}Query is complex (depth {depth}), decomposing...")
            decomposition = await PartToWholeParser.decompose_query(sparql, question)
            
            sub_query = decomposition.get('sub_query', '')
            final_query = decomposition.get('final_query', '')
            
            if not sub_query or not final_query:
                logger.warning("Decomposition failed to produce valid queries. Executing original query directly.")
                sparql_obj = PartToWholeParser.sparql_chain.invoke(sparql)
                return sparql_obj
            
            # Execute the sub-query
            logger.info(f"{'  ' * depth}Executing sub-query...")
            sub_sparql_obj = await PartToWholeParser.solve_query_recursively(
                sub_query, question, depth + 1, max_depth
            )
            
            if not sub_sparql_obj.has_results():
                logger.warning(f"{'  ' * depth}Sub-query returned no results. Returning empty result.")
                return sub_sparql_obj
            
            # Incorporate intermediate results into final query using VALUES clause
            # Extract the results from sub-query
            intermediate_results = sub_sparql_obj.execution_result
            
            # Build VALUES clause from intermediate results
            if isinstance(intermediate_results, list) and len(intermediate_results) > 0:
                # Get the first variable from the results
                first_result = intermediate_results[0]
                if isinstance(first_result, dict):
                    # Find the main variable (usually the first one that's not a label)
                    main_var = None
                    for key in first_result.keys():
                        if not key.endswith('Label') and not key.endswith('label'):
                            main_var = key
                            break
                    
                    if main_var:
                        # Build VALUES clause
                        values_clause = "VALUES ?intermediate_result {\n"
                        # TODO: Is this limit causing decreases in classification accuracy?
                        for result in intermediate_results[:100]:  # Limit to 100 values
                            value = result.get(main_var, {}).get('value', '')
                            if value:
                                # Extract QID or value from URI if needed
                                if 'Q' in value or 'P' in value:
                                    # Extract QID/PID
                                    match = re.search(r'(Q\d+|P\d+)', value)
                                    if match:
                                        values_clause += f"  wd:{match.group(1)}\n"
                                else:
                                    values_clause += f'  "{value}"\n'
                        values_clause += "}"
                        
                        # Insert VALUES clause into final query
                        # Try to insert after WHERE or before the main WHERE block
                        if "WHERE" in final_query.upper():
                            # Insert VALUES before WHERE
                            where_pos = final_query.upper().find("WHERE")
                            final_query_with_values = (
                                final_query[:where_pos] + 
                                values_clause + "\n" + 
                                final_query[where_pos:]
                            )
                        else:
                            # Append VALUES at the end before any closing brace
                            final_query_with_values = final_query + "\n" + values_clause
                        
                        final_query = final_query_with_values
                    else:
                        logger.warning("Could not find main variable in intermediate results. Using original final query.")
                else:
                    logger.warning("Intermediate results format not recognized. Using original final query.")
            else:
                logger.warning("No intermediate results to incorporate. Using original final query.")
            
            # Recursively solve the final query
            logger.info(f"{'  ' * depth}Executing final query...")
            final_sparql_obj = await PartToWholeParser.solve_query_recursively(
                final_query, question, depth + 1, max_depth
            )
            
            return final_sparql_obj

    @staticmethod
    async def router(state):
        move_back_on_duplicate_action = 2
        current_action = PartToWholeParser.get_current_action(state)
        if current_action in state["actions"][-5:-1]:
            logger.warning(
                "Took duplicate action %s, going back %d steps.",
                current_action.action_name,
                move_back_on_duplicate_action,
            )
            # current_action.observation = "I have already taken this action. I should not repeat the same action twice."
            # Remove generated_sparqls as well
            if len(state["actions"]) - 2 >= 0:
                for i in range(len(state["actions"]) - 2, -1, -1):
                    if state["actions"][i].action_name == "execute_sparql":
                        state["generated_sparqls"] = state["generated_sparqls"][:-1]
            state["actions"] = state["actions"][:-move_back_on_duplicate_action]
            state["action_counter"] -= move_back_on_duplicate_action
            return "controller"

        if state["action_counter"] >= 15:
            return "reporter"
        
        return state["actions"][-1].action_name

    @staticmethod
    @chain
    async def controller(state):
        # if state["actions"]:
            # print("last action = ", state["actions"][-1])
            # sys.stdout.flush()

        # make the history shorter
        actions = state["actions"]
        action_history = display_actions(actions)

        # try two times if there is an assertion error,
        # happens if an action not specified is chosen
        # if fails after 2 times, end the reasoning.
        # attempts = 2
        # for attempt in range(attempts):
        #     try:
        #         action = await PartToWholeParser.controller_chain.ainvoke(
        #             {"question": state["question"], "action_history": action_history}
        #         )
        #         return {"actions": action, "action_counter": 1}
        #     except AssertionError:
        #         if attempt == attempts - 1:
        #             return {"actions": Action(thought="", action_name="stop", action_argument=""), "action_counter": 1}
        action = await PartToWholeParser.controller_chain.ainvoke(
            {
                "conversation_history": state["conversation_history"],
                "question": state["question"],
                "action_history": action_history
            }
        )
        # print("controller output = ", action)
        return {"actions": action, "action_counter": 1}

    @staticmethod
    @chain
    async def execute_sparql(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "execute_sparql"
        # print("executing ", current_action.action_argument)
        
        # Use recursive query splitting to solve the query
        try:
            sparql = await PartToWholeParser.solve_query_recursively(
                current_action.action_argument,
                state["question"],
                depth=0,
                max_depth=5
            )
        except Exception as e:
            logger.warning(f"Recursive query splitting failed: {e}. Falling back to direct execution.")
            # Fallback to direct execution
            sparql = PartToWholeParser.sparql_chain.invoke(
                current_action.action_argument
            )
        
        current_action.action_argument = (
            sparql.sparql
        )  # update it with the cleaned and optimized SPARQL
        if sparql.has_results():
            current_action.observation = sparql.results_in_table_format()
            current_action.observation_markdown = sparql.results_in_markdown_format()
            # print("SPARQL execution result:", current_action.observation)
        else:
            if sparql.execution_status == SparqlExecutionStatus.SYNTAX_ERROR:
                msg = sparql.execution_status.get_message()
                current_action.observation = "Query had syntax error." if not msg else msg
            elif sparql.execution_status == SparqlExecutionStatus.TIMED_OUT:
                current_action.observation = "Query execution timed out."
            elif sparql.execution_status == SparqlExecutionStatus.OTHER_ERROR:
                current_action.observation = "Query execution ran into an error."
            else:
                current_action.observation = "Query returned empty result."

        return {"generated_sparqls": sparql}

    @staticmethod
    @chain
    async def get_wikidata_entry(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "get_wikidata_entry"
        wikidata_entry = get_outgoing_edges(
            current_action.action_argument, compact=True
        )
        # NOTE: The only seen exception from this so far
        # is due to contex too long
        try:
            action_result = await PartToWholeParser.prune_edges_chain.ainvoke(
                {
                    "question": state["question"],
                    "conversation_history": state["conversation_history"],
                    "outgoing_edges": json.dumps(
                        wikidata_entry, indent=2, ensure_ascii=False
                    ),
                    "entity_and_description": current_action.action_argument,
                }
            )
        except Exception:
            action_result = ""
        current_action.observation = action_result

    @staticmethod
    @chain
    async def search_wikidata(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "search_wikidata"
        action_results = search_span(
            current_action.action_argument,
            limit=8,
            return_full_results=True,
            type="item",
        )
        action_results += search_span(
            current_action.action_argument,
            limit=4,
            return_full_results=True,
            type="property",
        )
        current_action.observation = format_wikidata_search_result(action_results)

    @staticmethod
    @chain
    async def get_property_examples(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "get_property_examples"
        examples = get_property_examples(current_action.action_argument)
        action_result = []
        try:
            for e in examples:
                action_result.append(f"{e[0]} -- {e[1]} --> {e[2]}")
        except:
            logger.exception(e)
        current_action.observation = "\n".join(action_result)

    @staticmethod
    @chain
    async def stop(state):
        current_action = PartToWholeParser.get_current_action(state)
        assert current_action.action_name == "stop"
        for s in state["generated_sparqls"]:
            assert isinstance(s, SparqlQuery)
        if (
            len(state["generated_sparqls"]) == 0
            or not state["generated_sparqls"][-1].has_results()
        ):
            logger.warning("Stop() was called without a good SPARQL. Starting over.")
            state["generated_sparqls"] = []
            state["actions"] = []
            state["action_counter"] = 0
            return

        logger.info("Finished run for question %s", state["question"])
        final_sparql = state["generated_sparqls"][-1]
        if final_sparql.sparql.strip().endswith("LIMIT 10"):
            logger.info("Removing LIMIT 10 from the final SPARQL")
            s = final_sparql.sparql.strip()[:-len("LIMIT 10")]
            final_sparql = SparqlQuery(sparql=s)
            final_sparql.execute()
        return {"final_sparql": final_sparql}

    @staticmethod
    @chain
    async def reporter(state):
        actions = state["actions"]
        action_history = []
        for i, a in enumerate(actions):
            include_observation = True
            if i < len(actions) - 2 and a.action_name in [
                "get_wikidata_entry",
                "search_wikidata",
            ]:
                include_observation = False
            elif a.action_name == "stop":
                include_observation = False
            action_history.append(a.to_jinja_string(include_observation))
        
        language = await PartToWholeParser.language_detection_chain.ainvoke(
            {
                "question": state["question"],
            }
        )
        
        response = await PartToWholeParser.reporter_chain.ainvoke(
            {
                "language": language,
                "question": state["question"],
                "conversation_history": state["conversation_history"],
                "action_history": action_history
            }
        )
        state["response"] = response
        return {"response": response}