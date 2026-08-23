
"""
nodes.py

Workflow nodes for the Enterprise AI Assistant.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from graph.state import ChatState

from graph.routing import QueryRouter
from rag.citations import citation_builder
from rag.filtering import context_filter
from rag.retrieval import query_rewriter, rewrite_decider

from rag.retrieval.retriever import Retriever
from rag.prompts.prompt_builder import PromptBuilder

from core.llm import title_llm,router_llm,answer_llm,observation_llm,planner_llm,rewriter_llm
from observability.tracing import trace_node
from utils.logger import get_logger

# from rag.retrieval.query_rewriter import QueryRewriter
# query_rewriter= QueryRewriter()

# from rag.retrieval.reranker import Reranker
#
# reranker = Reranker()
#
# from rag.filtering.context_filter import ContextFilter
# context_filter=ContextFilter()
#
# from rag.citations.citation_builder import CitationBuilder
# citation_builder=CitationBuilder()

from graph.history import build_history

# from rag.retrieval.rewrite_decider import RewriteDecider
# rewrite_decider=RewriteDecider()

logger = get_logger(__name__)

from rag.services.retrieval_service import get_retrieval_service



# ==========================================================
# Shared Services
# ==========================================================

router = QueryRouter()
retrieval_service = get_retrieval_service()

prompt_builder = PromptBuilder()



# ==========================================================
# Router Node
# ==========================================================

def router_node(
    state: ChatState,
):
    with trace_node(
            "Router",
            query=state["messages"][-1].content,
    ):
        logger.info("Running Router Node")

        query = state["messages"][-1].content

        route = router.route(query)

        logger.info("Selected Route : %s", route)

        return {

            "query": query,

            "route": route,

        }


# ==========================================================
# General Chat Node
# ==========================================================

def general_chat_node(
    state: ChatState,
):
    with trace_node("General Chat"):
        logger.info("Running General Chat Node")

        response = answer_llm.invoke(
            state["messages"]
        )

        logger.info("Generated General Chat Response")

        return {

            "messages": [

                AIMessage(
                    content=response.content
                )

            ]

        }


# def rewrite_node(
#     state: ChatState,
# ):
#
#     logger.info("Running Rewrite Node")
#
#     history = build_history(
#         state["messages"]
#     )
#
#     should_rewrite = False
#
#     if history.strip():
#
#         should_rewrite = rewrite_decider.should_rewrite(
#             state["query"]
#         )
#
#     logger.info(
#         "Should Rewrite : %s",
#         should_rewrite,
#     )
#
#     if should_rewrite:
#
#         retrieval_query = query_rewriter.rewrite(
#             query=state["query"],
#             history=history,
#         )
#
#     else:
#
#         retrieval_query = state["query"]
#
#     logger.info(
#         "Retrieval Query : %s",
#         retrieval_query,
#     )
#
#     return {
#         "retrieval_query": retrieval_query,
#     }

def retrieval_node(
    state: ChatState,
):
    with trace_node(
            "Retrieval",
            query=state["query"],
    ):
        logger.info("Running Retrieval Node")

        result = retrieval_service.retrieve(

            query=state["query"],
            thread_id=state["metadata"]["thread_id"],
            messages=state["messages"][-6:],
        )

        return {

            "retrieval_result": result,

            "retrieved_documents": result.documents,

            "citations": result.citations,
        }

def generation_node(
    state: ChatState,
):
    with trace_node(
            "Generation",
            query=state["query"],
    ):
        logger.info(
            "Running Generation Node"
        )

        prompt = prompt_builder.build(

            query=state["query"],

            documents=state["retrieval_result"].documents,

        )
        print(prompt)
        response = answer_llm.invoke(
            prompt
        )

    # citations = citation_builder.build(
    #     state["retrieved_documents"]
    # )

        return {

            "prompt": prompt,

            "citations": state["retrieval_result"].citations,

            "messages": [

                AIMessage(
                    content=response.content
                )

            ],

        }

from graph.state import AgentState

from agents.planner import Planner
from agents.executor import Executor
from agents.observation import Observation
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from agents.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_context,
)
from guardrails.input_guard import get_input_guard


planner = Planner()
executor = Executor()
observer = Observation()

input_guard = get_input_guard()

def planner_node(state: AgentState):

    tool_call = planner.plan(state)

    # Planner/provider refused or failed
    if tool_call is None:

        logger.warning(
            "Planner returned no tool call. "
            "Ending agent execution safely."
        )

        return {
            "tool_call": None,
            "status": "PLANNER_BLOCKED",
            "final_answer": (
                "I can't help with that request."
            ),
        }

    return {
        "tool_call": tool_call,
        "status": "PLANNING",
    }

def executor_node(
    state: AgentState,
):
    with trace_node("Executor"):
        result = executor.execute(
            tool_call= state["tool_call"],
            state=state
        )

        return {

            "tool_result": result,

            "tool_results": state[
                "tool_results"
            ] + [result],

            "status": "EXECUTING",
        }


def observation_node(
    state: AgentState,
):
    with trace_node("Observation"):
        decision = observer.observe(state)

        return {

            "observation": decision,

            "iteration": state[
                "iteration"
            ] + 1,

            "status": "OBSERVING",
        }

def answer_node(
    state: AgentState,
):
    if state.get("status") == "PLANNER_BLOCKED":
        return {
            "final_answer": state["final_answer"],
        }
    with trace_node("Answer"):
        context = build_answer_context(

            query=state["query"],

            tool_results=state["tool_results"],
        )

        messages = [

            SystemMessage(

                content=ANSWER_SYSTEM_PROMPT,
            ),

            HumanMessage(

                content=context,
            ),
        ]

        response = answer_llm.invoke(
            messages
        )

        return {

            "final_answer": response.content,

            "messages": [

                AIMessage(
                    content=response.content
                )

            ],

            "status": "FINISHED",
        }



# ==========================================================
# Input Guardrail Node
# ==========================================================
from guardrails.responses import get_guardrail_response
def input_guardrail_node(
    state: AgentState,
):
    """
    Validate user input before it enters
    the main LangGraph workflow.
    """

    with trace_node("Input Guardrail"):

        query = state["messages"][-1].content
        logger.info("INPUT GUARDRAIL RECEIVED QUERY: %s", query)
        result = input_guard.validate(
            query
        )
        logger.info(
            "Guardrail result | allowed=%s | reason=%s",
            result.allowed,
            result.reason,
        )
        if not result.allowed:
            response=get_guardrail_response(result.category)
            logger.warning(
                "Input blocked by guardrail: %s",
                result.reason,
            )

            return {
                "query": query,
                "guardrail_status": "BLOCKED",
                "guardrail_reason": result.reason,
                "final_answer": response,
                "messages": [
                    AIMessage(
                        content=response
                    )
                ],
                "status": "BLOCKED",
            }

        logger.info(
            "Input passed guardrail validation"
        )

        return {
            "query": query,
            "guardrail_status": "SAFE",
            "guardrail_reason": None,
        }



from guardrails.tool_guard import get_tool_guard
tool_guard = get_tool_guard()
def tool_guardrail_node(
    state: AgentState,
):
    """
    Validate the selected tool and its arguments
    before the executor runs.
    """

    with trace_node("Tool Guardrail"):

        tool_call = state.get("tool_call")

        # --------------------------------------------------
        # No tool selected
        # --------------------------------------------------

        if tool_call is None:

            logger.info(
                "No tool selected. Skipping tool guardrail."
            )

            return {
                "tool_guardrail_status": "SAFE",
                "tool_guardrail_reason": None,
            }

        # --------------------------------------------------
        # Get arguments safely
        # --------------------------------------------------

        # Supports different ToolCall schemas:
        # arguments / args / parameters

        arguments = getattr(
            tool_call,
            "arguments",
            None,
        )

        if arguments is None:

            arguments = getattr(
                tool_call,
                "args",
                None,
            )

        if arguments is None:

            arguments = getattr(
                tool_call,
                "parameters",
                None,
            )

        # --------------------------------------------------
        # Validate
        # --------------------------------------------------

        result = tool_guard.validate(
            tool_name=tool_call.tool,
            arguments=arguments,
        )

        # --------------------------------------------------
        # Blocked
        # --------------------------------------------------

        if not result.allowed:

            logger.warning(
                "Tool call blocked | "
                "tool=%s | category=%s | reason=%s",
                tool_call.tool,
                result.category,
                result.reason,
            )

            return {
                "tool_guardrail_status": "BLOCKED",

                "tool_guardrail_reason": (
                    result.reason
                ),

                "tool_result": {
                    "success": False,
                    "blocked": True,
                    "tool": tool_call.tool,
                    "category": result.category,
                    "error": result.reason,
                },

                "status": "TOOL_BLOCKED",
            }

        # --------------------------------------------------
        # Safe
        # --------------------------------------------------

        logger.info(
            "Tool call passed guardrail | tool=%s",
            tool_call.tool,
        )

        return {
            "tool_guardrail_status": "SAFE",
            "tool_guardrail_reason": None,
        }