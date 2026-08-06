#
# from langgraph.graph import (
#     StateGraph,
#     START,
#     END
# )
#
# from graph.state import ChatState
# from graph.nodes import chat_node
#
# from db.checkpoint import checkpointer
#
#
# graph = StateGraph(ChatState)
#
# graph.add_node(
#     "chat_node",
#     chat_node
# )
#
# graph.add_edge(
#     START,
#     "chat_node"
# )
#
# graph.add_edge(
#     "chat_node",
#     END
# )
# #
# chatbot = graph.compile(
#     checkpointer=checkpointer
# )