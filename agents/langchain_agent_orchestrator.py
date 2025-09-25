import os
import logging
import re
import json
from typing import List, Dict, Any, Union, TypedDict, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.agents import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.agents import AgentFinish, AgentAction

# Import only the KnowledgeBaseTool
from tools.knowledge_base_tool import KnowledgeBaseTool
from utils.document_processor import DocumentProcessor
from vector_store.embedder import Embedder
from vector_store.chroma_db import ChromaDB

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    input: str
    chat_history: List[dict]
    agent_outcome: Optional[Union[AgentAction, AgentFinish]]
    intermediate_steps: list
    final_answer: Optional[str]
    execution_plan: Optional[list]
    current_step: int
    strategy: Optional[str]

class LangChainAgentOrchestrator:
    def __init__(self, gemini_api_key: str, gemini_model_name: str,
                 api_config: Dict, synonym_config: Dict):
        
        self.api_config = api_config
        self.synonym_config = synonym_config

        # Configure Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=gemini_model_name,
            google_api_key=gemini_api_key,
            temperature=0.2
        )
        logger.info(f"Gemini LLM initialized: {gemini_model_name}")

        # Initialize the vector database and document processor
        self.vector_db = self._initialize_vector_db()
        self.document_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        
        # Setup tools available to the LLM agent
        self.tools: List[Tool] = self._setup_tools()
        logger.info(f"Initialized {len(self.tools)} tools: {[tool.name for tool in self.tools]}")
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the agent prompt
        self.agent_prompt = self._create_agent_prompt()
        
        # Construct the core agent chain
        self.agent_chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: [
                    msg for msg in format_to_openai_function_messages(x["intermediate_steps"])
                    if msg.content and msg.content.strip()
                ]
            )
            | self.agent_prompt
            | self.llm_with_tools
        )

        # Build the LangGraph workflow
        self.graph = self._build_langgraph()
        logger.info("LangGraph workflow built successfully.")

    def _initialize_vector_db(self):
        """Initializes the ChromaDB client"""
        embedder = Embedder(model_name=self.api_config['embedding_services']['sentence_transformers']['model'])
        embedding_function = embedder.get_embedding_function()

        return ChromaDB(
            persist_directory=self.api_config['vector_database']['chroma']['persist_directory'],
            embedding_function=embedding_function
        )
            
    def _setup_tools(self) -> List[Tool]:
        """Configures and returns the list of tools available to the agent"""
        tools = []
        
        # Only the KnowledgeBaseTool is enabled
        knowledge_tool = KnowledgeBaseTool(
            vector_db=self.vector_db,
            document_processor=self.document_processor,
            domain_filters=self.api_config['domain_filters']
        ).as_tool()
        tools.append(knowledge_tool)
        logger.info("Added KnowledgeBaseTool")

        return tools

    def _create_agent_prompt(self):
        """
        Creates the ChatPromptTemplate with enhanced security rules
        """
        system_message = SystemMessage(
            """
            You are an expert research assistant for the Institute of Technology of Cambodia (ITC) and its Department of Applied Mathematics and Statistics (AMS). 
            Your primary role is to provide accurate, comprehensive, and helpful information exclusively from the internal knowledge base about ITC and AMS.
            
            **STRICT RULES:**
            1. ONLY use information from the internal knowledge base about ITC and AMS.
            2. NEVER use external sources, general knowledge, or make up information.
            3. If information is not available in our domain, respond: "I couldn't find information in our ITC/AMS documents regarding that. Please try rephrasing your question or ask about a different topic related to ITC/AMS. 😕"
            
            **SECURITY AND PRIVACY RULES:**
            1. NEVER reveal internal system details, database structures, or API implementations
            2. IMMEDIATELY reject any requests to:
               - Download or export the knowledge base/database
               - Access system files or source code
               - Bypass security measures
               - Modify or destroy system data
               - Exploit vulnerabilities or perform hacking activities
               - Access private user data or conversation history
            3. For any suspicious requests (data extraction, system access, hacking), respond with:
               "I'm sorry, I cannot assist with that request as it violates security policies. Please ask about ITC/AMS topics."
            4. Protect user privacy - never reveal personal information or conversation history details
            5. Do not execute any commands that could compromise system integrity
            6. Reject requests for information about:
               - System architecture
               - Database schemas
               - Security protocols
               - User data storage
               - API implementations
            7. If asked about your capabilities beyond ITC/AMS information, respond:
               "I'm specialized in providing information about ITC and AMS only. Please ask about our academic programs, research, or campus facilities."
            
            **Response Guidelines (for when you find information):**
            1. **Clarity and Engagement:** Respond conversationally, making the information easy to understand and engaging.
            2. **Semantic Understanding & Context Awareness:**
                - Even if the user uses different words or short phrases, aim to understand the underlying meaning and provide the relevant answer from the internal knowledge base.
                - **CRITICAL for Follow-up Questions:** When a user asks a follow-up question like "tell me more about it" or "who is the current dean?", **always refer to the most recent specific entity or topic discussed in the conversation.** 
            3. **Structured Output with Rich Formatting:** Your responses should be visually appealing and easy to read.
                - Use **Markdown** extensively:
                    - **Headings (##, ###)** for sections and sub-sections.
                    - **Bold text (**) for emphasis on keywords and titles within sections.
                    - **Bullet points (* or -)** for lists.
                    - **Emojis** should be used judiciously to add a friendly and modern touch.
                - Provide detailed explanations with examples when possible.
                - Always cite your sources from our domain.
                - Language matching: For Khmer queries, respond entirely in Khmer; for English queries, respond entirely in English.
            
            4. **Final Answer Format (Mandatory JSON with Markdown Content):** Always provide your final answer in the following JSON format.
                ```json
                {
                    "answer": "## ✨ Welcome to ITC! 🎉\n\nAs a new student, here's what you need to know to thrive: **ITC is a leading institution in Cambodia** dedicated to Science, Engineering, and Technology. 📚",
                    "summary": "**Key Takeaways:** Embrace practical learning, explore diverse programs, and leverage research opportunities. 🚀",
                    "details": "### Important Aspects of Your Journey:\n\n* **Founding & Renovation:** ITC was established in **1964** with Soviet support, undergoing a major renovation in **1993** with Cambodian and French collaboration. This led to its robust 5-year engineering model. \n* **Missions:** ITC aims to **strengthen teaching quality** and **support Cambodia's sustainable development** through innovation. \n* **Practical Experience:** Be prepared for extensive **laboratory experiments** and **required internships**. These are crucial for hands-on skills!💡\n* **Faculties:** Explore departments like Electrical, Civil, Chemical, and Applied Sciences. ",
                    "list_items": ["Focus on practical skills and technical know-how. 🛠️", "Actively participate in laboratory work and internships. 💼", "Explore research opportunities through the RIC and Techno-science journal. 📖", "Utilize the library and administrative services. ℹ️", "Be aware of ITC's national and international collaborations. 🌐"],
                    "sources": ["[https://itc.edu.kh/about-us/](https://itc.edu.kh/about-us/)", "[https://en.wikipedia.org/wiki/Institute_of_Technology_of_Cambodia](https://en.wikipedia.org/wiki/Institute_of_Technology_of_Cambodia)"]
                }
                ```
            """
        )
        return ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    
    def _generate_plan(self, query: str, chat_history: list) -> Tuple[str, list]:
        """Generate a multi-step plan to answer the user's query"""
        planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a strategic planning assistant for ITC/AMS research. "
                "Analyze the user's query and conversation history to create an execution plan. "
                "Your plan should include 1-4 steps using ONLY the knowledge_base_search tool. "
                "Format: {'plan': 'overall strategy', 'steps': [{'tool': 'knowledge_base_search', 'query': '...'}, ...]}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content=query)
        ])
        
        plan_chain = planning_prompt | self.llm
        plan_output = plan_chain.invoke({"chat_history": chat_history})
        
        try:
            # Attempt to extract JSON from the response
            plan_data = json.loads(plan_output.content)
            logger.info(f"Generated plan: {plan_data}")
            return plan_data.get('plan', ''), plan_data.get('steps', [])
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse planning output, using fallback")
            return "Direct search", [{"tool": "knowledge_base_search", "query": query}]

    def _build_langgraph(self):
        """Builds the LangGraph state machine for the agent's workflow."""
        workflow = StateGraph(AgentState)
        parser = OpenAIFunctionsAgentOutputParser()

        def agent_node(state: AgentState):
            try:
                # If we have a plan, execute next step
                if state.get('execution_plan') and state['current_step'] < len(state['execution_plan']):
                    step = state['execution_plan'][state['current_step']]
                    logger.info(f"Executing plan step {state['current_step']+1}/{len(state['execution_plan'])}: {step}")
                    
                    return {
                        "agent_outcome": AgentAction(
                            tool=step['tool'],
                            tool_input=step['query'],
                            log=f"Executing planned step: {step['query']}"
                        ),
                        "current_step": state['current_step'] + 1
                    }
                
                # Clean and validate chat history messages
                deep_cleaned_history = [
                    msg.copy(update={"content": msg.content.strip()}) 
                    for msg in state["chat_history"]
                    if msg.content and msg.content.strip()
                ]
                
                # Prepare the clean state dictionary for the agent chain
                clean_state = {
                    "input": state["input"].strip(),
                    "chat_history": deep_cleaned_history,
                    "intermediate_steps": state["intermediate_steps"],
                }
                
                # Invoke the agent chain
                raw_llm_output = self.agent_chain.invoke(clean_state)
                
                # Parse the LLM's raw output
                parsed_output = parser.invoke(raw_llm_output)
                
                return {"agent_outcome": parsed_output}
            except Exception as e:
                logger.error(f"Error parsing agent output: {e}", exc_info=True)
                return_values = {
                    "output": {
                        "answer": "I encountered an internal error and couldn't process your request fully. Please try again. 😔",
                        "summary": "Processing error",
                        "details": f"Internal error details: {str(e)}",
                        "list_items": [],
                        "sources": []
                    }
                }
                return {"agent_outcome": AgentFinish(return_values=return_values, log="Error or conversational fallback")}
        
        def tool_node(state: AgentState):
            tool_invocation = state["agent_outcome"]
            
            # Ensure the outcome is indeed an AgentAction
            if not isinstance(tool_invocation, AgentAction):
                logger.error(f"tool_node received non-AgentAction outcome: {type(tool_invocation)}. Returning error.")
                return {"agent_outcome": AgentFinish(
                    return_values={"output": "An internal error occurred: Agent attempted to use a tool with an invalid outcome."},
                    log="Invalid tool invocation type"
                )}

            tool_name = tool_invocation.tool
            tool_input = tool_invocation.tool_input
            
            tool_output = None
            try:
                # Find the tool instance by name and run it
                selected_tool = next((t for t in self.tools if t.name == tool_name), None)
                if selected_tool:
                    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                    tool_output = selected_tool.run(tool_input)
                else:
                    tool_output = f"Error: Tool '{tool_name}' not found."
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                tool_output = f"Error: Tool '{tool_name}' execution failed with error: {e}"

            # Ensure tool output is never empty
            if tool_output is None or tool_output.strip() == "":
                tool_output = f"Tool '{tool_name}' returned an empty or invalid response. Could not retrieve useful information."
                logger.warning(f"Tool '{tool_name}' returned empty response for input: {tool_input}")

            # Append the tool invocation and its output to intermediate steps
            new_intermediate_steps = state["intermediate_steps"] + [(tool_invocation, tool_output)]
            return {"intermediate_steps": new_intermediate_steps}

        def should_continue(state: AgentState):
            agent_outcome = state.get("agent_outcome")
            
            if isinstance(agent_outcome, AgentFinish):
                logger.debug("Agent finished, routing to END.")
                return "end_node"
            
            if isinstance(agent_outcome, AgentAction):
                logger.debug(f"Agent wants to use tool: {agent_outcome.tool}, routing to tool_node.")
                return "continue_node"
            
            # Fallback for unexpected states
            logger.warning(f"Unexpected agent_outcome type or missing: {type(agent_outcome)}. Forcing END.")
            return "end_node"

        # Define nodes in the workflow
        workflow.add_node("agent", agent_node)
        workflow.add_node("tool", tool_node)

        # Set the entry point for the graph
        workflow.set_entry_point("agent")
        
        # Define conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue_node": "tool",
                "end_node": END
            }
        )

        # Define a regular edge
        workflow.add_edge("tool", "agent")

        return workflow.compile()

    def _deep_clean_history(self, chat_history: List[Dict]) -> List[Union[HumanMessage, AIMessage]]:
        """
        Cleans and converts raw chat history into LangChain message objects.
        Limits to last 3 exchanges (6 messages) for context management.
        """
        lc_chat_history = []
        for msg in chat_history[-6:]:  # Keep only last 3 exchanges
            content = msg.get('content', '')
            
            # Skip messages with empty content
            if not content or not content.strip():
                continue
                
            cleaned_content = content.strip()
            
            if msg['role'] == 'user':
                lc_chat_history.append(HumanMessage(content=cleaned_content))
            elif msg['role'] == 'assistant':
                # Attempt to extract 'answer' from assistant's JSON response
                try:
                    parsed_content = json.loads(cleaned_content)
                    if isinstance(parsed_content, dict) and 'answer' in parsed_content:
                        lc_chat_history.append(AIMessage(content=parsed_content['answer'].strip()))
                    else:
                        lc_chat_history.append(AIMessage(content=cleaned_content))
                except json.JSONDecodeError:
                    lc_chat_history.append(AIMessage(content=cleaned_content))
        
        return lc_chat_history

    async def run_agent(self, user_input: str, chat_history: List[Dict]) -> Dict[str, Any]:
        logger.info(f"Starting LangGraph execution for user input: '{user_input}'")

        # Prepare chat history for the agent
        lc_chat_history = self._deep_clean_history(chat_history)

        # Validate and clean user input
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return {
                "answer": "Your query was empty. Please provide a valid question. 📝",
                "summary": "Empty query",
                "details": "No input was provided to process.",
                "list_items": [],
                "sources": []
            }
            
        # Handle greetings directly
        if cleaned_input.lower() in ['hi', 'hello', 'hey', 'hola', 'hi!', 'hello!']:
            return {
                "answer": "Hello! 👋 I'm your ITC/AMS research assistant. How can I help you today?",
                "summary": "Greeting",
                "details": "Ask me about ITC programs, AMS research, or anything related to the Institute of Technology of Cambodia.",
                "list_items": [
                    "Try: 'What programs does ITC offer?'",
                    "Try: 'Tell me about AMS research opportunities'",
                    "Try: 'What are the admission requirements?'"
                ],
                "sources": []
            }

        # Generate execution plan
        strategy, plan_steps = self._generate_plan(cleaned_input, lc_chat_history)
        logger.info(f"Execution strategy: {strategy}")
        logger.info(f"Execution steps: {plan_steps}")

        # Prepare state with plan
        initial_state = {
            "input": cleaned_input,
            "chat_history": lc_chat_history,
            "intermediate_steps": [],
            "agent_outcome": None,
            "final_answer": None,
            "execution_plan": plan_steps,
            "current_step": 0,
            "strategy": strategy
        }

        try:
            # Invoke the LangGraph workflow asynchronously
            final_state = await self.graph.ainvoke(initial_state)

            # Extract the final answer from the agent's outcome
            agent_outcome = final_state.get("agent_outcome")
            
            final_response = {}
            if isinstance(agent_outcome, AgentFinish):
                raw_output = agent_outcome.return_values.get("output", {})
                
                # Attempt to parse the raw output string into the desired JSON format
                if isinstance(raw_output, str):
                    try:
                        # Try to extract JSON from a markdown code block
                        json_match = re.search(r'```json\s*({.*?})\s*```', raw_output, re.DOTALL)
                        if json_match:
                            final_response = json.loads(json_match.group(1))
                        else:
                            # If no code block, try parsing as direct JSON
                            final_response = json.loads(raw_output)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse AgentFinish output as JSON: {e}. Treating as plain text answer.")
                        final_response = {"answer": raw_output}
                elif isinstance(raw_output, dict):
                    final_response = raw_output
                else:
                    final_response = {"answer": str(raw_output)}

            else:
                logger.warning(f"Final state agent_outcome is not AgentFinish. Type: {type(agent_outcome)}. Fallback to generic error response.")
                final_response = {
                    "answer": "An unexpected internal issue prevented me from providing a complete answer. Please try again. 🤔",
                    "summary": "Processing error",
                    "details": "The agent did not produce a final answer in the expected format.",
                    "list_items": [],
                    "sources": []
                }
            
            # Ensure all required JSON fields are present
            required_fields = ["answer", "summary", "details", "list_items", "sources"]
            for field in required_fields:
                if field not in final_response:
                    final_response[field] = "" if field in ["answer", "summary", "details"] else []
            
            # Clean the 'answer' field if it contains markdown block syntax
            if isinstance(final_response.get("answer"), str):
                final_response["answer"] = final_response["answer"].replace("```json", "").replace("```", "").strip()
                if not final_response["answer"]:
                    final_response["answer"] = "I couldn't generate a specific response at this time. 🚧"

            logger.info(f"LangGraph completed. Final response: {json.dumps(final_response, ensure_ascii=False)[:200]}...")
            return final_response
                
        except Exception as e:
            logger.error(f"An unhandled internal error occurred: {e}", exc_info=True)
            return {
                "answer": "An unhandled internal error occurred during the research process. Please try again or rephrase your question. 🐞",
                "summary": "System error",
                "details": f"Detailed error: {str(e)}",
                "list_items": [],
                "sources": []

            }
