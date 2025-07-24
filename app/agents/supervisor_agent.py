"""
Supervisor Agent - Orchestrates multiple specialized agents using LangGraph's supervisor pattern.
This agent analyzes incoming tasks and delegates them to the most appropriate specialized agent.
"""

from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from app.api.settings import get_api_key
from app.agents.notion_agent import NotionAgent
from app.agents.math_agent import MathAgent
from app.agents.transport_agent import TransportAgent
from app.agents.weather_agent import WeatherAgent


class SupervisorState(BaseModel):
    """State management for the supervisor workflow."""
    messages: List[Dict[str, Any]] = []
    task: str = ""
    sub_tasks: List[Dict[str, Any]] = []  # List of decomposed sub-tasks
    selected_agent: Optional[str] = None
    agent_response: Optional[Dict[str, Any]] = None
    agent_responses: List[Dict[str, Any]] = []  # Responses from multiple agents
    final_response: str = ""
    reasoning: str = ""
    confidence_scores: Dict[str, float] = {}
    current_task_index: int = 0  # Track which sub-task we're processing


class SupervisorAgent:
    """
    Supervisor agent that coordinates multiple specialized agents.
    Uses LangGraph's state management to orchestrate agent selection and task delegation.
    """
    
    def __init__(self):
        """Initialize the supervisor with specialized agents and memory management."""
        self.name = "supervisor_agent"
        self.description = (
            "I am a supervisor agent that coordinates between specialized agents. "
            "I analyze your request and delegate it to the most appropriate agent."
        )
        
        # Memory management configuration
        self.max_messages_per_user = 10  # Maximum messages before summarization
        
        # Initialize specialized agents
        self.agents = {
            "notion_agent": NotionAgent(),
            "transport_agent": TransportAgent(),
            "weather_agent": WeatherAgent(),
            "math_agent": MathAgent()
        }
        
        self.supervisor_model = self._create_supervisor_model()
        self.workflow = self._create_workflow()
    
    def _create_supervisor_model(self) -> ChatGoogleGenerativeAI:
        """Create the supervisor's reasoning model."""
        api_key = get_api_key()
        if not api_key:
            raise ValueError("Google API key is not configured")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,  # Low temperature for consistent reasoning
            max_retries=2,
            google_api_key=api_key
        )
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for agent supervision."""
        from typing import Annotated
        
        # Define the state schema for LangGraph - using simple dict for messages
        class WorkflowState(BaseModel):
            messages: List[Dict[str, Any]] = []
            task: str = ""
            sub_tasks: List[Dict[str, Any]] = []
            selected_agent: Optional[str] = None
            agent_response: Optional[Dict[str, Any]] = None
            agent_responses: List[Dict[str, Any]] = []
            final_response: str = ""
            reasoning: str = ""
            confidence_scores: Dict[str, float] = {}
            timestamp_context: Dict[str, Any] = {}
            current_task_index: int = 0
        
        workflow = StateGraph(WorkflowState)
        
        # Add nodes - simplified workflow for better performance
        workflow.add_node("decompose_and_execute", self._decompose_and_execute_parallel)
        workflow.add_node("aggregate_responses", self._aggregate_responses)
        
        # Add edges - simplified linear flow
        workflow.set_entry_point("decompose_and_execute")
        workflow.add_edge("decompose_and_execute", "aggregate_responses")
        workflow.add_edge("aggregate_responses", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _manage_conversation_memory(self, messages: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
        """
        Manage conversation memory with 10-message limit and summarization.

        Args:
            messages: Current conversation messages
            user_id: User identifier for logging/debugging
            
        Returns:
            Managed message list with maximum 10 messages
        """
        # If we're within the limit, return as-is
        if len(messages) <= self.max_messages_per_user:
            return messages
        
        # We need to summarize - take all but the last message (current user message)
        messages_to_summarize = messages[:-1]  # All except the current message
        current_message = messages[-1]  # The current user message
        
        # Create summary of the conversation
        summary = self._summarize_conversation(messages_to_summarize, user_id)
        
        # Create a summary message
        summary_message = {
            "role": "system", 
            "content": f"Previous conversation summary: {summary}",
            "type": "summary"
        }
        
        # Return: [summary_message, current_user_message]
        return [summary_message, current_message]
    
    def _summarize_conversation(self, messages: List[Dict[str, Any]], user_id: str) -> str:
        """
        Summarize a conversation using the supervisor model.
        
        Args:
            messages: Messages to summarize
            user_id: User identifier
            
        Returns:
            Conversation summary
        """
        try:
            # Create conversation text for summarization
            conversation_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in messages
                if msg.get('content') and msg.get('type') != 'summary'  # Skip previous summaries
            ])
            
            if not conversation_text.strip():
                return "No significant conversation history to summarize."
            
            summary_prompt = f"""
            Summarize this conversation concisely, focusing on:
            1. Key topics discussed
            2. Important calculations, results, or answers provided
            3. User preferences or context that should be remembered
            4. Any ongoing tasks or requests
            
            Conversation:
            {conversation_text}
            
            Provide a concise summary (max 150 words):
            """
            
            response = self.supervisor_model.invoke(summary_prompt)
            return response.content[:400]  # Limit summary length
            
        except Exception as e:
            return f"Conversation included topics discussed previously (summary unavailable: {str(e)})"
    
    def _decompose_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose the main task into sub-tasks that can be handled by different agents."""
        # Handle both dict and object state formats
        if hasattr(state, 'task'):
            task = state.task
        else:
            task = state.get("task", "")
        
        # Check if the task contains multiple questions or requests
        decomposition_prompt = f"""
You are a task decomposition expert. Analyze this user request and break it down into separate sub-tasks if it contains multiple distinct questions.

User request: "{task}"

Available agent types and their capabilities:
- notion_agent: Calendar events, scheduling, Notion database operations, event management
- transport_agent: Public transport directions, travel routes, transportation information
- weather_agent: Weather forecasts, weather conditions, temperature information
- math_agent: Mathematical calculations, arithmetic operations, math problems

Rules for decomposition:
1. If the request contains multiple distinct questions/tasks separated by "and", "also", commas, or similar connectors, break them down
2. Each sub-task should be focused on ONE specific question/request
3. Assign the most appropriate agent for each sub-task based on the content
4. Return a JSON array with each sub-task as an object

Examples:
Input: "What's the weather tomorrow and show me my calendar events?"
Output: [
  {{"task": "What's the weather tomorrow?", "suggested_agent": "weather_agent"}},
  {{"task": "Show me my calendar events", "suggested_agent": "notion_agent"}}
]

Input: "Calculate 25 * 30 and how do I get from Marina Bay to Orchard Road?"
Output: [
  {{"task": "Calculate 25 * 30", "suggested_agent": "math_agent"}},
  {{"task": "How do I get from Marina Bay to Orchard Road?", "suggested_agent": "transport_agent"}}
]

Input: "What's the weather like today?"
Output: [
  {{"task": "What's the weather like today?", "suggested_agent": "weather_agent"}}
]

Now analyze: "{task}"

Return ONLY the JSON array, no other text:"""
        
        try:
            response = self.supervisor_model.invoke(decomposition_prompt)
            decomposition_text = response.content.strip()
            
            # Clean up the response - remove any markdown formatting
            if decomposition_text.startswith("```json"):
                decomposition_text = decomposition_text[7:]
            if decomposition_text.endswith("```"):
                decomposition_text = decomposition_text[:-3]
            decomposition_text = decomposition_text.strip()
            
            # Try to parse as JSON
            import json
            try:
                sub_tasks = json.loads(decomposition_text)
                # Validate format
                if not isinstance(sub_tasks, list) or not sub_tasks:
                    raise ValueError("Invalid format")
                
                # Ensure each sub-task has required fields
                for i, subtask in enumerate(sub_tasks):
                    if not isinstance(subtask, dict) or "task" not in subtask:
                        subtask["task"] = task  # Fallback to original task
                    if "suggested_agent" not in subtask:
                        subtask["suggested_agent"] = "notion_agent"  # Default
                    
                print(f"✅ Task decomposition successful: {len(sub_tasks)} sub-tasks identified")
                for i, st in enumerate(sub_tasks):
                    print(f"  {i+1}. '{st['task']}' → {st['suggested_agent']}")
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: try to manually parse if we can detect multiple questions
                print(f"⚠️ JSON parsing failed: {e}. Attempting manual decomposition...")
                sub_tasks = self._manual_decomposition(task)
                
        except Exception as e:
            # Fallback on any error
            print(f"⚠️ Decomposition failed: {e}. Using single task fallback.")
            sub_tasks = [{"task": task, "suggested_agent": "notion_agent"}]
        
        return {
            "sub_tasks": sub_tasks,
            "current_task_index": 0,
            "agent_responses": []
        }
    
    def _manual_decomposition(self, task: str) -> List[Dict[str, Any]]:
        """Manual task decomposition as fallback when LLM fails - optimized for speed."""
        import re
        
        # Enhanced patterns to detect multiple questions - more aggressive splitting
        patterns = [
            # Split on ", and " or " and " with more context
            r',\s*and\s+',
            r'\s+and\s+(?=\w)',
            # Split on conjunctions with common question words
            r'(?<=\?)\s*(?=[A-Z]|\b(?:what|how|when|where|show|get|calculate|tell)\b)',
            # Split on sentence boundaries when multiple commands are present
            r'(?<=\.)\s*(?=[A-Z])',
            r';\s*',
            # Split on imperative patterns
            r',\s*(?=(?:show|get|calculate|tell|find|book|schedule)\b)',
        ]
        
        # Start with the full task
        parts = [task.strip()]
        
        # Apply each pattern
        for pattern in patterns:
            new_parts = []
            for part in parts:
                if len(part) > 15:  # Only split longer parts
                    split_parts = re.split(pattern, part, flags=re.IGNORECASE)
                    if len(split_parts) > 1:
                        # Clean and filter parts
                        cleaned_parts = []
                        for p in split_parts:
                            p = p.strip()
                            # Remove empty parts and parts that are too short
                            if p and len(p) > 5:
                                # If part doesn't end with punctuation, add appropriate punctuation
                                if not p.endswith(('.', '?', '!')):
                                    if any(word in p.lower() for word in ['what', 'how', 'when', 'where', 'show', 'get', 'tell']):
                                        p += '?'
                                    else:
                                        p += '.'
                                cleaned_parts.append(p)
                        
                        if len(cleaned_parts) > 1:
                            new_parts.extend(cleaned_parts)
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            parts = new_parts
            
            # If we successfully split into multiple parts, stop trying more patterns
            if len(parts) > 1:
                break
        
        # Additional smart splitting for common patterns
        if len(parts) == 1:
            task_lower = task.lower()
            # Look for calculation + other task patterns
            calc_pattern = r'(calculate|compute|find|solve)\s+[^,?]+(?:\?|,|\s+and\s+)'
            weather_pattern = r'(weather|temperature|forecast|rain|sunny|cloudy)[^,?]*(?:\?|,|\s+and\s+)'
            transport_pattern = r'(direction|transport|get\s+to|travel|route|mrt|bus)[^,?]*(?:\?|,|\s+and\s+)'
            calendar_pattern = r'(calendar|event|schedule|meeting|appointment|book)[^,?]*(?:\?|,|\s+and\s+)'
            
            found_patterns = []
            for pattern, agent in [(calc_pattern, 'math_agent'), 
                                 (weather_pattern, 'weather_agent'), 
                                 (transport_pattern, 'transport_agent'),
                                 (calendar_pattern, 'notion_agent')]:
                matches = re.finditer(pattern, task, re.IGNORECASE)
                for match in matches:
                    found_patterns.append((match.start(), match.end(), agent, match.group()))
            
            # If we found multiple patterns, try to split based on them
            if len(found_patterns) > 1:
                found_patterns.sort(key=lambda x: x[0])  # Sort by start position
                parts = []
                last_end = 0
                
                for start, end, agent, matched_text in found_patterns:
                    # Extract text around the pattern
                    if start > last_end:
                        # Get some context before the pattern
                        context_start = max(last_end, start - 20)
                        full_part = task[context_start:end].strip()
                    else:
                        full_part = matched_text.strip()
                    
                    if full_part and len(full_part) > 5:
                        if not full_part.endswith(('.', '?', '!')):
                            if any(word in full_part.lower() for word in ['what', 'how', 'when', 'where', 'show', 'get']):
                                full_part += '?'
                        parts.append(full_part)
                    
                    last_end = end
                
                # Get any remaining text
                if last_end < len(task):
                    remaining = task[last_end:].strip()
                    if remaining and len(remaining) > 5:
                        if not remaining.endswith(('.', '?', '!')):
                            remaining += '.'
                        parts.append(remaining)
        
        # Create sub-tasks from parts
        if len(parts) > 1:
            sub_tasks = []
            for part in parts:
                part = part.strip()
                if len(part) > 5:  # Filter out very short parts
                    suggested_agent = self._quick_agent_assignment(part)
                    sub_tasks.append({"task": part, "suggested_agent": suggested_agent})
            
            if len(sub_tasks) > 1:
                print(f"✅ Enhanced manual decomposition: {len(sub_tasks)} sub-tasks")
                for i, st in enumerate(sub_tasks):
                    print(f"   {i+1}. '{st['task']}' → {st['suggested_agent']}")
                return sub_tasks
        
        # Single task fallback
        return [{"task": task, "suggested_agent": self._quick_agent_assignment(task)}]
    
    def _decompose_and_execute_parallel(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose task and execute all sub-tasks in parallel for better performance."""
        import asyncio
        import concurrent.futures
        from functools import partial
        
        # Handle both dict and object state formats
        if hasattr(state, 'task'):
            task = state.task
            messages = state.messages
            timestamp_context = state.timestamp_context
        else:
            task = state.get("task", "")
            messages = state.get("messages", [])
            timestamp_context = state.get("timestamp_context", {})
        
        # Step 1: Fast decomposition with simpler logic
        sub_tasks = self._fast_decompose_task(task)
        print(f"⚡ Fast decomposition: {len(sub_tasks)} sub-tasks identified")
        
        # Step 2: Execute all sub-tasks in parallel
        if len(sub_tasks) == 1:
            # Single task - no need for parallel processing
            agent_responses = [self._execute_single_subtask(sub_tasks[0], task, messages, timestamp_context, 0)]
        else:
            # Multiple tasks - use parallel execution
            agent_responses = self._execute_subtasks_parallel(sub_tasks, task, messages, timestamp_context)
        
        return {
            "sub_tasks": sub_tasks,
            "agent_responses": agent_responses,
            "current_task_index": len(sub_tasks)  # Mark as completed
        }
    
    def _fast_decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """Fast task decomposition using pattern matching and minimal LLM calls."""
        task_lower = task.lower()
        
        # Quick check for obvious single tasks
        if (len(task.split()) < 6 or 
            (task.count('?') <= 1 and ' and ' not in task_lower and ', ' not in task_lower)):
            return [{"task": task, "suggested_agent": self._quick_agent_assignment(task)}]
        
        # For multi-task candidates, go straight to LLM for best accuracy
        try:
            llm_result = self._llm_decompose_task(task)
            if len(llm_result) > 1:
                print(f"✅ LLM decomposition: {len(llm_result)} sub-tasks")
                for i, st in enumerate(llm_result):
                    print(f"   {i+1}. '{st['task']}' → {st['suggested_agent']}")
                return llm_result
        except Exception as e:
            print(f"⚠️ LLM decomposition failed: {e}")
        
        # Fallback to simple manual split
        if ' and ' in task:
            parts = [p.strip() for p in task.split(' and ') if p.strip() and len(p.strip()) > 3]
            if len(parts) > 1:
                result = []
                for part in parts:
                    if not part.endswith(('?', '.', '!')):
                        if any(word in part.lower() for word in ['what', 'how', 'show', 'get', 'tell']):
                            part += '?'
                    result.append({"task": part, "suggested_agent": self._quick_agent_assignment(part)})
                print(f"✅ Simple split: {len(result)} sub-tasks")
                return result
        
        # Single task fallback
        return [{"task": task, "suggested_agent": self._quick_agent_assignment(task)}]
    
    def _enhanced_manual_decomposition(self, task: str) -> List[Dict[str, Any]]:
        """Enhanced manual decomposition using multiple strategies."""
        import re
        
        # Strategy 1: Split on clear separators
        clear_separators = [
            r',\s*and\s+',           # ", and "
            r'\?\s*(?=[A-Z]|\b(?:what|how|when|where|show|get|calculate|tell|find)\b)',  # ? followed by question word
            r';\s*',                 # semicolon
            r'\.\s*(?=[A-Z])',       # period followed by capital
        ]
        
        parts = [task.strip()]
        
        for separator in clear_separators:
            new_parts = []
            for part in parts:
                split_parts = re.split(separator, part, flags=re.IGNORECASE)
                if len(split_parts) > 1:
                    new_parts.extend([p.strip() for p in split_parts if p.strip() and len(p.strip()) > 5])
                else:
                    new_parts.append(part)
            parts = new_parts
            
            if len(parts) > 1:
                break  # Found a good split, stop trying
        
        # Strategy 2: If still one part, try semantic splitting
        if len(parts) == 1:
            parts = self._semantic_split(task)
        
        # Clean and validate parts
        if len(parts) > 1:
            sub_tasks = []
            for part in parts:
                part = part.strip()
                if len(part) > 5:  # Filter very short parts
                    # Add appropriate punctuation
                    if not part.endswith(('.', '?', '!')):
                        if any(word in part.lower() for word in ['what', 'how', 'when', 'where', 'show', 'get', 'tell', 'find']):
                            part += '?'
                        else:
                            part += '.'
                    
                    suggested_agent = self._quick_agent_assignment(part)
                    sub_tasks.append({"task": part, "suggested_agent": suggested_agent})
            
            return sub_tasks if len(sub_tasks) > 1 else [{"task": task, "suggested_agent": self._quick_agent_assignment(task)}]
        
        return [{"task": task, "suggested_agent": self._quick_agent_assignment(task)}]
    
    def _semantic_split(self, task: str) -> List[str]:
        """Split task based on semantic patterns and keywords."""
        import re
        
        # Define patterns for different agent types with their keywords
        agent_patterns = [
            (r'\b(?:calculate|compute|solve|find|what\s*(?:is|are)|multiply|divide|add|subtract|\d+\s*[+\-*/]\s*\d+)', 'math'),
            (r'\b(?:weather|temperature|forecast|rain|sunny|cloudy|hot|cold|climate)', 'weather'),
            (r'\b(?:direction|transport|get\s+to|travel|route|mrt|bus|train|taxi|uber|grab)', 'transport'),
            (r'\b(?:calendar|event|schedule|meeting|appointment|book|plan|today|tomorrow)', 'notion'),
        ]
        
        # Find all matches with their positions
        matches = []
        for pattern, agent_type in agent_patterns:
            for match in re.finditer(pattern, task, re.IGNORECASE):
                matches.append((match.start(), match.end(), agent_type, match.group()))
        
        if len(matches) < 2:
            return [task]  # Not enough patterns to split
        
        # Sort matches by position
        matches.sort()
        
        # Try to create meaningful splits
        parts = []
        last_end = 0
        
        for i, (start, end, agent_type, matched_text) in enumerate(matches):
            # Determine split boundaries
            if i == 0:
                # First match - take from beginning
                split_start = 0
            else:
                # Find a good split point between previous and current match
                between_text = task[matches[i-1][1]:start]
                if ' and ' in between_text:
                    split_point = task.rfind(' and ', 0, start)
                    split_start = split_point + 5 if split_point != -1 else matches[i-1][1]
                elif ', ' in between_text:
                    split_point = task.rfind(', ', 0, start)
                    split_start = split_point + 2 if split_point != -1 else matches[i-1][1]
                else:
                    split_start = matches[i-1][1]
            
            # Determine end point
            if i == len(matches) - 1:
                # Last match - take to end
                split_end = len(task)
            else:
                # Find good end point
                to_next = task[end:matches[i+1][0]]
                if ' and ' in to_next:
                    end_point = task.find(' and ', end)
                    split_end = end_point if end_point != -1 else matches[i+1][0]
                elif ', ' in to_next:
                    end_point = task.find(', ', end)
                    split_end = end_point if end_point != -1 else matches[i+1][0]
                else:
                    split_end = matches[i+1][0]
            
            part = task[split_start:split_end].strip()
            if part and len(part) > 5 and part not in parts:
                parts.append(part)
        
        # If we got meaningful parts, return them
        if len(parts) > 1:
            return parts
        
        # Fallback: simple split on " and "
        if ' and ' in task:
            return [part.strip() for part in task.split(' and ') if part.strip()]
        
        return [task]
    
    def _quick_agent_assignment(self, task: str) -> str:
        """Quick agent assignment based on keywords - no LLM call."""
        task_lower = task.lower()
        
        # Priority-based assignment (most specific first)
        if any(word in task_lower for word in ['calculate', 'multiply', 'divide', 'add', 'subtract', 'math', '+', '-', '*', '/', 'equation', 'solve']):
            return "math_agent"
        elif any(word in task_lower for word in ['weather', 'temperature', 'rain', 'sunny', 'cloudy', 'forecast', 'climate']):
            return "weather_agent"
        elif any(word in task_lower for word in ['direction', 'transport', 'mrt', 'bus', 'get to', 'travel', 'route', 'train', 'taxi']):
            return "transport_agent"
        elif any(word in task_lower for word in ['calendar', 'event', 'schedule', 'meeting', 'appointment', 'book', 'plan']):
            return "notion_agent"
        else:
            return "notion_agent"  # Default
    
    def _llm_decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """LLM-based decomposition - streamlined for better performance."""
        
        # Simplified, faster prompt
        decomposition_prompt = f"""Split this request into separate tasks. Each task should go to one agent.

Request: "{task}"

Agents:
- math_agent: calculations, math
- weather_agent: weather, temperature  
- transport_agent: directions, travel
- notion_agent: calendar, events, scheduling

Return JSON array only:
[{{"task": "...", "agent": "..."}}]

Examples:
"Calculate 5*3 and weather tomorrow" → [{{"task": "Calculate 5*3", "agent": "math_agent"}}, {{"task": "Weather tomorrow", "agent": "weather_agent"}}]
"Show calendar today" → [{{"task": "Show calendar today", "agent": "notion_agent"}}]

JSON:"""
        
        try:
            response = self.supervisor_model.invoke(decomposition_prompt)
            text = response.content.strip()
            
            # Clean response
            if text.startswith("```"):
                lines = text.split('\n')
                text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text
            
            import json
            result = json.loads(text)
            
            if isinstance(result, list) and result:
                # Convert to expected format
                sub_tasks = []
                for item in result:
                    if isinstance(item, dict):
                        task_text = item.get('task', '').strip()
                        agent = item.get('agent', item.get('suggested_agent', 'notion_agent'))
                        if task_text:
                            sub_tasks.append({"task": task_text, "suggested_agent": agent})
                
                if sub_tasks:
                    return sub_tasks
            
        except Exception as e:
            print(f"LLM parse error: {e}")
        
        # Fallback to simple split
        if ' and ' in task:
            parts = [p.strip() for p in task.split(' and ') if p.strip()]
            if len(parts) > 1:
                return [{"task": part, "suggested_agent": self._quick_agent_assignment(part)} for part in parts]
        
        return [{"task": task, "suggested_agent": self._quick_agent_assignment(task)}]
    
    def _execute_subtasks_parallel(self, sub_tasks: List[Dict[str, Any]], original_task: str, 
                                 messages: List[Dict[str, Any]], timestamp_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute multiple sub-tasks in parallel using ThreadPoolExecutor."""
        import concurrent.futures
        import threading
        
        def execute_subtask_wrapper(subtask_data):
            subtask, index = subtask_data
            try:
                return self._execute_single_subtask(subtask, original_task, messages, timestamp_context, index)
            except Exception as e:
                return {
                    "success": False,
                    "response": f"Parallel execution failed: {str(e)}",
                    "error": str(e),
                    "sub_task_index": index,
                    "sub_task": subtask.get("task", "Unknown")
                }
        
        # Execute in parallel with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sub_tasks), 4)) as executor:
            # Prepare data for parallel execution
            subtask_data = [(subtask, i) for i, subtask in enumerate(sub_tasks)]
            
            # Submit all tasks
            future_to_index = {
                executor.submit(execute_subtask_wrapper, data): i 
                for i, data in enumerate(subtask_data)
            }
            
            # Collect results in order
            results = [None] * len(sub_tasks)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per task
                    results[index] = result
                except concurrent.futures.TimeoutError:
                    results[index] = {
                        "success": False,
                        "response": "Task timed out after 30 seconds",
                        "error": "Timeout",
                        "sub_task_index": index,
                        "sub_task": sub_tasks[index].get("task", "Unknown")
                    }
                except Exception as e:
                    results[index] = {
                        "success": False,
                        "response": f"Task execution failed: {str(e)}",
                        "error": str(e),
                        "sub_task_index": index,
                        "sub_task": sub_tasks[index].get("task", "Unknown")
                    }
        
        # Filter out None results (shouldn't happen, but safety check)
        return [r for r in results if r is not None]
    
    def _execute_single_subtask(self, subtask: Dict[str, Any], original_task: str, 
                              messages: List[Dict[str, Any]], timestamp_context: Dict[str, Any], 
                              index: int) -> Dict[str, Any]:
        """Execute a single sub-task with optimized agent selection."""
        subtask_text = subtask.get("task", "")
        suggested_agent = subtask.get("suggested_agent", "notion_agent")
        
        # Fast agent selection - use suggestion if agent exists, otherwise quick assignment
        if suggested_agent in self.agents:
            selected_agent_name = suggested_agent
        else:
            selected_agent_name = self._quick_agent_assignment(subtask_text)
        
        selected_agent = self.agents[selected_agent_name]
        
        try:
            # Build enhanced task with minimal context processing
            enhanced_task = subtask_text
            
            # Add multi-task context only if needed
            if len(messages) > 1:  # Has conversation history
                # Simplified context - just add the original task context
                enhanced_task = f"[Context: Part of larger request: '{original_task}']\n\nCurrent task: {subtask_text}"
            
            # Add timestamp context for time-aware agents
            if selected_agent_name in ["notion_agent", "weather_agent"] and timestamp_context:
                timestamp_info = f"""
[TIME] Singapore: {timestamp_context.get('singapore_date', 'Unknown')} ({timestamp_context.get('singapore_time', 'Unknown')})"""
                enhanced_task = enhanced_task + timestamp_info
            
            # Execute the task
            response = selected_agent.process_task(enhanced_task)
            
            # Add metadata
            response["agent_used"] = selected_agent_name
            response["sub_task_index"] = index
            response["sub_task"] = subtask_text
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Agent execution failed: {str(e)}",
                "error": str(e),
                "agent": selected_agent_name,
                "sub_task_index": index,
                "sub_task": subtask_text
            }
    
    def _analyze_subtask(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current sub-task and get confidence scores."""
        # Handle both dict and object state formats
        if hasattr(state, 'sub_tasks'):
            sub_tasks = state.sub_tasks
            current_index = state.current_task_index
        else:
            sub_tasks = state.get("sub_tasks", [])
            current_index = state.get("current_task_index", 0)
        
        if not sub_tasks or current_index >= len(sub_tasks):
            return {"confidence_scores": {}}
        
        current_subtask = sub_tasks[current_index]
        subtask_text = current_subtask.get("task", "")
        
        confidence_scores = {}
        
        # Get confidence scores from each agent for this specific sub-task
        for agent_name, agent in self.agents.items():
            confidence_scores[agent_name] = agent.can_handle(subtask_text)
        
        # Add reasoning about the sub-task
        reasoning_prompt = f"""
        Analyzing sub-task {current_index + 1} of {len(sub_tasks)}: "{subtask_text}"
        Suggested agent: {current_subtask.get('suggested_agent', 'unknown')}
        
        Agent confidence scores:
        {chr(10).join([f"- {name}: {score:.2f}" for name, score in confidence_scores.items()])}
        
        Provide a brief analysis of this sub-task and agent suitability.
        """
        
        reasoning = ""
        try:
            response = self.supervisor_model.invoke(reasoning_prompt)
            reasoning = response.content
        except Exception as e:
            reasoning = f"Analysis error: {str(e)}"
        
        return {
            "confidence_scores": confidence_scores,
            "reasoning": reasoning
        }
    
    def _select_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select the most appropriate agent based on confidence scores for current sub-task."""
        # Handle both dict and object state formats
        if hasattr(state, 'confidence_scores'):
            confidence_scores = state.confidence_scores
            sub_tasks = state.sub_tasks
            current_index = state.current_task_index
        else:
            confidence_scores = state.get("confidence_scores", {})
            sub_tasks = state.get("sub_tasks", [])
            current_index = state.get("current_task_index", 0)
        
        if not confidence_scores:
            return {"selected_agent": "notion_agent"}  # Default fallback
        
        # Get suggested agent from decomposition if available
        suggested_agent = None
        if sub_tasks and current_index < len(sub_tasks):
            suggested_agent = sub_tasks[current_index].get("suggested_agent")
        
        # Select agent with highest confidence score, but consider suggestion
        selected_agent = max(confidence_scores.items(), key=lambda x: x[1])[0]
        
        # If suggested agent has reasonable confidence (>0.2), prefer it
        if (suggested_agent and 
            suggested_agent in confidence_scores and 
            confidence_scores[suggested_agent] > 0.2 and
            confidence_scores[suggested_agent] >= max(confidence_scores.values()) - 0.3):
            selected_agent = suggested_agent
        
        # If confidence is too low across all agents, use a general approach
        max_confidence = max(confidence_scores.values())
        reasoning_update = ""
        if max_confidence < 0.3:
            reasoning_update = f"\n\nNote: Low confidence across all agents (max: {max_confidence:.2f}). Using {selected_agent} as best option."
        
        result = {"selected_agent": selected_agent}
        if reasoning_update:
            if hasattr(state, 'reasoning'):
                current_reasoning = state.reasoning
            else:
                current_reasoning = state.get("reasoning", "")
            result["reasoning"] = current_reasoning + reasoning_update
        
        return result
    
    def _execute_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the current sub-task using the selected agent with conversation context."""
        # Handle both dict and object state formats
        if hasattr(state, 'selected_agent'):
            selected_agent_name = state.selected_agent
            sub_tasks = state.sub_tasks
            current_index = state.current_task_index
            messages = state.messages
            timestamp_context = state.timestamp_context
            agent_responses = state.agent_responses
        else:
            selected_agent_name = state.get("selected_agent")
            sub_tasks = state.get("sub_tasks", [])
            current_index = state.get("current_task_index", 0)
            messages = state.get("messages", [])
            timestamp_context = state.get("timestamp_context", {})
            agent_responses = state.get("agent_responses", [])
        
        if not selected_agent_name or selected_agent_name not in self.agents:
            error_response = {
                "success": False,
                "response": "No suitable agent found",
                "error": "Agent selection failed",
                "sub_task_index": current_index
            }
            return {"agent_responses": agent_responses + [error_response]}
        
        # Get current sub-task
        if not sub_tasks or current_index >= len(sub_tasks):
            error_response = {
                "success": False,
                "response": "No sub-task to execute",
                "error": "Sub-task index out of range",
                "sub_task_index": current_index
            }
            return {"agent_responses": agent_responses + [error_response]}
        
        current_subtask = sub_tasks[current_index]
        subtask_text = current_subtask.get("task", "")
        selected_agent = self.agents[selected_agent_name]
        
        try:
            # Build context from conversation history for the agent
            if len(messages) > 1:  # More than just the current message
                # Filter out the current message and any system summaries for context
                context_messages = [
                    msg for msg in messages[:-1] 
                    if msg.get('role') != 'system' or msg.get('type') != 'summary'
                ]
                
                # Include summary if present
                summary_messages = [
                    msg for msg in messages 
                    if msg.get('role') == 'system' and msg.get('type') == 'summary'
                ]
                
                context_parts = []
                
                # Add summary context if available
                if summary_messages:
                    latest_summary = summary_messages[-1]  # Get the most recent summary
                    context_parts.append(f"Context: {latest_summary.get('content', '')}")
                
                # Add recent conversation context (last 2 exchanges)
                if context_messages:
                    recent_context = context_messages[-2:]  # Last 2 non-summary messages
                    context_text = "\n".join([
                        f"Recent: {msg.get('content', str(msg))}" 
                        for msg in recent_context
                    ])
                    context_parts.append(context_text)
                
                if context_parts:
                    enhanced_task = f"{chr(10).join(context_parts)}\n\nCurrent request: {subtask_text}"
                else:
                    enhanced_task = subtask_text
            else:
                enhanced_task = subtask_text
            
            # Add multi-task context if we have multiple sub-tasks
            if len(sub_tasks) > 1:
                task_context = f"""
[MULTI-TASK CONTEXT] This is sub-task {current_index + 1} of {len(sub_tasks)} from a larger request.
Original full request: {state.get('task', '') if hasattr(state, 'get') else getattr(state, 'task', '')}
Current sub-task: {subtask_text}
"""
                enhanced_task = task_context + enhanced_task
            
            # Add timestamp context for time-aware agents (notion and weather)
            if selected_agent_name in ["notion_agent", "weather_agent"] and timestamp_context:
                timestamp_info = f"""
[SYSTEM CONTEXT] Current Singapore time information:
- Current date and time: {timestamp_context.get('singapore_time', 'Unknown')}
- Today's date: {timestamp_context.get('singapore_date', 'Unknown')}
- ISO datetime: {timestamp_context.get('singapore_datetime', 'Unknown')}

When using tools that require datetime parameters:
- For "today": use {timestamp_context.get('singapore_date', 'Unknown')}
- For current datetime: use {timestamp_context.get('singapore_datetime', 'Unknown')}
- Parse relative dates (tomorrow, yesterday) based on today's date: {timestamp_context.get('singapore_date', 'Unknown')}
"""
                enhanced_task = enhanced_task + timestamp_info
            
            response = selected_agent.process_task(enhanced_task)
            
            # Add metadata to the response
            response["agent_used"] = selected_agent_name
            response["sub_task_index"] = current_index
            response["sub_task"] = subtask_text
            
            return {"agent_responses": agent_responses + [response]}
            
        except Exception as e:
            error_response = {
                "success": False,
                "response": f"Agent execution failed: {str(e)}",
                "error": str(e),
                "agent": selected_agent_name,
                "sub_task_index": current_index,
                "sub_task": subtask_text
            }
            return {"agent_responses": agent_responses + [error_response]}
    
    def _check_remaining_tasks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if there are more sub-tasks to process and increment the index."""
        # Handle both dict and object state formats
        if hasattr(state, 'sub_tasks'):
            sub_tasks = state.sub_tasks
            current_index = state.current_task_index
        else:
            sub_tasks = state.get("sub_tasks", [])
            current_index = state.get("current_task_index", 0)
        
        # Increment to next task
        next_index = current_index + 1
        
        return {"current_task_index": next_index}
    
    def _should_continue_processing(self, state: Dict[str, Any]) -> str:
        """Determine if we should continue processing more sub-tasks or finish."""
        # Handle both dict and object state formats
        if hasattr(state, 'sub_tasks'):
            sub_tasks = state.sub_tasks
            current_index = state.current_task_index
        else:
            sub_tasks = state.get("sub_tasks", [])
            current_index = state.get("current_task_index", 0)
        
        # Check if there are more tasks to process
        if current_index < len(sub_tasks):
            return "continue"
        else:
            return "finish"
    
    def _aggregate_responses(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate responses from all agents into a coherent final response - optimized."""
        # Handle both dict and object state formats
        if hasattr(state, 'agent_responses'):
            agent_responses = state.agent_responses
            messages = state.messages
            sub_tasks = state.sub_tasks
        else:
            agent_responses = state.get("agent_responses", [])
            messages = state.get("messages", [])
            sub_tasks = state.get("sub_tasks", [])
        
        if not agent_responses:
            final_response = "No responses were generated."
        elif len(agent_responses) == 1:
            # Single response - use as-is (most common case)
            response = agent_responses[0]
            final_response = response.get("response", "No response generated")
        else:
            # Multiple responses - fast aggregation
            successful_responses = [r for r in agent_responses if r.get("success", False)]
            failed_responses = [r for r in agent_responses if not r.get("success", False)]
            
            if not successful_responses and failed_responses:
                # All failed
                error_messages = [f"• {r.get('sub_task', 'Task')}: {r.get('error', 'Failed')}" for r in failed_responses]
                final_response = f"I encountered issues with your requests:\n" + "\n".join(error_messages)
            elif successful_responses and not failed_responses:
                # All successful
                if len(successful_responses) == 1:
                    final_response = successful_responses[0].get("response", "")
                else:
                    # Multiple successful - format with simple headers
                    parts = []
                    for resp in successful_responses:
                        task = resp.get("sub_task", "").strip()
                        response_text = resp.get("response", "")
                        
                        # Simple formatting
                        if task and len(successful_responses) > 1:
                            # Capitalize first letter and add colon
                            header = task[0].upper() + task[1:] if len(task) > 1 else task.upper()
                            if not header.endswith(':'):
                                header += ':'
                            parts.append(f"**{header}**\n{response_text}")
                        else:
                            parts.append(response_text)
                    
                    final_response = "\n\n".join(parts)
            else:
                # Mixed success and failure
                parts = []
                
                # Add successful responses
                for resp in successful_responses:
                    task = resp.get("sub_task", "").strip()
                    response_text = resp.get("response", "")
                    if task:
                        header = task[0].upper() + task[1:] if len(task) > 1 else task.upper()
                        if not header.endswith(':'):
                            header += ':'
                        parts.append(f"**{header}**\n{response_text}")
                    else:
                        parts.append(response_text)
                
                # Add failure summary
                if failed_responses:
                    failed_tasks = [r.get('sub_task', 'Unknown task') for r in failed_responses]
                    parts.append(f"⚠️ Could not complete: {', '.join(failed_tasks)}")
                
                final_response = "\n\n".join(parts)
        
        # Update conversation history
        assistant_message = {"role": "assistant", "content": final_response}
        updated_messages = messages + [assistant_message]
        
        return {
            "final_response": final_response,
            "messages": updated_messages
        }
    
    def process_message(self, message: str, user_id: str = None, timestamp_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a message through the supervisor workflow with user-specific memory.
        
        Args:
            message: The user's input message
            user_id: Unique identifier for the user (for memory isolation)
            timestamp_context: Current timestamp information for time-aware agents
            
        Returns:
            Dict containing the supervisor's orchestrated response
        """
        try:
            # Use user-specific thread ID for memory isolation
            thread_id = f"user_{user_id}" if user_id else "default"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get current conversation state (if any) from memory
            try:
                current_state = self.workflow.get_state(config)
                existing_messages = current_state.values.get("messages", []) if current_state.values else []
            except:
                existing_messages = []
            
            # Create initial state with conversation history and memory management
            new_message = {"role": "user", "content": message}
            all_messages = existing_messages + [new_message]
            
            # Apply memory management (summarize if over 5 messages)
            managed_messages = self._manage_conversation_memory(all_messages, user_id or "default")
            
            initial_state = {
                "task": message,
                "messages": managed_messages,
                "sub_tasks": [],
                "selected_agent": None,
                "agent_response": None,
                "agent_responses": [],
                "final_response": "",
                "reasoning": "",
                "confidence_scores": {},
                "timestamp_context": timestamp_context or {},
                "current_task_index": 0
            }
            
            # Execute workflow with user-specific thread ID for memory isolation
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Extract results from final state - handle both dict and object formats
            if hasattr(final_state, 'final_response'):
                response = final_state.final_response
                sub_tasks = getattr(final_state, 'sub_tasks', [])
                agent_responses = getattr(final_state, 'agent_responses', [])
                confidence_scores = final_state.confidence_scores
                reasoning = final_state.reasoning
            else:
                response = final_state.get("final_response", "No response generated")
                sub_tasks = final_state.get("sub_tasks", [])
                agent_responses = final_state.get("agent_responses", [])
                confidence_scores = final_state.get("confidence_scores", {})
                reasoning = final_state.get("reasoning", "")
            
            # Determine which agents were used
            agents_used = []
            for agent_resp in agent_responses:
                agent_name = agent_resp.get("agent_used")
                if agent_name and agent_name not in agents_used:
                    agents_used.append(agent_name)
            
            return {
                "success": True,
                "response": response,
                "supervisor_metadata": {
                    "sub_tasks_count": len(sub_tasks),
                    "agents_used": agents_used,
                    "confidence_scores": confidence_scores,
                    "reasoning": reasoning,
                    "agent_responses": agent_responses,
                    "sub_tasks": sub_tasks
                },
                "workflow_type": "supervisor_multi_agent"
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Supervisor workflow failed: {str(e)}",
                "error": str(e),
                "workflow_type": "supervisor"
            }
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all managed agents."""
        capabilities = {}
        for agent_name, agent in self.agents.items():
            capabilities[agent_name] = agent.get_capabilities()
        return capabilities
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on supervisor and all agents."""
        try:
            api_key = get_api_key()
            agent_status = {}
            
            for agent_name, agent in self.agents.items():
                try:
                    # Simple capability check
                    capabilities = agent.get_capabilities()
                    agent_status[agent_name] = {
                        "status": "healthy",
                        "capabilities_count": len(capabilities)
                    }
                except Exception as e:
                    agent_status[agent_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            return {
                "status": "healthy",
                "supervisor_model": "gemini-2.5-flash",
                "api_key_configured": bool(api_key),
                "agents": agent_status,
                "workflow_compiled": self.workflow is not None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
