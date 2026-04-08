## Brings in all the necessary libraries from the ADK and Google Cloud. 
## It also sets up logging and loads the environment variables from your .env file, 
## which is crucial for accessing your model and server URL.
import os
import logging
import google.cloud.logging

from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import sqlite3

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL", "gemini-2.5-flash")
if model_name and "image" in model_name.lower():
    logging.warning(f"Warning: You have configured an image model ({model_name}) for a text-based conversational agent. Consider using a text model like 'gemini-2.5-flash' instead.")


## Define the tools:
# =========== Greet user and save their prompt ===========
def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's initial prompt to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}

# Configuring the Wikipedia Tool
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# ========================= Additional =========================

# ---------- Initialize SQLite Database (OK) ----------
def init_database():
    """Ensures the database and table exist by running the SQL file if needed."""
    db_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'zoo_database.db')
    sql_path = os.path.join(db_dir, 'zoo_database.sql')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='animals'")
    if not cursor.fetchone() and os.path.exists(sql_path):
        logging.info("Initializing database from zoo_database.sql...")
        with open(sql_path, 'r') as sql_file:
            conn.executescript(sql_file.read())
    conn.close()

init_database()

# ---------- End of SQLite Database initialization ----------


# ------------ Initilize access animals in zoo ------------
def animals_in_zoo(animal_names: list[str]) -> str:
    """Gets specific data about animals at our zoo including their names, ages, and locations."""
    try:
        # Connect to your real database. Here we use SQLite as an example.
        db_dir = os.path.join(os.path.dirname(__file__), 'data')
                
        db_path = os.path.join(db_dir, 'zoo_database.db')
        sql_path = os.path.join(db_dir, 'zoo_database.sql')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        output = []
        for animal_name in animal_names:
            # Query the database securely using parameterized queries to prevent SQL injection
            cursor.execute("SELECT name, age, location FROM animals WHERE species = ?", (animal_name.lower(),))
            results = cursor.fetchall()
            
            if results:
                output.append(f"Results for {animal_name.capitalize()}:")
                output.extend([f" - Name: {name}, Age: {age}, Location: {location}" for name, age, location in results])
            else:
                output.append(f"We don't currently have any {animal_name}s at our zoo.")
                
        conn.close()
        return "\n".join(output)
    except Exception as e:
        logging.error(f"Error accessing zoo database for {animal_names}: {e}")
        return "Sorry, I am currently unable to access the zoo database."

# ------------ End of access animals in zoo initialization ------------



# Provide Zoo's Show Time (OK)
def zoo_shows(show_type: str = "all") -> str:
    """Gets the schedule for animal shows at the zoo. show_type can be 'sea lion', 'birds', 'night safari', or 'all'."""
    schedules = {
        "sea lion": "Sea Lion Show: Daily at 11:00 AM and 2:00 PM at the Water Kingdom",
        "birds": "Birds Show: Daily at 10:00 AM and 3:00 PM at the Bird Palace",
        "night safari": "Night Safari: Fridays and Saturdays at 7:00 PM to 10:00 PM at the Southern Park"
    }
    query = show_type.lower()
    if query in schedules:
        return schedules[query]
    elif query == "all":
        return "\n".join(schedules.values())
    else:
        return f"Sorry, we don't have a schedule for a '{show_type}' show. We offer Sea Lion, Birds, and Night Safari shows."



## Define the Specialist agents: to define the comprehensive_researcher and response_formatter agents
# 1. Researcher Agent
comprehensive_researcher = Agent(
    name="comprehensive_researcher",
    model=model_name,
    description="The primary researcher that can access internal zoo data, show schedules, and external knowledge from Wikipedia.",
    instruction="""
    You are a helpful research assistant. Your goal is to fully answer the user's PROMPT.
    You have access to three tools:
    1. A tool for getting specific data about animals AT OUR ZOO (names, ages, locations).
    2. A tool for searching Wikipedia for general knowledge (facts, lifespan, diet, habitat).
    3. A tool for getting the schedules of animal shows at the zoo.

    First, analyze the user's PROMPT.
    - If the prompt can be answered by only one tool, use that tool.
    - If the prompt is complex and requires information from both the zoo's database AND Wikipedia,
      you MUST use both tools to gather all necessary information.
    - Synthesize the results from the tool(s) you use into preliminary data outputs.

    PROMPT:
    {PROMPT}
    """,
    tools=[
        animals_in_zoo, # Tool 1: Zoo Database
        wikipedia_tool, # Tool 3: Wikipedia
        zoo_shows       # Tool 4: Show Schedules
        
    ],
    output_key="research_data" # A key to store the combined findings
)

# 2. Response Formatter Agent
response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Synthesizes all information into a friendly, readable response.",
    instruction="""
    You are the friendly voice of the Zoo Tour Guide. Your task is to take the
    RESEARCH_DATA and present it to the user in a brief and helpful answer.

    - First, present the specific information from the zoo (like names, ages, and where to find them).
    - Next, provide the show schedule if the user asked for it.
    - Then, add the interesting general facts from the research.
    - If some information is missing, just present the information you have.
    - Be conversational and engaging.

    RESEARCH_DATA:
    {research_data}
    """
)


## Define the Workflow agent: to define the sequential agent tour_guide_workflow
tour_guide_workflow = SequentialAgent(
    name="tour_guide_workflow",
    description="The main workflow for handling a user's request about an animal.",
    sub_agents=[
        comprehensive_researcher, # Step 1: Gather all data
        response_formatter,       # Step 2: Format the final response
    ]
)


## Assemble the main workflow
root_agent = Agent(
    name="greeter",
    model=model_name,
    description="The main entry point for the Zoo Tour Guide.",
    instruction="""
    - Your name is Zoe.
    - Let the user know that you are the Zoo Tour Guide, ready to help them learn about the animals we have in the zoo.
    - Mention the name of the zoo, "Surabaya Zoo".
    - When the user responds, use the 'add_prompt_to_state' tool to save their response.
    After using the tool, transfer control to the 'tour_guide_workflow' agent.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[tour_guide_workflow]
)


# ====================================================================
# Connection Check Utility
# ====================================================================
def check_bq_connection():
    """A simple utility to test the BigQuery connection and table access."""
    try:
        client = bigquery.Client(project=BILLING_PROJECT)
        table = client.get_table(BQ_TABLE)
        print(f"✅ Successfully connected to BigQuery!")
        print(f"✅ Table {table.project}.{table.dataset_id}.{table.table_id} exists with {table.num_rows} rows.")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to BigQuery or access the table: {e}")
        return False

if __name__ == "__main__":
    check_bq_connection()