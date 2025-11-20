import logging
from logging.handlers import TimedRotatingFileHandler
from flask import Flask, render_template, request, jsonify, send_file
import psycopg2
import csv
import requests
import io
import sqlparse
import google.generativeai as genai
from pglast import parse_sql
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, PG_SCHEMA, gemini_api_key
from datetime import datetime
from psycopg2 import pool
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import threading
import json

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler - rotate logs at midnight and keep 30 backups
file_handler = TimedRotatingFileHandler('app.log', when='midnight', backupCount=30)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["1000 per hour"],
    storage_uri="memory://"
)

DB_CONFIG = {
    "host": PG_HOST,
    "port": PG_PORT,
    "dbname": PG_DB,
    "user": PG_USER,
    "password": PG_PASSWORD
}

DEFAULT_SCHEMA = PG_SCHEMA

# Postgres connection pool for scalability
try:
    pg_pool = psycopg2.pool.ThreadedConnectionPool(
        1, 20,
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )
    logging.info("PostgreSQL connection pool created successfully.")
except Exception as e:
    logging.error("Failed to create PostgreSQL connection pool: %s", e)
    raise

# try:
#     LLM_API_URL = "http://vllm-server:8000/v1/chat/completions"
#     LLM_MODEL = "vicuna-7B-1.1-HF"
# except Exception as e:
#     logging.error("Failed to configure API: %s", e)
#     raise

try:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.error("Failed to configure Gemini API: %s", e)
    raise

def create_feedback_table():
    """Create feedback table if it doesn't exist"""
    conn = None
    try:
        conn = pg_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {DEFAULT_SCHEMA}.user_feedback (
                    id SERIAL PRIMARY KEY,
                    feedback_id VARCHAR(100) NOT NULL,
                    user_question TEXT,
                    sql_query TEXT,
                    rating VARCHAR(20),
                    feedback_text TEXT,
                    feedback_options TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR(100),
                    message_id VARCHAR(100)
                );
                
                CREATE INDEX IF NOT EXISTS idx_feedback_id ON {DEFAULT_SCHEMA}.user_feedback(feedback_id);
                CREATE INDEX IF NOT EXISTS idx_created_at ON {DEFAULT_SCHEMA}.user_feedback(created_at);
                CREATE INDEX IF NOT EXISTS idx_message_id ON {DEFAULT_SCHEMA}.user_feedback(message_id);
            """)
            conn.commit()
            logging.info("Feedback table created/verified successfully.")
    except Exception as e:
        logging.error(f"Error creating feedback table: {e}")
    finally:
        if conn:
            pg_pool.putconn(conn)

# Create feedback table on startup
create_feedback_table()

def get_all_tables_overview():
    """
    Lightweight DB overview — only schema, tables, columns & types.
    Prevents context overflow in LLM prompt.
    """
    logging.info("Building lightweight DB schema overview…")

    overview = []
    conn = None
    try:
        conn = pg_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                AND table_type='BASE TABLE'
                ORDER BY table_schema, table_name;
            """)
            tables = cur.fetchall()

            for schema, table in tables:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema=%s AND table_name=%s
                    ORDER BY ordinal_position
                """, (schema, table))
                cols = cur.fetchall()

                col_desc = ", ".join([f"{c}:{t}" for c, t in cols])
                overview.append(f"{schema}.{table} → {col_desc}")

        return "\n".join(overview)

    except Exception as e:
        logging.error(f"[DB_OVERVIEW_ERROR] Could not fetch schema overview: {e}")
        return ""
    finally:
        if conn:
            pg_pool.putconn(conn)

def is_safe_sql(sql: str) -> bool:
    """
    Validate SQL safety - only allow SELECT queries.
    """
    logging.debug("Validating SQL safety.")
    sql_l = sql.lower().strip()
    
    # Remove comments and extra whitespace
    sql_l = ' '.join(sql_l.split())
    
    # Must start with SELECT
    if not sql_l.startswith('select'):
        logging.warning("Blocked SQL: does not start with SELECT")
        return False
    
    # Block dangerous keywords anywhere in query
    forbidden_keywords = [
        "insert", "update", "delete", "drop", "alter", 
        "truncate", "create", "grant", "revoke", "exec"
    ]
    
    for keyword in forbidden_keywords:
        if f" {keyword} " in f" {sql_l} ":
            logging.warning(f"Blocked unsafe SQL due to forbidden keyword: {keyword}")
            return False
    
    # Validate syntax
    try:
        parse_sql(sql)
    except Exception as e:
        logging.warning(f"SQL syntax validation failed: {e}")
        return False
    
    return True

def run_query(sql: str, params=None):
    """Execute SQL query and return results."""
    logging.debug("Running SQL query")

    conn = None
    try:
        conn = pg_pool.getconn()
        logging.info(f"🟦 Executing SQL: {sql}")
        with conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            try:
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                logging.debug(f"Query returned {len(rows)} rows.")
                return rows, cols
            except Exception:
                logging.debug("Query executed but no rows returned.")
                return [], []
    except Exception as e:
        logging.error(f"Query error: {e}")
        logging.error(f"❌ SQL Execution Error: {e}")
        return [], []
    finally:
        if conn:
            pg_pool.putconn(conn)

@limiter.limit("10/minute")
def ask_gemini(prompt: str) -> str:
    """Call Gemini AI with prompt."""
    logging.debug("Sending prompt to Gemini AI.")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return ""

def generate_smart_suggestions(question: str, rows, cols, db_overview: str):
    """
    Generate contextual follow-up suggestions based on the current query results.
    """
    suggestions = []
    
    # Analyze data characteristics
    has_data = len(rows) > 0
    num_cols = len(cols) if cols else 0
    num_rows = len(rows)
    
    if not has_data:
        return ["Show me all available tables", "What data can you help me analyze?"]
    
    # Detect numeric columns
    numeric_cols = []
    date_cols = []
    text_cols = []
    
    if rows and cols:
        for idx, col in enumerate(cols):
            sample_values = [row[idx] for row in rows[:5] if row[idx] is not None]
            if sample_values:
                sample = sample_values[0]
                if isinstance(sample, (int, float)):
                    numeric_cols.append(col)
                elif isinstance(sample, str):
                    if any(date_word in col.lower() for date_word in ['date', 'time', 'year', 'month']):
                        date_cols.append(col)
                    else:
                        text_cols.append(col)
    
    # Always offer table/chart options first
    if num_rows > 0:
        suggestions.append("📊 Show this data as a table")
        if numeric_cols:
            suggestions.append("📈 Visualize this data")
    
    # Add drill-down suggestions
    if numeric_cols:
        suggestions.append(f"What are the top 10 records by {numeric_cols[0]}?")
        if len(numeric_cols) > 1:
            suggestions.append(f"Compare {numeric_cols[0]} vs {numeric_cols[1]}")
    
    if date_cols and num_rows > 1:
        suggestions.append(f"Show trends over {date_cols[0]}")
    
    if text_cols:
        suggestions.append(f"Group this data by {text_cols[0]}")
    
    # Add aggregation suggestions
    if numeric_cols:
        suggestions.append(f"What is the total {numeric_cols[0]}?")
        suggestions.append(f"Show average {numeric_cols[0]}")
    
    # Add exploration suggestions using AI
    try:
        ai_prompt = f"""Based on this query result, suggest 2 insightful follow-up questions:
Question: {question}
Columns: {', '.join(cols[:10])}
Row count: {num_rows}

Return ONLY 2 short questions (max 10 words each), one per line, no numbering."""
        
        ai_suggestions = ask_gemini(ai_prompt)
        if ai_suggestions:
            ai_lines = [s.strip() for s in ai_suggestions.split('\n') if s.strip()]
            suggestions.extend(ai_lines[:2])
    except:
        pass
    
    # Return unique suggestions, limit to 8
    unique_suggestions = []
    seen = set()
    for s in suggestions:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique_suggestions.append(s)
            if len(unique_suggestions) >= 8:
                break
    
    return unique_suggestions

def detect_visualization_intent(question: str):
    """Detect if user wants a table or visualization"""
    q_lower = question.lower()
    
    viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'show me visually']
    table_keywords = ['table', 'tabular', 'show data', 'display data', 'list']
    
    wants_viz = any(kw in q_lower for kw in viz_keywords)
    wants_table = any(kw in q_lower for kw in table_keywords)
    
    return {
        'wants_visualization': wants_viz,
        'wants_table': wants_table,
        'auto_show': wants_viz or wants_table
    }

@app.route("/")
def index():
    return render_template("index.html", default_schema=DEFAULT_SCHEMA)

@app.route("/chat", methods=["POST"])
@limiter.limit("20 per minute")
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    conversation_history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Please provide a message"}), 400

    # -----------------------------
    # 1. GREETING HANDLER
    # -----------------------------
    if user_message.lower() in ["hi", "hello", "hii"]:
        assistant_msg = "Hi there! How can I help you analyze your data today?"
        updated_history = conversation_history + [
            {"role": "assistant", "content": assistant_msg}
        ]
        return jsonify({
            "type": "text",
            "message_id": f"msg_{datetime.now().timestamp()}",
            "message": assistant_msg,
            "summary": assistant_msg,
            "suggestions": [
                "What tables are in the database?",
                "Show me sample data",
                "Help me explore my data"
            ],
            "sql": None,
            "data": None,
            "history": updated_history
        })

    # -----------------------------
    # 2. FETCH COMPLETE DB OVERVIEW
    # -----------------------------
    db_overview = get_all_tables_overview()

    # -----------------------------
    # 3. BUILD AI PROMPT FOR SQL
    # -----------------------------
    sql_prompt = f"""
You are Solis AI, an expert PostgreSQL query generator.
You MUST behave as if you **have full access** to the database described below.

IMPORTANT — NEVER say:
"As an AI", "I don’t have access", "I was trained on", 
or any statement about your training data or limitations.

Use ONLY the database schema provided below.
Never claim you don't have database access.

DATABASE SCHEMA (summaries only):
{db_overview}

USER QUESTION:
{user_message}

STRICT OUTPUT RULES:
1. If user asks conversational / greetings → return EXACTLY: NO_QUERY
2. If user requests table/chart/visualization → return EXACTLY: USE_LAST_DATA
3. Otherwise output:
      • A **single-line** PostgreSQL SELECT query ONLY
      • No backticks, no commentary, no markdown
4. MUST use schema.table format
5. MUST avoid all modifying queries:
   No INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE
6. Add LIMIT 100 if query may return many rows

OUTPUT ONLY ONE OF:
- NO_QUERY
- USE_LAST_DATA
- A single SELECT SQL query
"""


    sql_or_cmd = ask_gemini(sql_prompt)
    if not sql_or_cmd:
        return jsonify({"error": "AI failed to generate SQL"}), 500

# Keep only lines that start with SELECT (case-insensitive)
    for line in sql_or_cmd.splitlines():
        clean = line.strip()
        if clean.lower().startswith("select"):
            sql_or_cmd = clean
            break

    # -----------------------------
    # 4. HANDLE NON-SQL COMMANDS
    # -----------------------------

    # CASE A: Conversation
    if "NO_QUERY" in sql_or_cmd:
        llm_response = ask_gemini(
            f"You are Solis AI. Reply friendly (max 50 words) to: {user_message}"
        )
        updated_history = conversation_history + [
            {"role": "assistant", "content": llm_response}
        ]
        return jsonify({
            "type": "text",
            "message": llm_response,
            "message_id": f"msg_{datetime.now().timestamp()}",
            "summary": llm_response,
            "suggestions": [
                "What data can you show me?",
                "What tables exist?",
                "Show me a summary of my data"
            ],
            "sql": None,
            "data": None,
            "history": updated_history
        })

    # CASE B: User asked "Show table" or "visualize"
    if "USE_LAST_DATA" in sql_or_cmd:
        return jsonify({
            "type": "info",
            "message_id": f"msg_{datetime.now().timestamp()}",
            "summary": "Please ask a data question first before I can visualize or display results.",
            "suggestions": [
                "Show me available tables",
                "Give me sample data",
                "Help me explore my data"
            ],
            "sql": None,
            "data": None,
            "history": conversation_history
        })

    # -----------------------------
    # 5. CLEAN + VALIDATE SQL
    # -----------------------------
    sql = sql_or_cmd.replace("```", "").replace("sql", "").strip()
    sql = sqlparse.format(sql, strip_comments=True).strip()

    if not is_safe_sql(sql):
        return jsonify({"error": "Generated SQL failed safety validation"}), 400

    # -----------------------------
    # 6. RUN QUERY
    # -----------------------------
    rows, cols = run_query(sql)

    # No results?
    if not rows:
        updated_history = conversation_history + [
            {"role": "assistant", "content": "No data found."}
        ]
        return jsonify({
            "type": "no_data",
            "message_id": f"msg_{datetime.now().timestamp()}",
            "summary": "I executed your query but found no data.",
            "sql": sql,
            "data": None,
            "suggestions": [
                "Show me all tables",
                "Show sample data",
                "Help me explore"
            ],
            "history": updated_history
        })

    # -----------------------------
    # 7. GENERATE SUMMARY
    # -----------------------------
    summary_prompt = f"""
Summarize these PostgreSQL results in 2–3 sentences.

USER QUESTION:
{user_message}

SQL:
{sql}

COLUMNS:
{cols}

FIRST ROWS:
{rows[:3]}
"""
    summary = ask_gemini(summary_prompt)

    # -----------------------------
    # 8. SMART FOLLOW-UP SUGGESTIONS
    # -----------------------------
    suggestions = generate_smart_suggestions(user_message, rows, cols, db_overview)

    # -----------------------------
    # 9. FINAL RESPONSE
    # -----------------------------
    updated_history = conversation_history + [
        {"role": "assistant", "content": summary}
    ]

    return jsonify({
        "type": "data",
        "message_id": f"msg_{datetime.now().timestamp()}",
        "summary": summary,
        "message": summary,
        "sql": sql,
        "suggestions": suggestions,
        "data": {
            "columns": cols,
            "rows": rows[:1000],
            "total_rows": len(rows)
        },
        "history": updated_history
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    """Store user feedback."""
    data = request.json
    message_id = data.get("message_id", "")
    rating = data.get("rating", "").strip()
    feedback_text = data.get("feedback_text", "").strip()
    user_question = data.get("user_question", "")
    sql_query = data.get("sql_query", "")
    session_id = data.get("session_id", "")
    
    if not message_id:
        return jsonify({"error": "Invalid message id"}), 400
    
    if not rating and not feedback_text:
        return jsonify({"error": "Empty feedback"}), 400

    conn = None
    try:
        conn = pg_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {DEFAULT_SCHEMA}.user_feedback 
                (feedback_id, message_id, user_question, sql_query, rating, feedback_text, session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (message_id, message_id, user_question, sql_query, rating, feedback_text, session_id))
            conn.commit()
            logging.info(f"Feedback stored for message {message_id}")
    except Exception as e:
        logging.error(f"Error storing feedback: {e}")
        return jsonify({"error": "Failed to store feedback"}), 500
    finally:
        if conn:
            pg_pool.putconn(conn)

    return jsonify({"message": "Thank you for your feedback!"})

@app.route("/download_csv", methods=["POST"])
def download_csv():
    """Download query results as CSV."""
    data = request.json
    cols = data.get("columns", [])
    rows = data.get("rows", [])

    if not cols or not rows:
        return jsonify({"error": "No data to download"}), 400

    csv_io = generate_csv(rows, cols)
    csv_io.seek(0)

    return send_file(
        io.BytesIO(csv_io.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'solis_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

@app.route("/get_initial_suggestions", methods=["GET"])
def get_initial_suggestions():
    """Get initial conversation starter suggestions"""
    try:
        db_overview = get_all_tables_overview()
        
        # Extract table names
        import re
        table_matches = re.findall(r'Table: (\w+)', db_overview)
        tables = list(set(table_matches))[:5]  # Get up to 5 unique tables
        
        suggestions = [
            "What data is available in this database?",
            "Show me a summary of all tables"
        ]
        
        if tables:
            suggestions.append(f"What's in the {tables[0]} table?")
            if len(tables) > 1:
                suggestions.append(f"Compare data from {tables[0]} and {tables[1]}")
        
        suggestions.extend([
            "Help me explore my data",
            "What insights can you show me?"
        ])
        
        return jsonify({"suggestions": suggestions[:6]})
    except Exception as e:
        logging.error(f"Error getting initial suggestions: {e}")
        return jsonify({"suggestions": [
            "What data is available?",
            "Show me all tables",
            "Help me get started"
        ]}), 200

@app.route("/visualize", methods=["POST"])
def visualize():
    """Generate visualization data based on chart type"""
    data = request.json
    chart_type = data.get("chart_type", "bar")
    rows = data.get("rows", [])
    cols = data.get("cols", [])
    
    if not rows or not cols:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        # Detect column types
        numeric_indices = []
        text_indices = []
        
        if len(rows) > 0:
            for idx, col in enumerate(cols):
                sample_val = rows[0][idx]
                if isinstance(sample_val, (int, float)):
                    numeric_indices.append(idx)
                else:
                    text_indices.append(idx)
        
        # Default to first text column for labels, first numeric for values
        label_idx = text_indices[0] if text_indices else 0
        value_idx = numeric_indices[0] if numeric_indices else (1 if len(cols) > 1 else 0)
        
        # Prepare data for Chart.js
        labels = [str(row[label_idx]) for row in rows[:50]]
        values = []
        
        for row in rows[:50]:
            try:
                val = float(row[value_idx]) if row[value_idx] is not None else 0
                values.append(val)
            except:
                values.append(0)
        
        chart_data = {
            "labels": labels,
            "datasets": [{
                "label": cols[value_idx],
                "data": values
            }],
            "chart_type": chart_type,
            "title": f"{cols[value_idx]} by {cols[label_idx]}"
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return jsonify({"error": "Failed to generate visualization"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)