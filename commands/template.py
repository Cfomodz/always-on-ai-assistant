import typer
from typing import Optional
import sqlite3
import os
import json
import csv
import difflib
import random
import string
import shutil
from datetime import datetime

import yaml

app = typer.Typer()

# -----------------------------------------------------
# Database helpers: create/connect/seed
# -----------------------------------------------------
DB_NAME = "app_data.db"


def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_NAME)


def create_db_if_not_exists():
    """Create tables if they do not exist and seed them with mock data."""
    conn = get_connection()
    cur = conn.cursor()

    # Create a sample 'users' table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    )

    # Create a sample 'tasks' table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name TEXT NOT NULL,
        priority INTEGER NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    )

    # Create a sample 'logs' table
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        level TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    )

    # Check if 'users' table has data; if not, seed 25 rows
    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]
    if user_count == 0:
        roles = ["guest", "admin", "editor", "viewer"]
        for i in range(25):
            username = f"user_{i}"
            role = random.choice(roles)
            created_at = datetime.now().isoformat()
            cur.execute(
                "INSERT INTO users (username, role, created_at) VALUES (?, ?, ?)",
                (username, role, created_at),
            )

    # Seed 'tasks' table
    cur.execute("SELECT COUNT(*) FROM tasks")
    task_count = cur.fetchone()[0]
    if task_count == 0:
        statuses = ["pending", "in-progress", "complete"]
        for i in range(25):
            task_name = f"task_{i}"
            priority = random.randint(1, 5)
            status = random.choice(statuses)
            created_at = datetime.now().isoformat()
            cur.execute(
                "INSERT INTO tasks (task_name, priority, status, created_at) VALUES (?, ?, ?, ?)",
                (task_name, priority, status, created_at),
            )

    # Seed 'logs' table
    cur.execute("SELECT COUNT(*) FROM logs")
    logs_count = cur.fetchone()[0]
    if logs_count == 0:
        levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        for i in range(25):
            message = f"Log entry number {i}"
            level = random.choice(levels)
            created_at = datetime.now().isoformat()
            cur.execute(
                "INSERT INTO logs (message, level, created_at) VALUES (?, ?, ?)",
                (message, level, created_at),
            )

    conn.commit()
    conn.close()


# Ensure the database and tables exist before we do anything
create_db_if_not_exists()


# -----------------------------------------------------
# Simple Caesar cipher for “encryption/decryption” demo
# -----------------------------------------------------
def caesar_cipher_encrypt(plaintext: str, shift: int = 3) -> str:
    """A simple Caesar cipher encryption function."""
    result = []
    for ch in plaintext:
        if ch.isalpha():
            start = ord("A") if ch.isupper() else ord("a")
            offset = (ord(ch) - start + shift) % 26
            result.append(chr(start + offset))
        else:
            result.append(ch)
    return "".join(result)


def caesar_cipher_decrypt(ciphertext: str, shift: int = 3) -> str:
    """A simple Caesar cipher decryption function."""
    return caesar_cipher_encrypt(ciphertext, -shift)


# -----------------------------------------------------
# 1) ping_server
# -----------------------------------------------------
@app.command()
def ping_server(
    wait: bool = typer.Option(False, "--wait", help="Wait for server response?")
):
    """
    Pings the server, optionally waiting for a response.
    """
    # Mock a server response time
    response_time_ms = random.randint(50, 300)
    result = f"Server pinged. Response time: {response_time_ms} ms."
    if wait:
        result += " (Waited for a response.)"
    typer.echo(result)
    return result


# -----------------------------------------------------
# 2) show_config
# -----------------------------------------------------
@app.command()
def show_config(
    verbose: bool = typer.Option(False, "--verbose", help="Show config in detail?")
):
    """
    Shows the current configuration from modules/assistant_config.py.
    """
    try:

        config = ""

        with open("./assistant_config.yml", "r") as f:
            config = f.read()

        if verbose:
            result = f"Verbose config:\n{json.dumps(yaml.safe_load(config), indent=2)}"
        else:
            result = f"Config: {config}"
        typer.echo(result)
        return result
    except ImportError:
        result = "Error: Could not load assistant_config module"
        typer.echo(result)
        return result


# -----------------------------------------------------
# 3) list_files
# -----------------------------------------------------
@app.command()
def list_files(
    path: str = typer.Argument(..., help="Path to list files from"),
    all_files: bool = typer.Option(False, "--all", help="Include hidden files"),
):
    """
    Lists files in a directory. Optionally show hidden files.
    """
    if not os.path.isdir(path):
        msg = f"Path '{path}' is not a valid directory."
        typer.echo(msg)
        return msg

    entries = os.listdir(path)
    if not all_files:
        entries = [e for e in entries if not e.startswith(".")]

    result = f"Files in '{path}': {entries}"
    typer.echo(result)
    return result


# -----------------------------------------------------
# 3.5) list_users
# -----------------------------------------------------
@app.command()
def list_users(
    role: str = typer.Option(None, "--role", help="Filter users by role"),
    sort: str = typer.Option(
        "username", "--sort", help="Sort by field (username, role, created_at)"
    ),
):
    """
    Lists all users, optionally filtered by role and sorted by specified field.
    """
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT username, role, created_at FROM users"
    params = []

    if role:
        query += " WHERE role = ?"
        params.append(role)

    query += f" ORDER BY {sort}"

    cur.execute(query, params)
    users = cur.fetchall()
    conn.close()

    if not users:
        result = "No users found."
        typer.echo(result)
        return result

    # Format output
    result = "Users:\n"
    for user in users:
        result += f"- {user[0]} (Role: {user[1]}, Created: {user[2]})\n"

    typer.echo(result)
    return result


# -----------------------------------------------------
# 4) create_user
# -----------------------------------------------------
@app.command()
def create_user(
    username: str = typer.Argument(..., help="Name of the new user"),
    role: str = typer.Option("guest", "--role", help="Role for the new user"),
):
    """
    Creates a new user with an optional role.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO users (username, role, created_at) VALUES (?, ?, ?)",
        (username, role, now),
    )
    conn.commit()
    conn.close()
    result = f"User '{username}' created with role '{role}'."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 5) delete_user
# -----------------------------------------------------
@app.command()
def delete_user(
    user_id: str = typer.Argument(..., help="ID of user to delete"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompt"),
):
    """
    Deletes a user by ID.
    """
    if not confirm:
        # In a real scenario, you'd prompt or handle differently
        typer.echo(f"Confirmation needed to delete user {user_id}. Use --confirm.")
        return f"Deletion of user {user_id} not confirmed."

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    changes = cur.rowcount
    conn.close()

    if changes > 0:
        msg = f"User with ID {user_id} deleted."
    else:
        msg = f"No user found with ID {user_id}."
    typer.echo(msg)
    return msg


# -----------------------------------------------------
# 6) generate_report
# -----------------------------------------------------
@app.command()
def generate_report(
    table_name: str = typer.Argument(..., help="Name of table to generate report from"),
    output_file: str = typer.Option("report.json", "--output", help="Output file name"),
):
    """
    Generates a report from an existing database table and saves it to a file.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Get all data from the specified table
    cur.execute(f"SELECT * FROM {table_name}")
    rows = cur.fetchall()

    # Get column names from cursor description
    columns = [description[0] for description in cur.description]

    # Convert rows to list of dicts with column names
    data = []
    for row in rows:
        data.append(dict(zip(columns, row)))

    report_data = {
        "table": table_name,
        "timestamp": datetime.now().isoformat(),
        "columns": columns,
        "row_count": len(rows),
        "data": data,
    }

    with open(output_file, "w") as f:
        json.dump(report_data, f, indent=2)

    conn.close()

    result = f"Report for table '{table_name}' generated and saved to {output_file}."
    typer.echo(result)
    typer.echo(json.dumps(report_data, indent=2))
    return report_data


# -----------------------------------------------------
# 7) backup_data
# -----------------------------------------------------
@app.command()
def backup_data(
    directory: str = typer.Argument(..., help="Directory to store backups"),
    full: bool = typer.Option(False, "--full", help="Perform a full backup"),
):
    """
    Back up data to a specified directory, optionally performing a full backup.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)

    backup_file = os.path.join(
        directory, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    )
    shutil.copy(DB_NAME, backup_file)

    result = (
        f"{'Full' if full else 'Partial'} backup completed. Saved to {backup_file}."
    )
    typer.echo(result)
    return result


# -----------------------------------------------------
# 8) restore_data
# -----------------------------------------------------
@app.command()
def restore_data(
    file_path: str = typer.Argument(..., help="File path of backup to restore"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing data"
    ),
):
    """
    Restores data from a backup file.
    """
    if not os.path.isfile(file_path):
        msg = f"Backup file {file_path} does not exist."
        typer.echo(msg)
        return msg

    if not overwrite:
        msg = "Overwrite not confirmed. Use --overwrite to proceed."
        typer.echo(msg)
        return msg

    shutil.copy(file_path, DB_NAME)
    msg = f"Data restored from {file_path} to {DB_NAME}."
    typer.echo(msg)
    return msg


# -----------------------------------------------------
# 9) summarize_logs
# -----------------------------------------------------
@app.command()
def summarize_logs(
    logs_path: str = typer.Argument(..., help="Path to log files"),
    lines: int = typer.Option(100, "--lines", help="Number of lines to summarize"),
):
    """
    Summarizes log data from a specified path, limiting lines.
    """
    if not os.path.isfile(logs_path):
        msg = f"Log file {logs_path} not found."
        typer.echo(msg)
        return msg

    with open(logs_path, "r") as f:
        all_lines = f.readlines()

    snippet = all_lines[:lines]
    result = f"Showing first {lines} lines from {logs_path}:\n" + "".join(snippet)
    typer.echo(result)
    return result


# -----------------------------------------------------
# 10) upload_file
# -----------------------------------------------------
@app.command()
def upload_file(
    file_path: str = typer.Argument(..., help="Path of file to upload"),
    destination: str = typer.Option(
        "remote", "--destination", help="Destination label"
    ),
    secure: bool = typer.Option(True, "--secure", help="Use secure upload"),
):
    """
    Uploads a file to a destination, optionally enforcing secure upload.
    """
    if not os.path.isfile(file_path):
        msg = f"File {file_path} not found."
        typer.echo(msg)
        return msg

    # Mock upload
    result = f"File '{file_path}' uploaded to '{destination}' using {'secure' if secure else 'insecure'} mode."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 11) download_file
# -----------------------------------------------------
@app.command()
def download_file(
    url: str = typer.Argument(..., help="URL of file to download"),
    output_path: str = typer.Option(".", "--output", help="Local output path"),
    retry: int = typer.Option(3, "--retry", help="Number of times to retry"),
):
    """
    Downloads a file from a URL with a specified number of retries.
    """
    # In real scenario, you'd do requests, etc. We'll just mock it.
    filename = os.path.join(output_path, os.path.basename(url))
    with open(filename, "w") as f:
        f.write("Downloaded data from " + url)

    result = f"File downloaded from {url} to {filename} with {retry} retries allowed."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 12) filter_records
# -----------------------------------------------------
@app.command()
def filter_records(
    source: str = typer.Argument(..., help="Data source to filter"),
    query: str = typer.Option("", "--query", help="Filtering query string"),
    limit: int = typer.Option(10, "--limit", help="Limit the number of results"),
):
    """
    Filters records from a data source using a query, limiting the number of results.
    Example usage: filter_records table_name --query "admin" --limit 5
    """
    conn = get_connection()
    cur = conn.cursor()

    # For demonstration, we'll assume the 'source' is a table name in the DB
    # and the 'query' is a substring to match against username or message, etc.
    # This is just a simple example.
    try:
        sql = f"SELECT * FROM {source} WHERE "
        if source == "users":
            sql += "username LIKE ?"
        elif source == "logs":
            sql += "message LIKE ?"
        elif source == "tasks":
            sql += "task_name LIKE ?"
        else:
            typer.echo(f"Unknown table: {source}")
            return f"Table '{source}' not recognized."

        sql += f" LIMIT {limit}"

        wildcard_query = f"%{query}%"
        cur.execute(sql, (wildcard_query,))
        rows = cur.fetchall()

        result = (
            f"Found {len(rows)} records in '{source}' with query '{query}'.\n{rows}"
        )
        typer.echo(result)
        return result

    except sqlite3.OperationalError as e:
        msg = f"SQL error: {e}"
        typer.echo(msg)
        return msg
    finally:
        conn.close()


# -----------------------------------------------------
# 16) compare_files
# -----------------------------------------------------
@app.command()
def compare_files(
    file_a: str = typer.Argument(..., help="First file to compare"),
    file_b: str = typer.Argument(..., help="Second file to compare"),
    diff_only: bool = typer.Option(
        False, "--diff-only", help="Show only the differences"
    ),
):
    """
    Compares two files, optionally showing only differences.
    """
    if not os.path.isfile(file_a) or not os.path.isfile(file_b):
        msg = f"One or both files do not exist: {file_a}, {file_b}"
        typer.echo(msg)
        return msg

    with open(file_a, "r") as fa, open(file_b, "r") as fb:
        lines_a = fa.readlines()
        lines_b = fb.readlines()

    diff = difflib.unified_diff(lines_a, lines_b, fromfile=file_a, tofile=file_b)

    if diff_only:
        # Show only differences
        differences = []
        for line in diff:
            if line.startswith("+") or line.startswith("-"):
                differences.append(line)
        result = "\n".join(differences)
    else:
        # Show entire unified diff
        result = "".join(diff)

    typer.echo(result if result.strip() else "Files are identical.")
    return result


# -----------------------------------------------------
# 17) encrypt_data
# -----------------------------------------------------
@app.command()
def encrypt_data(
    input_path: str = typer.Argument(..., help="Path of the file to encrypt"),
    output_path: str = typer.Option("encrypted.bin", "--output", help="Output file"),
    algorithm: str = typer.Option("AES", "--algorithm", help="Encryption algorithm"),
):
    """
    Encrypts data using a specified algorithm (mocked by Caesar cipher here).
    """
    if not os.path.isfile(input_path):
        msg = f"File {input_path} not found."
        typer.echo(msg)
        return msg

    with open(input_path, "r") as f:
        data = f.read()

    # We'll just mock the encryption using Caesar cipher
    encrypted = caesar_cipher_encrypt(data, 3)

    with open(output_path, "w") as f:
        f.write(encrypted)

    result = f"Data from {input_path} encrypted with {algorithm} (mock) and saved to {output_path}."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 18) decrypt_data
# -----------------------------------------------------
@app.command()
def decrypt_data(
    encrypted_file: str = typer.Argument(..., help="Path to encrypted file"),
    key: str = typer.Option(..., "--key", help="Decryption key"),
    output_path: str = typer.Option("decrypted.txt", "--output", help="Output file"),
):
    """
    Decrypts an encrypted file using a key (ignored in this mock Caesar cipher).
    """
    if not os.path.isfile(encrypted_file):
        msg = f"Encrypted file {encrypted_file} not found."
        typer.echo(msg)
        return msg

    with open(encrypted_file, "r") as f:
        encrypted_data = f.read()

    # Key is ignored in this Caesar cipher demo
    decrypted = caesar_cipher_decrypt(encrypted_data, 3)

    with open(output_path, "w") as f:
        f.write(decrypted)

    result = f"Data from {encrypted_file} decrypted and saved to {output_path}."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 21) migrate_database
# -----------------------------------------------------
@app.command()
def migrate_database(
    old_db: str = typer.Argument(..., help="Path to old database"),
    new_db: str = typer.Option(..., "--new-db", help="Path to new database"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Perform a trial run without changing data"
    ),
):
    """
    Migrates data from an old database to a new one, optionally doing a dry run.
    """
    if not os.path.isfile(old_db):
        msg = f"Old database '{old_db}' not found."
        typer.echo(msg)
        return msg

    if dry_run:
        result = f"Dry run: would migrate {old_db} to {new_db}."
        typer.echo(result)
        return result

    shutil.copy(old_db, new_db)
    result = f"Database migrated from {old_db} to {new_db}."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 27) queue_task
# -----------------------------------------------------
@app.command()
def queue_task(
    task_name: str = typer.Argument(..., help="Name of the task to queue"),
    priority: int = typer.Option(1, "--priority", help="Priority of the task"),
    delay: int = typer.Option(
        0, "--delay", help="Delay in seconds before starting task"
    ),
):
    """
    Queues a task with a specified priority and optional delay.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO tasks (task_name, priority, status, created_at) VALUES (?, ?, ?, ?)",
        (task_name, priority, "pending", now),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()

    result = f"Task '{task_name}' queued with priority {priority}, delay {delay}s, assigned ID {task_id}."
    typer.echo(result)
    return result


# -----------------------------------------------------
# 28) remove_task
# -----------------------------------------------------
@app.command()
def remove_task(
    task_id: str = typer.Argument(..., help="ID of the task to remove"),
    force: bool = typer.Option(False, "--force", help="Remove without confirmation"),
):
    """
    Removes a queued task by ID, optionally forcing removal without confirmation.
    """
    if not force:
        msg = f"Confirmation required to remove task {task_id}. Use --force."
        typer.echo(msg)
        return msg

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    removed = cur.rowcount
    conn.close()

    if removed:
        msg = f"Task {task_id} removed."
    else:
        msg = f"Task {task_id} not found."
    typer.echo(msg)
    return msg


# -----------------------------------------------------
# 29) list_tasks
# -----------------------------------------------------
@app.command()
def list_tasks(
    show_all: bool = typer.Option(
        False, "--all", help="Show all tasks, including completed"
    ),
    sort_by: str = typer.Option(
        "priority", "--sort-by", help="Sort tasks by this field"
    ),
):
    """
    Lists tasks, optionally including completed tasks or sorting by a different field.
    """
    valid_sort_fields = ["priority", "status", "created_at"]
    if sort_by not in valid_sort_fields:
        msg = f"Invalid sort field. Must be one of {valid_sort_fields}."
        typer.echo(msg)
        return msg

    conn = get_connection()
    cur = conn.cursor()
    if show_all:
        sql = f"SELECT id, task_name, priority, status, created_at FROM tasks ORDER BY {sort_by} ASC"
    else:
        sql = f"SELECT id, task_name, priority, status, created_at FROM tasks WHERE status != 'complete' ORDER BY {sort_by} ASC"

    cur.execute(sql)
    tasks = cur.fetchall()
    conn.close()

    result = "Tasks:\n"
    for t in tasks:
        result += (
            f"ID={t[0]}, Name={t[1]}, Priority={t[2]}, Status={t[3]}, Created={t[4]}\n"
        )

    typer.echo(result.strip())
    return result


# -----------------------------------------------------
# 30) inspect_task
# -----------------------------------------------------
@app.command()
def inspect_task(
    task_id: str = typer.Argument(..., help="ID of the task to inspect"),
    json_output: bool = typer.Option(
        False, "--json", help="Show output in JSON format"
    ),
):
    """
    Inspects a specific task by ID, optionally in JSON format.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, task_name, priority, status, created_at FROM tasks WHERE id = ?",
        (task_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        msg = f"No task found with ID {task_id}."
        typer.echo(msg)
        return msg

    task_dict = {
        "id": row[0],
        "task_name": row[1],
        "priority": row[2],
        "status": row[3],
        "created_at": row[4],
    }

    if json_output:
        result = json.dumps(task_dict, indent=2)
    else:
        result = f"Task ID={task_dict['id']}, Name={task_dict['task_name']}, Priority={task_dict['priority']}, Status={task_dict['status']}, Created={task_dict['created_at']}"
    typer.echo(result)
    return result


# -----------------------------------------------------
# Web Search commands
# -----------------------------------------------------
from modules.web_search import search as web_search_fn, search_news, get_answer
from modules import api_integrations


# -----------------------------------------------------
# Weather (Open-Meteo)
# -----------------------------------------------------
@app.command()
def weather(
    location: str = typer.Argument(None, help="City or place name (e.g. London, Tokyo)"),
    latitude: float = typer.Option(None, "--lat", help="Latitude (optional with location)"),
    longitude: float = typer.Option(None, "--lon", help="Longitude (optional with location)"),
):
    """
    Gets weather forecast from Open-Meteo for a location or coordinates.
    """
    lat = latitude if latitude is not None else 40.7128
    lon = longitude if longitude is not None else -74.0060
    result = api_integrations.open_meteo_weather(latitude=lat, longitude=lon, location=location)
    typer.echo(result)
    return result


# -----------------------------------------------------
# Datamuse words
# -----------------------------------------------------
@app.command()
def datamuse_words(
    means_like: str = typer.Option(None, "--ml", "--means-like", help="Words with similar meaning"),
    sounds_like: str = typer.Option(None, "--sl", "--sounds-like", help="Words that sound similar"),
    spelled_like: str = typer.Option(None, "--sp", "--spelled-like", help="Words spelled similarly"),
    related_to: str = typer.Option(None, "--rel", "--related", help="Words related to this"),
    max_results: int = typer.Option(10, "--max", help="Max results"),
):
    """
    Queries Datamuse API for words by meaning, sound, spelling, or relation.
    """
    result = api_integrations.datamuse_words(
        means_like=means_like,
        sounds_like=sounds_like,
        spelled_like=spelled_like,
        related_to=related_to,
        max_results=max_results,
    )
    typer.echo(result)
    return result


# -----------------------------------------------------
# DeepL translation
# -----------------------------------------------------
@app.command()
def translate(
    text: str = typer.Argument(..., help="Text to translate"),
    target_lang: str = typer.Option("EN", "--to", help="Target language code (e.g. EN, DE, ES)"),
    source_lang: str = typer.Option(None, "--from-lang", help="Source language code (auto if omitted)"),
):
    """
    Translates text using DeepL. Requires DEEPL_API_KEY.
    """
    result = api_integrations.deepl_translate(text, target_lang=target_lang, source_lang=source_lang)
    typer.echo(result)
    return result


# -----------------------------------------------------
# Wolfram Alpha
# -----------------------------------------------------
@app.command()
def wolfram(
    query: str = typer.Argument(..., help="Query to send to Wolfram Alpha"),
):
    """
    Queries Wolfram Alpha for computation/knowledge. Requires WOLFRAM_APP_ID.
    """
    result = api_integrations.wolfram_query(query)
    typer.echo(result)
    return result


# -----------------------------------------------------
# Todoist
# -----------------------------------------------------
@app.command()
def todoist_list(
    project_id: str = typer.Option(None, "--project", help="Filter by project ID"),
):
    """
    Lists Todoist tasks. Requires TODOIST_API_TOKEN.
    """
    result = api_integrations.todoist_list_tasks(project_id=project_id)
    typer.echo(result)
    return result


@app.command()
def todoist_add(
    content: str = typer.Argument(..., help="Task content"),
    project_id: str = typer.Option(None, "--project", help="Project ID"),
):
    """
    Adds a Todoist task. Requires TODOIST_API_TOKEN.
    """
    result = api_integrations.todoist_add_task(content, project_id=project_id)
    typer.echo(result)
    return result


# -----------------------------------------------------
# Local Plane / n8n
# -----------------------------------------------------
@app.command()
def plane_status(
    base_url: str = typer.Option("http://localhost:8000", "--url", help="Plane API base URL"),
):
    """
    Checks status of local Plane API instance.
    """
    result = api_integrations.plane_status(base_url=base_url)
    typer.echo(result)
    return result


@app.command()
def n8n_status(
    base_url: str = typer.Option("http://localhost:5678", "--url", help="n8n base URL"),
):
    """
    Checks status of local n8n instance.
    """
    result = api_integrations.n8n_status(base_url=base_url)
    typer.echo(result)
    return result


# -----------------------------------------------------
# Web Search commands
# -----------------------------------------------------
@app.command()
def web_search(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(5, "--max-results", help="Maximum number of results"),
):
    """
    Searches the web using DuckDuckGo and returns results.
    """
    result = web_search_fn(query, max_results=max_results)
    typer.echo(result)
    return result


@app.command()
def web_news(
    query: str = typer.Argument(..., help="News search query"),
    max_results: int = typer.Option(5, "--max-results", help="Maximum number of results"),
):
    """
    Searches DuckDuckGo news for a query.
    """
    result = search_news(query, max_results=max_results)
    typer.echo(result)
    return result


@app.command()
def web_answer(
    query: str = typer.Argument(..., help="Question to look up"),
):
    """
    Gets an instant answer from DuckDuckGo, falls back to web search.
    """
    result = get_answer(query)
    typer.echo(result)
    return result


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
def main():
    app()


if __name__ == "__main__":
    main()
