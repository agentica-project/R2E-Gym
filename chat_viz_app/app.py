import os
import json
import re
from flask import Flask, render_template, request, redirect, url_for, abort
from collections import defaultdict
import markdown

app = Flask(__name__)

# Path to the directory containing JSONL files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
CHAT_DIR = os.path.join(PARENT_DIR, "chat_completions")

def get_jsonl_files():
    """Retrieve all JSONL files in the CHAT_DIR directory."""
    try:
        files = [f for f in os.listdir(CHAT_DIR) if f.endswith(".jsonl")]
        return sorted(files, key=lambda x: int(x.split('.')[0]))
    except FileNotFoundError:
        return []

def parse_response(response_text: str):
    """
    Parse assistant response into thought and action components.
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)
    
    if match:
        action = match.group(1)  # The entire <function=...></function> block
        thought = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        thought = response_text
        action = ""
    
    # Strip leading/trailing whitespace
    thought = thought.strip()
    action = action.strip()
    
    return thought, action

# Cache for loaded conversations
conversations_cache = {}

def load_conversations(filename):
    """Load all conversations from a JSONL file."""
    if filename in conversations_cache:
        return conversations_cache[filename]
    
    filepath = os.path.join(CHAT_DIR, filename)
    if not os.path.exists(filepath):
        return []
    
    conversations = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line_idx, line in enumerate(file):
            try:
                conversation = json.loads(line.strip())
                # Add metadata
                conversation_data = {
                    'id': line_idx,
                    'messages': conversation,
                    'message_count': len(conversation),
                    'filename': filename
                }
                conversations.append(conversation_data)
            except json.JSONDecodeError:
                continue
    
    conversations_cache[filename] = conversations
    return conversations

def get_conversation_summary(conversation):
    """Extract a summary from the conversation for display."""
    messages = conversation['messages']
    
    # Find the first user message (usually contains the problem description)
    user_message = None
    for msg in messages:
        if msg['role'] == 'user':
            user_message = msg['content']
            break
    
    if user_message:
        # Extract first few lines or characters for summary
        summary = user_message[:200] + "..." if len(user_message) > 200 else user_message
        # Try to extract GitHub issue title if present
        if "Consider the following github issue" in user_message.lower():
            lines = user_message.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('#') and len(line.strip()) > 10:
                    summary = line.strip()[:150] + "..." if len(line.strip()) > 150 else line.strip()
                    break
    else:
        summary = "No user message found"
    
    return summary

@app.route("/")
def index():
    """Home page: list all JSONL files with conversation counts."""
    files = get_jsonl_files()
    file_info = []
    
    for filename in files:
        conversations = load_conversations(filename)
        file_info.append({
            'filename': filename,
            'conversation_count': len(conversations),
            'total_messages': sum(conv['message_count'] for conv in conversations)
        })
    
    return render_template("index.html", files=file_info)

@app.route("/file/<filename>")
def file_conversations(filename):
    """Display all conversations in a file."""
    if ".." in filename or filename.startswith("/"):
        abort(400, description="Invalid filename.")
    
    conversations = load_conversations(filename)
    if not conversations:
        abort(404, description="File not found or empty.")
    
    # Add summaries to conversations
    for conv in conversations:
        conv['summary'] = get_conversation_summary(conv)
    
    return render_template("file.html", filename=filename, conversations=conversations)

@app.route("/conversation/<filename>/<int:conv_id>")
def conversation_detail(filename, conv_id):
    """Display detailed view of a single conversation."""
    if ".." in filename or filename.startswith("/"):
        abort(400, description="Invalid filename.")
    
    conversations = load_conversations(filename)
    if not conversations or conv_id < 0 or conv_id >= len(conversations):
        abort(404, description="Conversation not found.")
    
    conversation = conversations[conv_id]
    messages = conversation['messages']
    
    # Process messages for better display
    processed_messages = []
    for i, msg in enumerate(messages):
        processed_msg = {
            'role': msg['role'],
            'content': msg['content'],
            'index': i
        }
        
        # Parse assistant messages for thought/action
        if msg['role'] == 'assistant':
            thought, action = parse_response(msg['content'])
            processed_msg['thought'] = thought
            processed_msg['action'] = action
            processed_msg['has_action'] = bool(action)
        
        # Convert markdown content for system messages
        if msg['role'] == 'system':
            processed_msg['content_html'] = markdown.markdown(
                msg['content'], extensions=["fenced_code", "nl2br"]
            )
        
        processed_messages.append(processed_msg)
    
    # Navigation info
    total_conversations = len(conversations)
    
    return render_template(
        "conversation.html",
        filename=filename,
        conversation=conversation,
        messages=processed_messages,
        current_conv_id=conv_id,
        total_conversations=total_conversations
    )

@app.route("/stats")
def statistics():
    """Display statistics across all files."""
    files = get_jsonl_files()
    stats = {
        'total_files': len(files),
        'total_conversations': 0,
        'total_messages': 0,
        'role_counts': defaultdict(int),
        'message_length_stats': [],
        'conversation_length_stats': []
    }
    
    all_message_lengths = []
    all_conversation_lengths = []
    
    for filename in files:
        conversations = load_conversations(filename)
        stats['total_conversations'] += len(conversations)
        
        for conv in conversations:
            messages = conv['messages']
            stats['total_messages'] += len(messages)
            all_conversation_lengths.append(len(messages))
            
            for msg in messages:
                stats['role_counts'][msg['role']] += 1
                all_message_lengths.append(len(msg['content']))
    
    # Calculate statistics
    if all_message_lengths:
        stats['message_length_stats'] = {
            'min': min(all_message_lengths),
            'max': max(all_message_lengths),
            'avg': sum(all_message_lengths) / len(all_message_lengths),
            'median': sorted(all_message_lengths)[len(all_message_lengths) // 2]
        }
    
    if all_conversation_lengths:
        stats['conversation_length_stats'] = {
            'min': min(all_conversation_lengths),
            'max': max(all_conversation_lengths),
            'avg': sum(all_conversation_lengths) / len(all_conversation_lengths),
            'median': sorted(all_conversation_lengths)[len(all_conversation_lengths) // 2]
        }
    
    return render_template("stats.html", stats=stats)

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return f"<h1>404 Not Found</h1><p>{error.description}</p>", 404

@app.errorhandler(400)
def bad_request(error):
    return f"<h1>400 Bad Request</h1><p>{error.description}</p>", 400

if __name__ == "__main__":
    app.run(port=5761, host="0.0.0.0", debug=True) 