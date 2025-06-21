# Chat Completions Visualization App

A comprehensive Flask-based web application for visualizing and analyzing chat completion conversations from JSONL files.

## Features

### 🔍 **Browse & Search**
- View all JSONL files with conversation counts and statistics
- Search conversations by content
- Sort conversations by ID, message count, or content length
- Filter and navigate through large datasets

### 💬 **Conversation Analysis**
- **Thought/Action Parsing**: Automatically separates assistant responses into thought processes and actions
- **Role-based Styling**: Different visual styles for system, user, assistant, and tool messages
- **Message Navigation**: Quick navigation between messages with anchor links
- **Content Management**: Collapsible long messages for better readability

### 📊 **Rich Statistics**
- Overview statistics across all files
- Message role distribution with visual charts
- Message and conversation length analytics
- Export capabilities (JSON, CSV)

### 🎨 **Beautiful UI**
- Modern Bootstrap-based design
- Responsive layout for mobile and desktop
- Syntax highlighting for code blocks
- Smooth animations and transitions
- Dark theme code blocks

## Installation & Setup

### Prerequisites
- Python 3.8+
- Flask
- Required Python packages (see requirements below)

### Quick Start

1. **Navigate to the app directory:**
   ```bash
   cd chat_viz_app
   ```

2. **Install dependencies:**
   ```bash
   pip install flask markdown
   ```

3. **Ensure your JSONL files are in the parent directory:**
   ```
   R2E-Gym/
   ├── chat_completions/
   │   ├── 1.jsonl
   │   ├── 2.jsonl
   │   └── ...
   └── chat_viz_app/
       ├── app.py
       ├── templates/
       └── README.md
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:5761`

## Usage Guide

### Navigation Flow

1. **Home Page** (`/`) - Overview of all JSONL files
2. **File View** (`/file/<filename>`) - List of conversations in a specific file
3. **Conversation Detail** (`/conversation/<filename>/<conv_id>`) - Detailed view of a single conversation
4. **Statistics** (`/stats`) - Comprehensive analytics across all files

### Key Features

#### Thought/Action Parsing
The app automatically parses assistant messages to separate:
- **Thought Process**: Everything before the first `<function=...>` block
- **Action**: The entire `<function=...></function>` block

Example:
```
Input: "I need to search for the file. <function=search><parameter=query>example.py</parameter></function>"

Parsed:
- Thought: "I need to search for the file."
- Action: "<function=search><parameter=query>example.py</parameter></function>"
```

#### Message Types
- 🔧 **System**: Configuration and instructions (gray header)
- 👤 **User**: User inputs and queries (blue header)
- 🤖 **Assistant**: AI responses with thought/action parsing (green header)
- 🛠️ **Tool**: Tool outputs and results (yellow header)

#### Navigation Features
- **Keyboard Shortcuts**: 
  - `Ctrl + ←`: Previous conversation
  - `Ctrl + →`: Next conversation
- **Quick Jump**: Click message role buttons to jump to specific messages
- **Copy Code**: Click copy button on code blocks

## File Structure

```
chat_viz_app/
├── app.py                 # Main Flask application
├── templates/
│   ├── base.html         # Base template with styling
│   ├── index.html        # Home page template
│   ├── file.html         # File conversations list
│   ├── conversation.html # Detailed conversation view
│   └── stats.html        # Statistics dashboard
└── README.md             # This file
```

## Expected Data Format

The app expects JSONL files where each line contains a JSON array of OpenAI-format messages:

```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant..."
  },
  {
    "role": "user", 
    "content": "Please help me with..."
  },
  {
    "role": "assistant",
    "content": "I'll help you. <function=search><parameter=query>example</parameter></function>"
  }
]
```

### Supported Message Roles
- `system`: System prompts and configuration
- `user`: User messages and queries
- `assistant`: AI assistant responses
- `tool`: Tool execution results

## Customization

### Modifying the Parser
To customize thought/action parsing, modify the `parse_response()` function in `app.py`:

```python
def parse_response(response_text: str):
    # Custom parsing logic here
    pattern = re.compile(r"your_custom_pattern")
    # ... rest of implementation
```

### Styling Changes
- Edit `templates/base.html` for global styles
- Modify CSS classes for different message types
- Customize color schemes and animations

### Adding New Features
- Add new routes in `app.py`
- Create corresponding templates
- Extend the statistics collection logic

## Troubleshooting

### Common Issues

1. **No Files Found**
   - Ensure JSONL files exist in `../chat_completions/` relative to the app
   - Check file permissions

2. **Import Errors**
   - Install missing dependencies: `pip install flask markdown`
   - Ensure Python 3.8+ is being used

3. **Port Already in Use**
   - Change the port in `app.py`: `app.run(port=5762, host="0.0.0.0", debug=True)`

4. **Large Files Loading Slowly**
   - The app caches conversations in memory for better performance
   - For very large files, consider implementing pagination

### Performance Tips

- **Memory Usage**: Large JSONL files are cached in memory. Monitor usage for very large datasets.
- **Loading Speed**: First load of each file may be slow as conversations are parsed and cached.
- **Browser Performance**: Very long conversations (>100 messages) may affect browser rendering.

## Contributing

To extend the application:

1. **Adding New Views**: Create new routes in `app.py` and corresponding templates
2. **Enhancing Statistics**: Modify the `statistics()` function to include new metrics
3. **Improving Parsing**: Extend the `parse_response()` function for different message formats
4. **UI Improvements**: Update templates and CSS for better user experience

## Technical Details

### Dependencies
- **Flask**: Web framework
- **Markdown**: For rendering system prompts
- **Bootstrap 5**: UI framework
- **Font Awesome**: Icons
- **Highlight.js**: Code syntax highlighting

### Browser Compatibility
- Modern browsers (Chrome 90+, Firefox 88+, Safari 14+)
- Mobile responsive design
- Progressive enhancement for older browsers

### Security Considerations
- Input validation for filename parameters
- No direct file system access through URLs
- JSONL files are parsed safely

## License

This application is part of the R2E-Gym project. See the main project license for details. 