# 💻 Code Practice Feature

## Overview

The Code Practice feature is a new tab in the LLM Curriculum Assistant that allows students to practice programming skills with AI-generated problems and personalized feedback.

## Features

### 🎯 Problem Types

1. **Create Practice Problems**
   - Students write code from scratch based on problem descriptions
   - Covers various programming concepts (loops, functions, variables, etc.)

2. **Debug - Identify Error Type**
   - Students analyze buggy code to identify what type of error it contains
   - Helps develop debugging skills and error recognition

3. **Debug - Explain Error Reason**
   - Students explain why errors occur in given code
   - Builds understanding of error causes and programming logic

4. **Debug - Fix the Error**
   - Students fix buggy code and provide corrected solutions
   - Practical debugging experience with real code issues

5. **Optimize Code Performance**
   - Students improve existing code for better performance
   - Focus on efficiency, loops, algorithms, and best practices

### 🎨 User Interface

The Code Practice tab features:

- **Topic Input**: Students specify what they want to practice (e.g., "for loops", "functions", "recursion")
- **Problem Type Dropdown**: Choose from 5 different practice modes
- **Generate Problem Button**: Creates a new practice problem using LLM
- **Problem Description**: Clear instructions and requirements with formatted example outputs
- **Code Editor**: VS Code-style editor for writing solutions
- **Student Code Input**: Dedicated editor for student solutions
- **Analyze Button**: Execute code and get AI feedback
- **Terminal Output**: Shows actual code execution results and errors
- **Analysis Output**: Detailed feedback with suggestions and explanations

### 🤖 AI-Powered Features

- **Curriculum-Based Problem Generation**: LLM creates problems based on actual curriculum content and difficulty level
- **Intelligent Code Analysis**: Provides detailed feedback that references curriculum concepts and standards
- **Personalized Feedback**: Encouraging but honest assessment aligned with curriculum learning objectives
- **Educational Explanations**: Helps students understand concepts based on what they've learned in class

## How to Use

1. **Navigate to Code Practice Tab**
   - Click on the "💻 Code Practice" tab in the main interface

2. **Set Up Your Practice Session**
   - Enter a topic you want to practice (e.g., "for loops", "recursion", "arrays")
   - Select a problem type from the dropdown menu
   - Click "🎲 Generate Problem" to get a new practice problem

3. **Work on the Problem**
   - Read the problem description carefully
   - Use the code editor to write your solution
   - The editor supports Python syntax highlighting

4. **Get Feedback**
   - Click "🔍 Analyze My Code" to receive AI feedback
   - Review the detailed analysis and suggestions
   - Use the feedback to improve your solution

5. **Practice More**
   - Generate new problems to continue practicing
   - Try different problem types to develop various skills

## Technical Implementation

### CodePracticeAssistant Class

The new `CodePracticeAssistant` class handles:

- **Curriculum Integration**: Uses the curriculum assistant's vector database to find relevant content
- **LLM Integration**: Uses Claude 3.5 Haiku for problem generation and code analysis
- **Problem Generation**: Creates problems based on actual curriculum content and difficulty level
- **Code Analysis**: Provides comprehensive feedback that references curriculum concepts
- **Error Handling**: Graceful handling of API errors and edge cases

### Key Methods

- `generate_practice_problem(topic, problem_type)`: Creates new practice problems based on curriculum content
- `analyze_student_code(topic, problem_type, problem_description, student_code)`: Analyzes student solutions with curriculum context
- `execute_code(student_code)`: Safely executes student code and returns terminal output
- `_find_curriculum_content(topic)`: Finds relevant curriculum content for the given topic

### UI Components

- **Gradio Tabs**: Organized interface with separate tabs for chat and practice
- **Code Editors**: Syntax-highlighted code input areas
- **Markdown Outputs**: Formatted problem descriptions and analysis
- **Interactive Buttons**: Generate and analyze functionality

## Benefits for Students

1. **Personalized Learning**: Practice specific topics at your own pace
2. **Immediate Feedback**: Get instant analysis of your code
3. **Multiple Skill Development**: Practice writing, debugging, and optimizing code
4. **Safe Learning Environment**: Make mistakes and learn from AI feedback
5. **Progressive Difficulty**: Problems adapt to your skill level

## Integration with Existing Features

The Code Practice feature seamlessly integrates with the existing curriculum assistant:

- **Shared LLM Infrastructure**: Uses the same Claude API for consistency
- **Unified Interface**: Both features accessible from the same application
- **Consistent Experience**: Similar UI patterns and interaction models
- **Complementary Learning**: Chat for questions, Practice for hands-on coding

## Future Enhancements

Potential improvements could include:

- **Difficulty Levels**: Beginner, Intermediate, Advanced options
- **Progress Tracking**: Save completed problems and track improvement
- **Hints System**: Get progressive hints when stuck
- **Code Execution**: Run code and see actual output
- **Collaborative Features**: Share problems with classmates
- **Custom Problem Creation**: Teachers can create custom problems

## Requirements

- Python 3.7+
- Anthropic API key (ANTHROPIC_KEY environment variable)
- All existing dependencies from requirements.txt

## Usage Example

```python
# Initialize the practice assistant
practice_assistant = CodePracticeAssistant()

# Generate a practice problem
problem, starter_code = practice_assistant.generate_practice_problem(
    topic="for loops", 
    problem_type="Create Practice Problems"
)

# Analyze student code
analysis = practice_assistant.analyze_student_code(
    topic="for loops",
    problem_type="Create Practice Problems", 
    problem_description=problem,
    student_code="def sum_list(nums):\n    return sum(nums)"
)
```

The Code Practice feature transforms the curriculum assistant into a comprehensive learning platform that combines theoretical knowledge with practical coding experience. 