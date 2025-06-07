# QuizGPT

<img src="https://github.com/user-attachments/assets/228a34c9-d5c7-45db-85d5-834c2d7b9681" width="400" />

<br>
<br>

## Description

A quiz maker which generates a set of testsheet for the given file or topic. Used function call to format output.

<br>

## The Structure of Chain

> - **Question prompt + LLM(gpt-4.1-nano-2025-04-14) binding Function Call**

<br>

## Function for Formating Function Call

```
format = {
            "name": "create_quiz",
            "description": "function that takes a list of questions and answers and returns a quiz",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                },
                                "answers": {
                                    "type": "array",
                                    ...
                                },
                               ...
                            },
                            "required": ["question", "answers"],
                            },
                        }
                    },
                    "required": ["questions"],
            },
}
```

<br>
<br>
<br>
