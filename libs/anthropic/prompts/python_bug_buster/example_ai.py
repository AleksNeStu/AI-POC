import anthropic
from anthropic.types import TextBlock

from common.cfg import *
from common import helper

text = helper.read_file("example.py")


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=ANTHROPIC_API_KEY,
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    system="Your task is to analyze the provided Python code snippet, identify any bugs or errors present, and provide a corrected version of the code that resolves these issues. Explain the problems you found in the original code and how your fixes address them. The corrected code should be functional, efficient, and adhere to best practices in Python programming.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ]
        }
    ]
)
act = message.content
print(act)


exp = [TextBlock(text='The provided Python code snippet is correct and does not contain any bugs or errors. Let\'s '
                        'go '
                 'through the code to understand how it works:\n\n```python\ndef calculate_average(nums):\n    total = 0\n    for num in nums:\n        total += num\n    average = total / len(nums)\n    return average\n```\n\n1. The `calculate_average` function takes a list of numbers (`nums`) as input.\n2. It initializes a variable `total` to 0, which will store the sum of all the numbers.\n3. It uses a `for` loop to iterate over each number (`num`) in the `nums` list.\n4. Inside the loop, it adds each number to the `total` variable using the `+=` operator.\n5. After the loop, it calculates the average by dividing the `total` by the length of the `nums` list using `len(nums)`.\n6. Finally, it returns the calculated average.\n\n```python\nnumbers = [10, 20, 30, 40, 50]\nresult = calculate_average(numbers)\nprint("The average is:", result)\n```\n\n7. The code creates a list of numbers called `numbers` with the values `[10, 20, 30, 40, 50]`.\n8. It calls the `calculate_average` function, passing the `numbers` list as an argument, and assigns the returned value to the `result` variable.\n9. Finally, it prints the average using the `print` statement.\n\nThe output of this code will be:\n```\nThe average is: 30.0\n```\n\nThe code is functional, efficient, and adheres to best practices in Python programming. It correctly calculates the average of the numbers in the list and prints the result.', type='text')]