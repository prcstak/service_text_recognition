from taipy import Gui
from taipy.gui import notify
from pipeline import process_image

page = """
# Text Recognition App

<|{content}|file_selector|label= Upload Image|drop_message=Drop here to Upload|extensions=.png,.jpg|>
---
<|{content}|image|width=50%|>
---
<|Recognize text|button|class_name=plain|on_action=on_button_action|>
<|{status_text}|>

<|{result_text}|>
"""

content = ""
result_text = ""
status_text = ""


def on_button_action(state):
    notify(state, 'info', f'Process started')
    state.status_text = "InProgress"
    result = process_image(state.content)
    notify(state, 'success', f'Process completed')
    state.status_text = "Completed"
    state.result_text = result


Gui(page).run(use_reloader=True, dark_mode=False)
